# =============================================================================
# main.py – Bot “Gabi” · WhatsApp · OpenAI · Google Calendar
# =============================================================================
import os, io, json, logging, asyncio, mimetypes, tempfile, re, uuid, enum
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Query, Body
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv
import httpx, openai
from pydub import AudioSegment
from zoneinfo import ZoneInfo

from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build

from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Enum, JSON, func
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession

# -----------------------------------------------------------------------------
# CONFIG
# -----------------------------------------------------------------------------
load_dotenv()
VERIFY_TOKEN           = os.getenv("VERIFY_TOKEN")
WHATSAPP_ACCESS_TOKEN  = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_ID      = os.getenv("WHATSAPP_PHONE_ID")
WHATSAPP_API_URL       = f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/messages"
OPENAI_API_KEY         = os.getenv("OPENAI_API_KEY")
DATABASE_URL           = os.getenv("DATABASE_URL")

BASE_DIR   = Path(__file__).resolve().parent
TOKEN_DIR  = BASE_DIR / "token"; TOKEN_DIR.mkdir(exist_ok=True)
OAUTH_FILE = Path(os.getenv("GOOGLE_OAUTH_FILE", TOKEN_DIR / "dsm_oauth.json"))
OWNER_CALENDAR_ID = os.getenv("OWNER_CALENDAR_ID", "primary")
CALENDAR_ID       = os.getenv("GOOGLE_CALENDAR_ID", OWNER_CALENDAR_ID)

GOOGLE_SERVICE_ACCOUNT_CONTENT = os.getenv("GOOGLE_SERVICE_ACCOUNT_CONTENT", "{}")
PDF_LOCAL_PATH = os.path.join(os.path.dirname(__file__), "presentation", "dsm.pdf")

# Número do atendente humano (WhatsApp)
HUMAN_NUMBER = "+556182182423"

client = openai.OpenAI(api_key=OPENAI_API_KEY)
app    = FastAPI()
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

# -----------------------------------------------------------------------------
# GOOGLE TOKEN ÚNICO
# -----------------------------------------------------------------------------
_owner_creds: Optional[Credentials] = None
def owner_creds():
    global _owner_creds
    if _owner_creds:
        return _owner_creds
    if not OAUTH_FILE.exists():
        raise RuntimeError(f"token Google ausente em {OAUTH_FILE}")
    _owner_creds = Credentials.from_authorized_user_info(
        json.load(OAUTH_FILE.open()), ["https://www.googleapis.com/auth/calendar"])
    return _owner_creds

def gservice():
    return build("calendar", "v3", credentials=owner_creds(), cache_discovery=False)

# -----------------------------------------------------------------------------
# DATABASE
# -----------------------------------------------------------------------------
Base = declarative_base()

class FlowType(enum.Enum):
    roteiro = "roteiro"
    conversa = "conversa"

class User(Base):
    __tablename__ = "users"
    phone = Column(String(20), primary_key=True)
    name = Column(String(100))
    email = Column(String(100))
    data_nascimento = Column(String(20))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())

class ConversationSession(Base):
    __tablename__ = "conversation_sessions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_phone = Column(String(20), ForeignKey("users.phone"))
    flow = Column(Enum(FlowType))
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime)
    session_data = Column(JSON)
    raw_payload = Column(JSON)

engine = create_async_engine(DATABASE_URL, echo=False)
SessionAsync = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

async def update_user_email(phone, email):
    async with SessionAsync() as s:
        u = await s.get(User, phone)
        if u:
            u.email = email
            await s.commit()

async def store_viagem(session_id: int, detalhes: Dict[str, str]):
    async with SessionAsync() as s:
        cs = await s.get(ConversationSession, session_id)
        if cs:
            data = cs.session_data or {}
            data["viagem"] = detalhes
            cs.session_data = data
            await s.commit()

# -----------------------------------------------------------------------------
# ESTADOS & UTILS
# -----------------------------------------------------------------------------
sessions: Dict[str, Dict[str, Any]] = {}

def branch(nome):
    return (
        f"Olá {nome}, como posso te ajudar?\n"
        "1️⃣  Quer um orçamento?\n"
        "2️⃣  Quer conhecer a DSM?\n"
        "3️⃣  Gostaria de tirar uma dúvida?"
    )

def roteiro_q1():
    return "Você já tem uma viagem em mente?\n1️⃣  Sim\n2️⃣  Não"

def roteiro_q_det():
    return (
        "Legal! Em que estágio você está?\n"
        "1️⃣ Sei lugar e data\n"
        "2️⃣ Só sei o lugar\n"
        "3️⃣ Só sei a data\n"
        "4️⃣ Não pensei ainda"
    )

def roteiro_q_exemplo(opt: str):
    if opt == "1":
        return (
            "Ótimo! Envie no formato:\n"
            "*Destino*: Paris\n*Datas*: 12/09/2025 a 20/09/2025"
        )
    if opt == "2":
        return (
            "Beleza! Envie no formato:\n"
            "*Destino*: Paris\n*Datas*: flexível"
        )
    if opt == "3":
        return (
            "Certo! Envie no formato:\n"
            "*Destino*: flexível\n*Datas*: 12/09/2025 a 20/09/2025"
        )
    return "Conte-me como posso ajudar:"

def roteiro_q_agenda():
    return "Você gostaria de marcar uma reunião com nosso time?\n1️⃣  Sim\n2️⃣  Não"

email_re = re.compile(r"^[\w\.-]+@[\w\.-]+\.[A-Za-z]{2,}$")

# -----------------------------------------------------------------------------
# OpenAI parser para detalhes da viagem
# -----------------------------------------------------------------------------
json_block = re.compile(r"\{.*\}", re.DOTALL)

async def parse_viagem(texto: str) -> Dict[str, str]:
    prompt = (
        "Extraia destino, data_inicio e data_fim da mensagem do usuário.\n"
        "Se não houver data_inicio ou data_fim, sugira baseado no contexto.\n"
        "Responda SOMENTE JSON com as chaves destino,data_inicio,data_fim.\n\n"
        f"Mensagem: '''{texto}'''"
    )
    try:
        r = await httpx.AsyncClient().post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 60,
            },
            timeout=10,
        )
        content = r.json()["choices"][0]["message"]["content"].strip()
        m = json_block.search(content)
        if not m:
            raise ValueError("JSON não encontrado")
        js = json.loads(m.group(0))
        return {k: js.get(k, "") for k in ("destino", "data_inicio", "data_fim")}
    except Exception as e:
        logger.error("parse_viagem falhou: %s", e, exc_info=True)
        return {"destino": "", "data_inicio": "", "data_fim": ""}

# -----------------------------------------------------------------------------
# WhatsApp helpers
# -----------------------------------------------------------------------------
async def wpp(dest, body):
    await httpx.AsyncClient().post(
        WHATSAPP_API_URL,
        headers={"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"},
        json={
            "messaging_product": "whatsapp",
            "to": dest.lstrip("+"),
            "type": "text",
            "text": {"body": body, "preview_url": False},
        },
        timeout=10,
    )

async def wpp_doc(dest, path, caption="Apresentação DSM"):
    mime = mimetypes.guess_type(path)[0] or "application/pdf"
    up = await httpx.AsyncClient().post(
        f"https://graph.facebook.com/v18.0/{WHATSAPP_PHONE_ID}/media",
        data={"messaging_product": "whatsapp", "type": mime},
        files={"file": (os.path.basename(path), open(path, "rb"), mime)},
        headers={"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"},
        timeout=30,
    )
    mid = up.json()["id"]
    await httpx.AsyncClient().post(
        WHATSAPP_API_URL,
        headers={"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"},
        json={
            "messaging_product": "whatsapp",
            "to": dest.lstrip("+"),
            "type": "document",
            "document": {"id": mid, "caption": caption},
        },
        timeout=10,
    )

# -----------------------------------------------------------------------------
# Google Calendar helpers
# -----------------------------------------------------------------------------
def free_slots() -> List[Dict[str, str]]:
    tz, now = ZoneInfo("America/Sao_Paulo"), datetime.now(ZoneInfo("America/Sao_Paulo"))
    sa = service_account.Credentials.from_service_account_info(
        json.loads(GOOGLE_SERVICE_ACCOUNT_CONTENT),
        scopes=["https://www.googleapis.com/auth/calendar.readonly"],
    )
    svc = build("calendar", "v3", credentials=sa, cache_discovery=False)
    fb = svc.freebusy().query(
        body={
            "timeMin": now.isoformat(),
            "timeMax": (now + timedelta(days=14)).isoformat(),
            "timeZone": "America/Sao_Paulo",
            "items": [{"id": CALENDAR_ID}],
        }
    ).execute()
    busy = [
        (
            datetime.fromisoformat(b["start"].replace("Z", "+00:00")).astimezone(tz),
            datetime.fromisoformat(b["end"].replace("Z", "+00:00")).astimezone(tz),
        )
        for b in fb["calendars"][CALENDAR_ID]["busy"]
    ]
    allowed = ["09:00", "10:00", "13:00", "14:00", "16:00", "17:00"]
    dur = timedelta(hours=1)
    slots = []
    for i in range(14):
        d = now + timedelta(days=i)
        if d.weekday() not in (1, 3):
            continue
        day_busy = [(s, e) for s, e in busy if s.date() == d.date()]
        for h in allowed:
            hr, mn = map(int, h.split(":"))
            start = d.replace(hour=hr, minute=mn, second=0, microsecond=0)
            end = start + dur
            if any(start < b_end and end > b_start for b_start, b_end in day_busy):
                continue
            slots.append({"id": len(slots) + 1, "data": start.strftime("%Y-%m-%d"), "hora": h})
    return slots

def create_event(data: Dict[str, str], slot: Dict[str, str]) -> str:
    svc = gservice()
    ev = {
        "summary": f"Reunião DSM – {data['nome']}",
        "start": {"dateTime": f"{slot['data']}T{slot['hora']}:00", "timeZone": "America/Sao_Paulo"},
        "end": {
            "dateTime": f"{slot['data']}T{int(slot['hora'][:2]) + 1:02d}:{slot['hora'][3:]}:00",
            "timeZone": "America/Sao_Paulo",
        },
        "attendees": [{"email": data["email"]}],
        "conferenceData": {"createRequest": {"requestId": str(uuid.uuid4())}},
    }
    evc = svc.events().insert(
        calendarId=CALENDAR_ID, body=ev, conferenceDataVersion=1
    ).execute()
    return evc.get("hangoutLink") or next(
        (ep["uri"] for ep in evc["conferenceData"]["entryPoints"] if ep["entryPointType"] == "video"),
        "Link não informado",
    )

# -----------------------------------------------------------------------------
# AGENDA FLOW (inalterado)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# AGENDA FLOW (ajuste para redirecionar ao atendente humano no “Não”)
# -----------------------------------------------------------------------------
async def agenda_flow(dest, msg):
    s = sessions[dest]

    # Pergunta de agendamento
    if s["phase"] == "agenda_pending":
        if msg.strip() == "1":
            # segue fluxo normal de coleta de e-mail
            s["phase"] = "agenda_email"
            await wpp(dest, "Perfeito! Qual é o seu e-mail?")
            return

        elif msg.strip() == "2":
            # REDIRECIONA para atendimento humano
            await atendente_flow(dest)
            return

        else:
            await wpp(dest, "Responda 1 ou 2.")
            return

    # Coleta de e-mail
    if s["phase"] == "agenda_email":
        if not email_re.match(msg.strip()):
            await wpp(dest, "E-mail inválido.")
            return
        s["data"]["email"] = msg.strip()
        await update_user_email(dest, msg.strip())
        slots = free_slots()
        if not slots:
            await wpp(dest, "Sem horários agora. Tente depois.")
            # se não há slots, também vamos para o humano
            await atendente_flow(dest)
            return
        s.update({"phase": "agenda_choice", "slots": slots})
        txt = "Horários disponíveis:\n" + "\n".join(
            f"{d['id']}: {d['data']} às {d['hora']}" for d in slots
        )
        await wpp(dest, txt + "\nEscolha o número.")
        return

    # Escolha do slot
    if s["phase"] == "agenda_choice":
        try:
            op = int(msg.strip())
        except:
            await wpp(dest, "Digite o número.")
            return
        sl = next((d for d in s["slots"] if d["id"] == op), None)
        if not sl:
            await wpp(dest, "Opção inválida.")
            return
        link = create_event(s["data"], sl)
        await wpp(
            dest,
            f"✅ Reunião marcada!\n📅 {sl['data']} ⏰ {sl['hora']}\n🔗 Meet: {link}\n\nAté breve! ✈️",
        )
        if (sid := s.get("db_session_id")):
            await store_viagem(sid, s["data"].get("viagem", {}))
        del sessions[dest]
        return


# -----------------------------------------------------------------------------
# DSM FLOW (nova etapa 2)
# -----------------------------------------------------------------------------
async def dsm_flow(dest, msg):
    await wpp_doc(dest, PDF_LOCAL_PATH)
    s = sessions[dest]
    s["phase"] = "agenda_pending"
    await wpp(dest, roteiro_q_agenda())

# -----------------------------------------------------------------------------
# ATENDENTE FLOW (nova etapa 3)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ATENDENTE FLOW (nova etapa 3)
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# ATENDENTE FLOW (etapa 3, ajustado)
# -----------------------------------------------------------------------------
async def atendente_flow(dest):
    # 1) avisa o usuário sobre o atendente humano
    texto = (
        "👍 Um de nossos consultores humanos irá te atender.\n\n"
        f"Para falar agora, envie sua dúvida para: *{HUMAN_NUMBER}* no WhatsApp.\n"
        "Se preferir, aguarde que entraremos em contato por aqui em breve."
    )
    await wpp(dest, texto)

    # 2) envia alerta para o atendente DE FORMA EXPLÍCITA
    alert = (
        f"🚨 *Atendimento solicitado* 🚨\n"
        f"Usuário: {dest}\n"
        "Por favor, entre em contato com ele em breve."
    )
    await httpx.AsyncClient().post(
        WHATSAPP_API_URL,
        headers={"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"},
        json={
            "messaging_product": "whatsapp",
            "to": HUMAN_NUMBER.lstrip("+"),
            "type": "text",
            "text": {"body": alert, "preview_url": False}
        },
        timeout=10,
    )

    # 3) encerra a sessão automatizada
    sessions.pop(dest, None)


# -----------------------------------------------------------------------------
# CHAT (OpenAI) – igual
# -----------------------------------------------------------------------------
async def chat_flow(dest, msg):
    s = sessions[dest]
    s.setdefault("hist", "")
    s["hist"] += f"Usuário: {msg}\n"
    r = await httpx.AsyncClient().post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
        json={
            "model": "gpt-4o",
            "messages": [
                {"role": "system", "content": "Você é um especialista em viagens."},
                {"role": "user", "content": s["hist"]},
            ],
            "temperature": 0.85,
            "max_tokens": 250,
        },
        timeout=10,
    )
    ans = r.json()["choices"][0]["message"]["content"].strip()
    await wpp(dest, ans)
    s["hist"] += f"Bot: {ans}\n"

# -----------------------------------------------------------------------------
# ROTEIRO FLOW – com novos estados coleta & confirmação (inalterado)
# -----------------------------------------------------------------------------
async def roteiro_flow(dest, msg):
    s = sessions[dest]
    st = s.get("state", "viagem")
    ch = msg.strip()

    if st == "viagem":
        if ch not in {"1", "2"}:
            await wpp(dest, "Responda 1 ou 2.")
            return
        s["data"]["tem_viagem"] = "sim" if ch == "1" else "nao"
        if ch == "1":
            s["state"] = "det"
            await wpp(dest, roteiro_q_det())
            return
        s["state"] = "conhece"
        await wpp(dest, "Você já conhece nosso trabalho?\n1️⃣ Sim\n2️⃣ Não")
        return

    if st == "det":
        if ch not in {"1", "2", "3", "4"}:
            await wpp(dest, "Escolha 1-4.")
            return
        s["data"]["det_opt"] = ch
        if ch == "4":
            s["state"] = "conhece"
            await wpp(dest, "Você já conhece nosso trabalho?\n1️⃣ Sim\n2️⃣ Não")
            return
        s["state"] = "coleta"
        await wpp(dest, roteiro_q_exemplo(ch))
        return

    if st == "coleta":
        s["data"]["raw_descricao"] = msg.strip()
        parsed = await parse_viagem(msg.strip())
        s["data"]["viagem"] = parsed
        resumo = (
            f"Destino: {parsed['destino'] or 'flexível'}\n"
            f"Datas : {parsed['data_inicio'] or 'flexível'} até {parsed['data_fim'] or 'flexível'}\n"
            "Essas informações estão corretas?\n1️⃣ Sim\n2️⃣ Não"
        )
        s["state"] = "coleta_confirm"
        await wpp(dest, resumo)
        return

    if st == "coleta_confirm":
        if ch == "1":
            s["state"] = "conhece"
            await wpp(dest, "Você já conhece nosso trabalho?\n1️⃣ Sim\n2️⃣ Não")
            return
        if ch == "2":
            s["state"] = "coleta"
            await wpp(dest, "Tudo bem, envie novamente no formato sugerido.")
            return
        await wpp(dest, "Responda 1 ou 2.")
        return

    if st == "conhece":
        if ch not in {"1", "2"}:
            await wpp(dest, "Responda 1 ou 2.")
            return
        if ch == "2":
            await wpp_doc(dest, PDF_LOCAL_PATH)
        s.update({"state": "agenda_pending", "phase": "agenda_pending"})
        await wpp(dest, roteiro_q_agenda())
        return

    if st in {"agenda_pending", "agenda_email", "agenda_choice"}:
        await agenda_flow(dest, msg)
        return

# -----------------------------------------------------------------------------
# DISPATCHER (tecla 0 reinicia; cria ConversationSession para banco)
# -----------------------------------------------------------------------------
async def handle_text(dest, msg):
    if msg.strip() == "0" and dest in sessions:
        del sessions[dest]

    if dest not in sessions:
        sessions[dest] = {"phase": "common", "data": {}}
        await wpp(dest, "Para começarmos, qual é o seu nome?")
        return

    s = sessions[dest]

    if s["phase"] == "common":
        s["data"]["nome"] = msg.strip()
        await wpp(dest, branch(msg.strip()))
        s["phase"] = "branch"
        return

    if s["phase"] == "branch":
        if msg.strip() not in {"1", "2", "3"}:
            await wpp(dest, "Escolha 1-3.")
            return

        choice = msg.strip()
        if choice == "1":
            s["bot"] = "roteiro"
            s["state"] = "viagem"
        elif choice == "2":
            s["bot"] = "dsm"
        else:  # opção 3
            s["bot"] = "atendente"

        s["phase"] = "flow"

        async with SessionAsync() as db:
            cs = ConversationSession(
                user_phone=dest,
                flow=FlowType.roteiro if s["bot"] == "roteiro" else FlowType.conversa,
                session_data={}, raw_payload={},
            )
            db.add(cs)
            await db.commit()
            await db.refresh(cs)
            s["db_session_id"] = cs.id

        if s["bot"] == "roteiro":
            await wpp(dest, roteiro_q1())
        elif s["bot"] == "dsm":
            await dsm_flow(dest, msg)
        elif s["bot"] == "atendente":
            await atendente_flow(dest)
        else:
            await wpp(dest, "Sobre o que você quer conversar?")
        return

    # continuidade dentro dos fluxos já iniciados
    if s.get("bot") == "roteiro":
        await roteiro_flow(dest, msg)
    elif s.get("bot") == "dsm":
        await agenda_flow(dest, msg)
    else:
        await chat_flow(dest, msg)

# -----------------------------------------------------------------------------
# AUDIO helpers – igual
# -----------------------------------------------------------------------------
async def dl_media(fid, mime):
    h = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    meta = await httpx.AsyncClient().get(
        f"https://graph.facebook.com/v18.0/{fid}", headers=h
    )
    dl = await httpx.AsyncClient().get(meta.json()["url"], headers=h)
    ext = "ogg" if "opus" in mime.lower() else mime.split("/")[-1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as f:
        f.write(dl.content)
        return f.name

async def process_audio(msg, dest):
    p = await dl_media(msg.audio.id, msg.audio.mime_type)
    if os.path.getsize(p) < 2048:
        await wpp(dest, "Áudio muito curto.")
        return
    txt = await asyncio.get_running_loop().run_in_executor(
        None,
        lambda: client.audio.transcriptions.create(
            file=open(p, "rb"), model="whisper-1", response_format="text"
        ),
    )
    os.remove(p)
    await handle_text(dest, txt)

# -----------------------------------------------------------------------------
# Webhook Pydantic & endpoints (certifique-se de que ValuePayload inclui messages)
# -----------------------------------------------------------------------------
class TextPayload(BaseModel):
    body: str

class AudioPayload(BaseModel):
    mime_type: str
    sha256: str
    id: str

class MessagePayload(BaseModel):
    from_: str = Field(..., alias="from")
    id: str
    timestamp: str
    type: str
    text: Optional[TextPayload] = None
    audio: Optional[AudioPayload] = None

class ValuePayload(BaseModel):
    messaging_product: str
    metadata: Dict[str, Any]
    messages: Optional[List[MessagePayload]] = None

class ChangePayload(BaseModel):
    field: str
    value: ValuePayload

class EntryPayload(BaseModel):
    id: str
    changes: List[ChangePayload]

class WebhookPayload(BaseModel):
    object: str
    entry: List[EntryPayload]

@app.exception_handler(RequestValidationError)
async def val_err(req: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

@app.get("/", response_class=PlainTextResponse)
async def verify(
    hub_mode: str = Query(..., alias="hub.mode"),
    hub_challenge: str = Query(..., alias="hub.challenge"),
    hub_verify_token: str = Query(..., alias="hub.verify_token"),
):
    if hub_verify_token != VERIFY_TOKEN:
        raise HTTPException(403, "Token inválido.")
    return hub_challenge

@app.post("/")
async def webhook(raw: dict = Body(...), bg: BackgroundTasks = None):
    try:
        payload = WebhookPayload(**raw)
    except ValidationError:
        return {"status": "ignored"}

    v = payload.entry[0].changes[0].value
    if v.messages:
        m = v.messages[0]
        dest = m.from_
        if m.type == "audio":
            await process_audio(m, dest)
        elif m.type == "text":
            bg.add_task(handle_text, dest, m.text.body.strip())
    return {"status": "ok"}

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")

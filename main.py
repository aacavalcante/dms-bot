from sqlalchemy import Column, String, Integer, DateTime, Text, ForeignKey, Enum, JSON, func, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
import os
import re
import json
import logging
import asyncio
import httpx
import enum
import tempfile
import io
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.responses import PlainTextResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, BinaryIO
from dotenv import load_dotenv
import openai  # pip install openai
from pydub import AudioSegment  # pip install pydub
import random
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo  # disponível no Python 3.9+

# Novos imports para Google Calendar com conta de serviço
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Carrega as variáveis de ambiente
load_dotenv()

# Configurações do ambiente
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "seu_token_de_verificacao")
WHATSAPP_ACCESS_TOKEN = os.getenv(
    "WHATSAPP_ACCESS_TOKEN", "seu_access_token_whatsapp")
WHATSAPP_API_URL = os.getenv(
    "WHATSAPP_API_URL", "https://graph.facebook.com/v22.0/<YOUR_PHONE_NUMBER_ID>/messages")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DATABASE_URL = os.getenv(
    "DATABASE_URL", "mysql+asyncmy://usuario:senha@host:3306/chatai")

# Para o Google Calendar, o ID do calendário e os dados da conta de serviço
GOOGLE_CALENDAR_ID = os.getenv(
    "GOOGLE_CALENDAR_ID", "seu_calendar_id@gmail.com")
# A variável abaixo deve conter o JSON da conta de serviço (tudo em uma linha ou com quebras devidamente escapadas)
# Exemplo: '{"type": "service_account", "project_id": "...", ... }'
# NÃO inclua espaços ou quebras de linha sem o devido escape
# Para teste, você pode colocar o JSON em uma única linha.
# Certifique-se de definir a variável GOOGLE_SERVICE_ACCOUNT_CONTENT no .env.

# Inicializa o client da OpenAI e o FastAPI
client = openai.OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI()
logger = logging.getLogger("uvicorn.error")
logging.basicConfig(level=logging.INFO)

# =============================================================================
# BANCO DE DADOS: MODELOS E CONFIGURAÇÃO COM SQLAlchemy (Async)
# =============================================================================

Base = declarative_base()


class FlowType(enum.Enum):
    roteiro = "roteiro"
    conversa = "conversa"


class User(Base):
    __tablename__ = "users"
    phone = Column(String(20), primary_key=True, index=True)
    name = Column(String(100), nullable=True)
    email = Column(String(100), nullable=True)
    data_nascimento = Column(String(20), nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, onupdate=func.now())


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_phone = Column(String(20), ForeignKey(
        "users.phone"), nullable=False, index=True)
    flow = Column(Enum(FlowType), nullable=False)
    started_at = Column(DateTime, server_default=func.now())
    ended_at = Column(DateTime, nullable=True)
    session_data = Column(JSON, nullable=True)
    raw_payload = Column(JSON, nullable=True)


class ConversationMessage(Base):
    __tablename__ = "conversation_messages"
    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(Integer, ForeignKey(
        "conversation_sessions.id"), nullable=False, index=True)
    sender = Column(String(10), nullable=False)  # 'user' ou 'bot'
    message = Column(Text, nullable=False)
    step = Column(Integer, nullable=True)
    status = Column(String(20), nullable=True)
    timestamp = Column(DateTime, server_default=func.now())


class UserProfile(Base):
    __tablename__ = "user_profiles"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_phone = Column(String(20), ForeignKey("users.phone"), nullable=False)
    conversation_session_id = Column(Integer, ForeignKey(
        "conversation_sessions.id"), nullable=True)
    summary = Column(Text, nullable=False)
    created_at = Column(DateTime, server_default=func.now())


engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False)


async def get_db_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session


async def create_or_get_user(phone: str, name: Optional[str] = None) -> User:
    async with AsyncSessionLocal() as session:
        user = await session.get(User, phone)
        if not user:
            user = User(phone=phone, name=name)
            session.add(user)
            await session.commit()
            await session.refresh(user)
            logger.info(f"Usuário criado: {phone}")
        return user


async def update_user_info(phone: str, name: str, email: str, data_nascimento: str) -> None:
    async with AsyncSessionLocal() as session:
        user = await session.get(User, phone)
        if user:
            user.name = name
            user.email = email
            user.data_nascimento = data_nascimento
            await session.commit()
            logger.info(
                f"Usuário {phone} atualizado: {name}, {email}, {data_nascimento}")


async def create_conversation_session(phone: str, flow: FlowType, raw_payload: Dict[str, Any], session_data: dict = {}) -> ConversationSession:
    async with AsyncSessionLocal() as session:
        conv_session = ConversationSession(
            user_phone=phone, flow=flow, session_data=session_data, raw_payload=raw_payload)
        session.add(conv_session)
        await session.commit()
        await session.refresh(conv_session)
        logger.info(
            f"Nova sessão criada no DB: {conv_session.id} para o telefone {phone}")
        return conv_session


async def save_message(session_id: int, sender: str, message: str, step: Optional[int] = None, status: Optional[str] = None) -> ConversationMessage:
    async with AsyncSessionLocal() as session:
        conv_message = ConversationMessage(
            session_id=session_id, sender=sender, message=message, step=step, status=status)
        session.add(conv_message)
        await session.commit()
        await session.refresh(conv_message)
        logger.info(
            f"Mensagem salva no DB para sessão {session_id}: [{sender}] {message}")
        return conv_message


async def save_user_profile(user_phone: str, session_id: int, summary: str) -> None:
    q = text("""
        INSERT INTO user_profiles (user_phone, conversation_session_id, summary)
        VALUES (:user_phone, :session_id, :summary)
    """)
    params = {"user_phone": user_phone,
        "session_id": session_id, "summary": summary}
    async with AsyncSessionLocal() as session:
        try:
            await session.execute(q, params)
            await session.commit()
            logger.info(
                f"Perfil salvo com sucesso para {user_phone} na sessão {session_id}.")
        except Exception as e:
            logger.error(
                f"Erro ao salvar perfil para {user_phone} na sessão {session_id}: {e}")

# =============================================================================
# VALIDAÇÃO DOS CAMPOS DA FASE COMMON
# =============================================================================


def validate_name(name: str) -> bool:
    if len(name.split()) < 2:
        return False
    if not re.match(r"^[A-Za-zÀ-ÖØ-öø-ÿ ]+$", name):
        return False
    return True


def validate_email(email: str) -> bool:
    pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
    return re.match(pattern, email) is not None


def validate_date(date_str: str) -> bool:
    try:
        datetime.strptime(date_str, "%d/%m/%Y")
        return True
    except ValueError:
        return False


RESPONSES_TEMPLATES = {
    "nome": [
        "Eba, {name}! Que alegria conhecer seu nome completo!",
        "Ótimo, {name}! Que prazer receber seu cadastro!",
        "Show de bola, {name}! Obrigado por compartilhar seu nome completo!"
    ],
    "email": [
        "Maravilha! {email} é um e-mail super bacana.",
        "Legal, {email}! Esse e-mail vai facilitar nossa comunicação.",
        "Beleza! O e-mail {email} foi cadastrado com sucesso."
    ],
    "data_nascimento": [
        "Perfeito! Sabia que {date} é uma data cheia de histórias?",
        "Excelente! {date} é uma data incrível para celebrar.",
        "Show! A data {date} tem um charme todo especial."
    ]
}


def get_template_response(field: str, value: str) -> str:
    if field in RESPONSES_TEMPLATES:
        template = random.choice(RESPONSES_TEMPLATES[field])
        if field == "data_nascimento":
            try:
                dt = datetime.strptime(value, "%d/%m/%Y")
                value = dt.strftime("%d de %B de %Y")
            except Exception:
                pass
        return template.format(name=value, email=value, date=value)
    return f"Obrigado por informar seu {field}: {value}!"


async def validate_common_response(field: str, response: str) -> (bool, str):
    if field == "nome":
        if validate_name(response):
            return True, f"Eba, {response}! Que prazer conhecer seu nome completo!"
        else:
            return False, "O nome informado não parece estar completo. Por favor, digite seu nome completo (ex.: 'João da Silva')."
    elif field == "email":
        if validate_email(response):
            return True, f"Maravilha! {response} é um ótimo e-mail!"
        else:
            return False, "Hmm, esse e-mail não parece correto. Verifique e envie novamente (ex.: exemplo@dominio.com)."
    elif field == "data_nascimento":
        if validate_date(response):
            try:
                dt = datetime.strptime(response, "%d/%m/%Y")
                formatted_date = dt.strftime("%d de %B de %Y")
            except Exception:
                formatted_date = response
            return True, f"Perfeito! Sabia que {formatted_date} é uma data incrível!"
        else:
            return False, "A data de nascimento não está no formato correto. Use o formato dd/mm/aaaa, por favor."
    return True, "Campo válido."


async def humanize_common_message(field: str, response: str, conversation_history: str = "") -> str:
    prompt = (
        "Você é um assistente de viagens super descolado, empático e espontâneo. "
        "Você já está no meio de uma conversa para coleta de dados (Nome, e-mail e data de nascimento). Foque na coleta de dados. "
        "Use emojis e linguagem informal de forma sutil.\n\n"
        f"Histórico da conversa (contexto): {conversation_history}\n\n"
        f"Acabou de receber uma nova resposta para o campo '{field}': '{response}'.\n\n"
        "Gere uma resposta natural e fluida que se encaixe na conversa sem repetir saudações ou perguntar como o usuário está."
    )
    try:
        async with httpx.AsyncClient() as client:
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            data = {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "Você é um assistente de viagens super descolado, empático e espontâneo."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.9,
                "max_tokens": 60
            }
            response_ai = await client.post(url, headers=headers, json=data, timeout=10.0)
            response_ai.raise_for_status()
            ai_data = response_ai.json()
            message = ai_data["choices"][0]["message"]["content"].strip()
            return message
    except Exception as e:
        logger.error(f"Erro ao gerar mensagem humanizada via OpenAI: {e}")
        return f"Obrigado por informar seu {field}: {response}!"

# =============================================================================
# GERENCIAMENTO DE SESSÕES EM MEMÓRIA
# =============================================================================
# Cada sessão terá:
# - "phase": 'common', 'branch', 'flow', 'agenda_pending' ou 'agenda_choice'
# - "step": índice da etapa atual
# - "bot_selection": "roteiro" ou "conversa"
# - "data": dicionário com as respostas
# - "db_session_id": ID da sessão no banco
# - "conversation_history": para armazenar um resumo do diálogo
sessions: Dict[str, Dict[str, Any]] = {}

common_steps = [
    {"field": "nome", "question": "Para começarmos, qual é o seu nome completo?"},
    {"field": "email", "question": "Qual é o seu e-mail?"},
    {"field": "data_nascimento",
        "question": "Qual é a sua data de nascimento? (dd/mm/aaaa)"}
]

branching_question = (
    "Agora, escolha uma opção:\n"
    "1️⃣  *Roteiro de Viagem*\n"
    "2️⃣  *Conversa com um Especialista*\n"
    "Digite 1 ou 2."
)

roteiro_steps = [
    {"field": "preferencia_viagem",
        "question": "Como você prefere viajar? (Relaxando, tradicional, freneticamente ou se aventurando?)"},
    {"field": "cpf", "question": "Por favor, informe seu CPF."}
]

# =============================================================================
# MODELO DO PAYLOAD DO WHATSAPP
# =============================================================================


class TextPayload(BaseModel):
    body: str


class AudioPayload(BaseModel):
    mime_type: str
    sha256: str
    id: str
    voice: Optional[bool] = None


class MessagePayload(BaseModel):
    from_: str = Field(..., alias="from")
    id: str
    timestamp: str
    type: str
    text: Optional[TextPayload] = None
    audio: Optional[AudioPayload] = None


class ContactProfilePayload(BaseModel):
    name: str


class ContactPayload(BaseModel):
    wa_id: str
    profile: ContactProfilePayload


class MetadataPayload(BaseModel):
    display_phone_number: str
    phone_number_id: str


class ConversationPayload(BaseModel):
    id: str
    expiration_timestamp: Optional[str] = None
    origin: Dict[str, Any]


class PricingPayload(BaseModel):
    billable: Optional[bool] = None
    pricing_model: Optional[str] = None
    category: Optional[str] = None


class StatusPayload(BaseModel):
    id: str
    status: str
    timestamp: str
    recipient_id: str
    conversation: Optional[ConversationPayload] = None
    pricing: Optional[PricingPayload] = None


class ValuePayload(BaseModel):
    messaging_product: str
    metadata: MetadataPayload
    contacts: Optional[List[ContactPayload]] = None
    messages: Optional[List[MessagePayload]] = None
    statuses: Optional[List[StatusPayload]] = None


class ChangePayload(BaseModel):
    field: str
    value: ValuePayload


class EntryPayload(BaseModel):
    id: str
    changes: List[ChangePayload]


class WebhookPayload(BaseModel):
    object: str
    entry: List[EntryPayload]

# =============================================================================
# EXCEPTION HANDLER PARA ERROS DE VALIDAÇÃO
# =============================================================================


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    try:
        body = await request.body()
        body_str = body.decode("utf-8")
    except Exception as e:
        body_str = "Não foi possível ler o corpo da requisição"
        logger.error(f"Erro ao ler o corpo da requisição: {e}")
    logger.error(
        f"Erro de validação. Corpo: {body_str}\nDetalhes: {json.dumps(exc.errors(), indent=2)}")
    return JSONResponse(content={"detail": exc.errors(), "body": body_str}, status_code=422)

# =============================================================================
# FUNÇÃO PARA BAIXAR ARQUIVO DE MÍDIA DO WHATSAPP (usando file_id)
# =============================================================================


async def download_file_from_facebook(file_id: str, file_type: str, mime_type: str) -> str:
    url = f"https://graph.facebook.com/v22.0/{file_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    attempt = 0
    while attempt < 3:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)
                logger.info(f"[Tentativa {attempt+1}] GET {url} -> Status {response.status_code}")
                response.raise_for_status()
                media_info = response.json()
                logger.info(f"Media info obtida: {media_info}")
                download_url = media_info.get("url")
                if not download_url:
                    raise Exception("URL de mídia não encontrada no response.")
                # NÃO adicionamos o token à URL; usamos conforme retornado
                logger.info(f"Utilizando download_url sem modificação: {download_url}")
                logger.info(f"[Tentativa {attempt+1}] Tentando baixar mídia da URL: {download_url}")
                media_response = await client.get(download_url, headers=headers, timeout=30.0)
                logger.info(f"[Tentativa {attempt+1}] GET {download_url} -> Status {media_response.status_code}")
                media_response.raise_for_status()
                file_extension = "ogg" if "opus" in mime_type.lower() else mime_type.split("/")[-1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp_file:
                    tmp_file.write(media_response.content)
                    temp_file_path = tmp_file.name
                logger.info(f"Arquivo de mídia {file_id} salvo em: {temp_file_path} | Tamanho: {os.path.getsize(temp_file_path)} bytes")
                return temp_file_path
        except Exception as e:
            logger.error(f"Tentativa {attempt+1} - Erro ao baixar mídia {file_id}: {e}")
            attempt += 1
            await asyncio.sleep(2)
    raise Exception(f"Não foi possível baixar a mídia {file_id} após 3 tentativas.")

# =============================================================================
# FUNÇÃO PARA CONVERTER ÁUDIO PARA WAV (USANDO pydub)
# =============================================================================


def convert_audio_to_wav(input_path: str) -> BinaryIO:
    try:
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        return buffer
    except Exception as e:
        logger.error(f"Erro ao converter áudio: {e}")
        raise

# =============================================================================
# FUNÇÃO SINCRÔNICA PARA TRANSCRIÇÃO USANDO WHISPER-1
# =============================================================================


def sync_transcribe_audio_file(audio_file_path: str) -> str:
    """Transcreve o arquivo de áudio usando Whisper com a nova API."""
    try:
        # Converter para WAV (mesmo que o buffer convertido não seja usado)
        wav_buffer = convert_audio_to_wav(audio_file_path)
        if wav_buffer.getbuffer().nbytes < 1024:
            raise ValueError("Arquivo de áudio muito pequeno após conversão")
        with open(audio_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model="whisper-1",
                response_format="text"
            )
        # Se o retorno da API já for a string transcrita, retorne assim:
        return transcription  # ou, se necessário, `transcription.text`
    except Exception as e:
        logger.error(f"Erro na transcrição: {str(e)}")
        raise ValueError(f"Error transcribing audio: {str(e)}")

# =============================================================================
# FUNÇÃO ASSÍNCRONA PARA TRANSCRIÇÃO (EXECUTA A FUNÇÃO SINCRÔNICA)
# =============================================================================


async def transcribe_audio_file(file_path: str) -> str:
    """Executa a transcrição síncrona em um executor separado."""
    try:
        # Verificação adicional do arquivo
        if not os.path.exists(file_path) or os.path.getsize(file_path) < 1024:
            logger.error("Arquivo inválido ou muito pequeno")
            return ""
        
        loop = asyncio.get_running_loop()
        transcript = await loop.run_in_executor(None, sync_transcribe_audio_file, file_path)
        return transcript
    except Exception as e:
        logger.error(f"Erro na transcrição do arquivo {file_path}: {e}")
        return ""

# =============================================================================
# FUNÇÃO PARA PROCESSAR MENSAGENS DE ÁUDIO
# =============================================================================


async def process_audio_message(message: MessagePayload, remetente: str):
    """Processa mensagens de áudio recebidas."""
    logger.info("Processando mensagem de áudio com Whisper-1...")
    try:
        # Baixar o arquivo
        temp_file_path = await download_file_from_facebook(
            message.audio.id, 
            "audio", 
            message.audio.mime_type
        )
        
        # Verificar o arquivo antes de transcrever
        file_size = os.path.getsize(temp_file_path)
        logger.info(f"Tamanho do arquivo: {file_size} bytes")
        
        if file_size < 2048:  # 2KB
            await enviar_resposta_whatsapp(
                remetente, 
                "O áudio recebido é muito curto. Por favor, envie uma mensagem mais longa (pelo menos 2 segundos)."
            )
            return
        
        # Transcrever
        transcript = await transcribe_audio_file(temp_file_path)
        
        # Limpeza
        try:
            os.remove(temp_file_path)
            logger.info(f"Arquivo temporário {temp_file_path} removido.")
        except Exception as e:
            logger.error(f"Falha ao deletar arquivo {temp_file_path}: {e}")
        
        # Processar resposta
        if transcript:
            logger.info(f"Transcrição: {transcript}")
            await processa_e_responde(transcript, remetente)
        else:
            await enviar_resposta_whatsapp(
                remetente, 
                "Desculpe, não consegui transcrever seu áudio. Por favor, tente novamente com uma gravação mais clara."
            )
            
    except Exception as e:
        logger.error(f"Erro no processamento de áudio: {e}")
        await enviar_resposta_whatsapp(
            remetente, 
            "Desculpe, ocorreu um problema ao processar seu áudio. Por favor, tente novamente."
        )

# =============================================================================
# ENVIO DE MENSAGENS (SIMULA "DIGITANDO")
# =============================================================================


async def enviar_resposta_whatsapp(remetente: str, mensagem: str):
    logger.info(f"Simulando 'digitando' para {remetente}...")
    await asyncio.sleep(2)
    remetente_formatado = remetente.lstrip("+").strip()
    payload_data = {
        "messaging_product": "whatsapp",
        "to": remetente_formatado,
        "type": "text",
        "text": {"body": mensagem, "preview_url": False},
        "recipient_type": "individual"
    }
    headers = {"Content-Type": "application/json",
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}"}
    logger.info(f"Enviando mensagem: {json.dumps(payload_data, indent=2)}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(WHATSAPP_API_URL, headers=headers, json=payload_data, timeout=10.0)
            logger.info(
                f"Resposta WhatsApp: {response.status_code} - {response.text}")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error(f"Erro ao enviar mensagem: {exc}")

# =============================================================================
# FUNÇÕES PARA GERAR RESPOSTAS COM OPENAI (FLUXO DE ROTEIRO)
# =============================================================================


async def generate_followup(previous_question: str, user_answer: str, next_question: str, session_data: Dict[str, str], conversation_history: str = "") -> str:
    prompt = (
        "Você é um assistente de viagens super descolado, alegre e espontâneo, que conversa com a naturalidade de um amigo. "
        "Você já está no meio de um fluxo de roteiro com o usuário.\n"
        f"Histórico da conversa: {conversation_history}\n\n"
        f"Pergunta anterior: \"{previous_question}\"\n"
        f"Resposta do usuário: \"{user_answer}\"\n"
        f"Próxima pergunta: \"{next_question}\"\n\n"
        "Gere uma resposta única que reconheça o que o usuário disse e introduza a próxima pergunta de forma fluida e natural, sem repetir saudações padrão."
    )
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Você é um assistente de viagens super descolado e espontâneo, já no meio de um fluxo de roteiro."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.85,
        "max_tokens": 200
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=10.0)
            response.raise_for_status()
            json_response = response.json()
            reply = json_response["choices"][0]["message"]["content"].strip()
            logger.info(f"Resposta OpenAI (roteiro): {reply}")
            return reply
        except Exception as e:
            logger.error(f"Erro no OpenAI (roteiro): {e}")
            return next_question

# =============================================================================
# FUNÇÃO DO BOT DE CONVERSA (Especialista em Turismo)
# =============================================================================


async def run_conversa_agent(query: str, conversation_history: str = "") -> str:
    prompt = (
        "Você é um especialista em viagens com extenso conhecimento sobre turismo, destinos e experiências. "
        "Você já está conversando com o usuário de forma natural, sem repetir saudações desnecessárias.\n"
        f"Histórico da conversa: {conversation_history}\n\n"
        "Pergunta do usuário: " + query
    )
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json",
        "Authorization": f"Bearer {OPENAI_API_KEY}"}
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "Você é um especialista em viagens focado exclusivamente em temas de turismo e está conversando de forma natural."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.85,
        "max_tokens": 250
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, headers=headers, json=data, timeout=10.0)
            response.raise_for_status()
            json_response = response.json()
            answer = json_response["choices"][0]["message"]["content"].strip()
            logger.info(f"Resposta OpenAI (conversa): {answer}")
            return answer
        except Exception as e:
            logger.error(f"Erro no run_conversa_agent: {e}")
            return "Desculpe, ocorreu um problema ao processar sua mensagem."


async def processa_e_responde_conversa(texto_usuario: str, remetente: str):
    if "conversation_history" not in sessions[remetente]:
        sessions[remetente]["conversation_history"] = ""
    sessions[remetente]["conversation_history"] += f"Usuário: {texto_usuario}\n"
    try:
        resposta_bot = await run_conversa_agent(texto_usuario, sessions[remetente]["conversation_history"])
        await enviar_resposta_whatsapp(remetente, resposta_bot)
        sessions[remetente]["conversation_history"] += f"Bot: {resposta_bot}\n"
        if sessions.get(remetente) and sessions[remetente].get("db_session_id"):
            await save_message(sessions[remetente]["db_session_id"], "user", texto_usuario)
            await save_message(sessions[remetente]["db_session_id"], "bot", resposta_bot)
    except Exception as e:
        logger.error(f"Erro no fluxo de conversa: {e}")
        await enviar_resposta_whatsapp(remetente, "Desculpe, ocorreu um problema. Tente novamente.")

# =============================================================================
# FUNÇÃO DO BOT DE ROTEIRO (FLUXO ESPECÍFICO) COM AGENDAMENTO
# =============================================================================


async def processa_e_responde_roteiro_flow(texto_usuario: str, remetente: str):
    session = sessions[remetente]
    current_step = session["step"]
    # Salva mensagem do usuário no banco, se aplicável
    if sessions.get(remetente) and sessions[remetente].get("db_session_id"):
        await save_message(sessions[remetente]["db_session_id"], "user", texto_usuario, current_step)
    session["data"][roteiro_steps[current_step]["field"]] = texto_usuario
    logger.info(
        f"Registrada resposta para '{roteiro_steps[current_step]['field']}': {texto_usuario}")

    if "conversation_history" not in session:
        session["conversation_history"] = ""
    session["conversation_history"] += f"Usuário: {texto_usuario}\n"

    session["step"] = current_step + 1
    new_step = session["step"]
    if new_step < len(roteiro_steps):
        previous_question = roteiro_steps[current_step]["question"].format(
            **session["data"])
        next_question = roteiro_steps[new_step]["question"].format(
            **session["data"])
        resposta_bot = await generate_followup(previous_question, texto_usuario, next_question, session["data"], session["conversation_history"])
        await enviar_resposta_whatsapp(remetente, resposta_bot)
        if sessions.get(remetente) and sessions[remetente].get("db_session_id"):
            await save_message(sessions[remetente]["db_session_id"], "bot", resposta_bot, new_step)
        session["conversation_history"] += f"Bot: {resposta_bot}\n"
        session["conversation_history"] += f"Bot: {next_question}\n"
    else:
        # Fluxo de roteiro concluído: envia resumo e inicia etapa de agendamento
        resumo = "Beleza, terminamos nossa coleta de informações! Aqui vai um resumo do que você compartilhou:\n\n"
        for step in common_steps + roteiro_steps:
            field = step["field"]
            answer = session["data"].get(field, "Não informado")
            resumo += f"{field.capitalize()}: {answer}\n"
        resumo += "\nValeu! Em breve entraremos em contato para planejar a viagem dos seus sonhos."
        await enviar_resposta_whatsapp(remetente, resumo)
        if sessions.get(remetente) and sessions[remetente].get("db_session_id"):
            await save_message(sessions[remetente]["db_session_id"], "bot", resumo)
            await save_user_profile(remetente, sessions[remetente]["db_session_id"], resumo)
        # Inicia fluxo de agendamento em vez de finalizar a sessão:
        agenda_msg = (
            "Gostaria de agendar uma reunião com o nosso time DSM para discutir sua viagem? \n"
            "Responda 1 para *Sim* ou 2 para *Não*."
        )
        await enviar_resposta_whatsapp(remetente, agenda_msg)
        # Atualiza a fase para agendamento
        session["phase"] = "agenda_pending"
        session["step"] = 0

# =============================================================================
# INTEGRAÇÃO COM GOOGLE CALENDAR VIA CONTA DE SERVIÇO
# =============================================================================


def get_calendar_service():
    service_account_info_str = os.getenv("GOOGLE_SERVICE_ACCOUNT_CONTENT")
    if not service_account_info_str:
        raise Exception(
            "A variável de ambiente 'GOOGLE_SERVICE_ACCOUNT_CONTENT' não foi definida.")
    try:
        service_account_info = json.loads(service_account_info_str)
    except Exception as ex:
        logging.error(
            "Erro ao fazer parse do JSON da Service Account", exc_info=True)
        raise ex
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=["https://www.googleapis.com/auth/calendar"]
    )
    service = build("calendar", "v3", credentials=credentials)
    return service


def listar_calendarios():
    try:
        service = get_calendar_service()
        calendar_list = service.calendarList().list().execute()
        logger.info("Lista de calendários disponíveis: %s",
                    json.dumps(calendar_list, indent=2))
        return calendar_list
    except Exception as e:
        logger.error("Erro ao listar calendários", exc_info=True)
        return None


def marcar_evento_no_google_calendar(user_data: Dict[str, str], slot: Dict[str, Any]) -> bool:
    try:
        service = get_calendar_service()
        # Utilize o Calendar ID definido no .env (agora deverá estar "alissonalbuquerquecavalcante@gmail.com")
        calendar_id = os.getenv("GOOGLE_CALENDAR_ID",
                                "alissonalbuquerquecavalcante@gmail.com")
        logger.info("Usando o calendarId: %s", calendar_id)
        calendar_list = listar_calendarios()
        if calendar_list is None or not calendar_list.get("items"):
            logger.warning(
                "Nenhum calendário foi encontrado para a conta de serviço. Verifique o compartilhamento ou delegação.")

        event = {
            "summary": f"Reunião DSM com {user_data.get('nome', 'Usuário')}",
            "description": f"Agendamento realizado pelo bot. Email: {user_data.get('email', 'Não informado')}",
            "start": {
                "dateTime": f"{slot['data']}T{slot['hora']}:00",
                "timeZone": "America/Sao_Paulo",
            },
            "end": {
                "dateTime": f"{slot['data']}T{str(int(slot['hora'].split(':')[0]) + 1).zfill(2)}:{slot['hora'].split(':')[1]}:00",
                "timeZone": "America/Sao_Paulo",
            }
        }
        logger.info("Dados do evento a ser criado: %s",
                    json.dumps(event, indent=2))
        created_event = service.events().insert(
            calendarId=calendar_id, body=event).execute()
        logger.info("Evento criado com sucesso. Resposta completa:\n%s",
                    json.dumps(created_event, indent=2))
        return True
    except Exception as e:
        logger.error("Erro ao criar evento no Google Calendar", exc_info=True)
        return False


async def obter_disponibilidade_agenda():
    """
    Consulta a agenda (do calendário especificado por GOOGLE_CALENDAR_ID) para os próximos 7 dias,
    excluindo sábados e domingos, e gera uma lista de slots disponíveis (1 hora, de 09:00 a 17:00)
    que não conflitam com os eventos já agendados.
    """
    try:
        service = get_calendar_service()
        calendar_id = os.getenv("GOOGLE_CALENDAR_ID",
                                "alissonalbuquerquecavalcante@gmail.com")

        # Define o fuso horário local
        tz = ZoneInfo("America/Sao_Paulo")
        now = datetime.now(tz)

        # Define o período da consulta: de agora até 7 dias a partir de agora.
        timeMin = now.isoformat()
        timeMax = (now + timedelta(days=7)).isoformat()

        request_body = {
            "timeMin": timeMin,
            "timeMax": timeMax,
            "timeZone": "America/Sao_Paulo",
            "items": [{"id": calendar_id}]
        }
        freebusy_result = service.freebusy().query(body=request_body).execute()
        busy_times = freebusy_result["calendars"][calendar_id]["busy"]
        logger.info("Busy times: %s", json.dumps(busy_times, indent=2))

        available_slots = []
        meeting_duration = timedelta(hours=1)
        # Loop por cada dia da próxima semana
        for i in range(7):
            day = now + timedelta(days=i)
            # Ignora sábado (weekday() == 5) e domingo (weekday() == 6)
            if day.weekday() >= 5:
                continue
            # Define a janela de trabalho para esse dia: 09:00 a 17:00 com fuso tz
            work_start = day.replace(hour=9, minute=0, second=0, microsecond=0)
            work_end = day.replace(hour=17, minute=0, second=0, microsecond=0)

            busy_intervals = []
            for interval in busy_times:
                # Converte os horários de busy em datetime aware usando o fuso UTC e converte para tz
                busy_start = datetime.fromisoformat(
                    interval["start"].replace("Z", "+00:00")).astimezone(tz)
                busy_end = datetime.fromisoformat(
                    interval["end"].replace("Z", "+00:00")).astimezone(tz)
                if busy_start.date() == day.date():
                    busy_intervals.append((busy_start, busy_end))

            logger.info("Dia %s - Busy intervals: %s", work_start.strftime("%Y-%m-%d"),
                        json.dumps([(b[0].isoformat(), b[1].isoformat()) for b in busy_intervals], indent=2))

            current_time = work_start
            while current_time + meeting_duration <= work_end:
                slot_start = current_time
                slot_end = current_time + meeting_duration
                conflict = False
                for b_start, b_end in busy_intervals:
                    if slot_start < b_end and slot_end > b_start:
                        conflict = True
                        break
                if not conflict:
                    available_slots.append({
                        "id": len(available_slots) + 1,
                        "data": slot_start.strftime("%Y-%m-%d"),
                        "hora": slot_start.strftime("%H:%M")
                    })
                current_time += meeting_duration
        logger.info("Available slots: %s", json.dumps(
            available_slots, indent=2))
        return available_slots
    except Exception as e:
        logger.error("Erro ao obter disponibilidade da agenda", exc_info=True)
        return []

# Exemplo de uso dentro do seu fluxo de agendamento no código maior:


async def processa_agendamento(remetente: str, texto_usuario: str):
    session = sessions.get(remetente)
    if not session:
        return
    if session.get("phase") == "agenda_pending":
        escolha = texto_usuario.strip()
        if escolha == "1":
            disponiveis = await obter_disponibilidade_agenda()
            if disponiveis:
                session["disponiveis"] = disponiveis
                msg = "Segue as opções disponíveis:\n"
                for slot in disponiveis:
                    msg += f"{slot['id']}: {slot['data']} às {slot['hora']}\n"
                msg += "Por favor, responda com o número da opção desejada."
                await enviar_resposta_whatsapp(remetente, msg)
                session["phase"] = "agenda_choice"
            else:
                await enviar_resposta_whatsapp(remetente, "Desculpe, não foi possível obter a disponibilidade no momento. Tente novamente mais tarde.")
                del sessions[remetente]
        elif escolha == "2":
            await enviar_resposta_whatsapp(remetente, "Ok, sem problemas! Se precisar de algo mais, estou por aqui. Boa viagem!")
            del sessions[remetente]
        else:
            await enviar_resposta_whatsapp(remetente, "Opção inválida. Responda 1 para *Sim* ou 2 para *Não*.")
    elif session.get("phase") == "agenda_choice":
            try:
                opcao = int(texto_usuario.strip())
            except ValueError:
                await enviar_resposta_whatsapp(remetente, "❌ Por favor, informe um número correspondente a uma das opções.")
                return
            disponiveis = session.get("disponiveis", [])
            slot_selecionado = next((s for s in disponiveis if s["id"] == opcao), None)
            if slot_selecionado:
                sucesso = marcar_evento_no_google_calendar(session.get("data", {}), slot_selecionado)
                if sucesso:
                    confirm_msg = (
                        "✅ *Agendamento Confirmado!*\n"
                        "📅 *Data:* {}\n"
                        "⏰ *Horário:* {}\n\n"
                        "Seu agendamento foi marcado com sucesso! Em breve entraremos em contato. 😊"
                    ).format(slot_selecionado["data"], slot_selecionado["hora"])
                    await enviar_resposta_whatsapp(remetente, confirm_msg)
                else:
                    await enviar_resposta_whatsapp(remetente, "😔 Houve um erro ao marcar o agendamento. Por favor, tente novamente mais tarde.")
                del sessions[remetente]
            else:
                await enviar_resposta_whatsapp(remetente, "❌ Opção inválida. Por favor, informe um número correspondente à opção desejada.")

# =============================================================================
# FUNÇÃO PRINCIPAL DE GESTÃO DO FLUXO (BRANCHING)
# =============================================================================

async def processa_e_responde(texto_usuario: str, remetente: str):
    if remetente not in sessions:
        sessions[remetente] = {"phase": "common", "step": 0, "data": {}, "bot_selection": None, "db_session_id": None}
        welcome_msg = "Olá! Eu sou a *Gabi*, sua assistente de viagens. 😊\n\n" + common_steps[0]["question"]
        logger.info(f"Iniciando fase 'common' para {remetente}. Enviando: {welcome_msg}")
        await enviar_resposta_whatsapp(remetente, welcome_msg)
        return
    session = sessions[remetente]
    phase = session.get("phase")
    logger.info(f"Processando mensagem para {remetente} na fase: {phase}")
    if phase == "common":
        if session["step"] < len(common_steps):
            field = common_steps[session["step"]]["field"]
            is_valid, validation_msg = await validate_common_response(field, texto_usuario)
            if not is_valid:
                await enviar_resposta_whatsapp(remetente, validation_msg)
                return
            session["data"][field] = texto_usuario
            if "conversation_history" not in session:
                session["conversation_history"] = ""
            session["conversation_history"] += f"Usuário: {texto_usuario}\n"
            human_msg = await humanize_common_message(field, texto_usuario, session["conversation_history"])
            await enviar_resposta_whatsapp(remetente, human_msg)
            session["conversation_history"] += f"Bot: {human_msg}\n"
            logger.info(f"Coletado '{field}': {texto_usuario}")
            session["step"] += 1
            if session["step"] < len(common_steps):
                next_question = common_steps[session["step"]]["question"]
                await enviar_resposta_whatsapp(remetente, next_question)
                session["conversation_history"] += f"Bot: {next_question}\n"
            else:
                await update_user_info(
                    remetente,
                    session["data"].get("nome", ""),
                    session["data"].get("email", ""),
                    session["data"].get("data_nascimento", "")
                )
                await enviar_resposta_whatsapp(remetente, branching_question)
                session["conversation_history"] += f"Bot: {branching_question}\n"
                session["phase"] = "branch"
                session["step"] = 0
        return
    if phase == "branch":
        choice = texto_usuario.strip()
        logger.info(f"Resposta de branching para {remetente}: {choice}")
        if choice == "1":
            session["bot_selection"] = "roteiro"
        elif choice == "2":
            session["bot_selection"] = "conversa"
        else:
            await enviar_resposta_whatsapp(remetente, "Opção inválida! Digite 1 para Roteiro de Viagem ou 2 para Conversa com um Especialista.")
            return
        await create_or_get_user(remetente)
        flow = FlowType.roteiro if session["bot_selection"] == "roteiro" else FlowType.conversa
        conv_session = await create_conversation_session(remetente, flow, raw_payload={})
        session["db_session_id"] = conv_session.id
        session["phase"] = "flow"
        session["step"] = 0
        if session["bot_selection"] == "roteiro":
            first_question = roteiro_steps[0]["question"]
        else:
            first_question = "Beleza! Agora, sobre o que você quer conversar? Pode ser sobre viagens, dicas ou experiências."
        logger.info(f"Iniciando fase 'flow' para {remetente} com fluxo '{session['bot_selection']}'. Enviando: {first_question}")
        await enviar_resposta_whatsapp(remetente, first_question)
        if session.get("db_session_id"):
            await save_message(session["db_session_id"], "bot", first_question, 0)
        return
    if phase == "flow":
        if session.get("bot_selection") == "roteiro":
            await processa_e_responde_roteiro_flow(texto_usuario, remetente)
        else:
            await processa_e_responde_conversa(texto_usuario, remetente)
    elif phase in ["agenda_pending", "agenda_choice"]:
        await processa_agendamento(remetente, texto_usuario)

# =============================================================================
# ENDPOINTS DO WEBHOOK DO WHATSAPP
# =============================================================================

@app.get("/", response_class=PlainTextResponse)
async def verify_webhook(
    hub_mode: str = Query(..., alias="hub.mode"),
    hub_challenge: str = Query(..., alias="hub.challenge"),
    hub_verify_token: str = Query(..., alias="hub.verify_token")
):
    logger.info(f"Verificação do webhook: mode={hub_mode}, challenge={hub_challenge}, token={hub_verify_token}")
    if hub_verify_token != VERIFY_TOKEN:
        logger.warning("Token de verificação inválido.")
        raise HTTPException(status_code=403, detail="Token de verificação inválido")
    return hub_challenge

@app.post("/")
async def receive_message(payload: WebhookPayload, background_tasks: BackgroundTasks):
    logger.info("Payload recebido do webhook:")
    logger.info(json.dumps(payload.dict(), indent=2))
    try:
        if not payload.entry:
            logger.error("Nenhuma entry encontrada no payload.")
            raise HTTPException(status_code=400, detail="Payload inválido: Nenhuma entry encontrada.")
        entry = payload.entry[0]
        if not entry.changes:
            logger.error("Nenhuma change encontrada na entry.")
            raise HTTPException(status_code=400, detail="Payload inválido: Nenhuma change encontrada.")
        change = entry.changes[0]
        value = change.value
        if value.messages and len(value.messages) > 0:
            message = value.messages[0]
            remetente = message.from_
            if message.type == "audio":
                await process_audio_message(message, remetente)
            else:
                texto_usuario = message.text.body.strip()
                logger.info(f"Mensagem recebida de {remetente}: {texto_usuario}")
                background_tasks.add_task(processa_e_responde, texto_usuario, remetente)
        elif value.statuses and len(value.statuses) > 0:
            logger.info("Evento de status recebido:")
            logger.info(json.dumps([status.dict() for status in value.statuses], indent=2))
        else:
            logger.error("Payload inválido: Nenhuma mensagem ou status encontrada.")
            raise HTTPException(status_code=400, detail="Payload inválido: Nenhuma mensagem ou status encontrada.")
    except Exception as e:
        logger.error(f"Erro ao processar payload: {e}")
        raise HTTPException(status_code=400, detail="Erro na estrutura do payload.")
    return JSONResponse(content={"status": "recebido"}, status_code=200)

# =============================================================================
# EXECUÇÃO DO SERVIDOR
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, log_level="info")

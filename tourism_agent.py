import os
from langchain_openai import OpenAI  # Importa da nova biblioteca langchain-openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv

# Carrega a chave da API do OpenAI do ambiente
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializa o LLM com temperatura que favoreça respostas informativas e criativas
llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.7)

# Define o template do prompt para que o agente atue como consultor de turismo
tourism_prompt = """
You are a highly knowledgeable, friendly, and experienced travel advisor specializing in tourism.
When a user asks a question or requests travel advice, provide detailed recommendations, insightful cultural information, and itinerary suggestions tailored to their needs.
Your responses should be accurate, engaging, and demonstrate expertise in tourism.

Examples:
User: "Can you suggest some interesting places to visit in Paris?"
Advisor: "Certainly! In Paris, you should consider visiting iconic landmarks like the Eiffel Tower and Louvre Museum, strolling through Montmartre, and exploring hidden gems in the Marais district. Also, try local French cuisine at traditional bistros."
User: {user_query}
Advisor:
"""

# Cria o template do prompt utilizando o LangChain PromptTemplate
prompt_template = PromptTemplate(
    input_variables=["user_query"],
    template=tourism_prompt
)

# Inicializa a cadeia de linguagem com o prompt customizado
llm_chain = LLMChain(llm=llm, prompt=prompt_template)

def run_tourism_agent(user_input: str) -> str:
    """
    Recebe a consulta do usuário, processa com o agente especializado em turismo e retorna a resposta.
    """
    response = llm_chain.run({"user_query": user_input})
    return response

import os
import asyncio
import requests
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions import kernel_function


## Define the environment

load_dotenv()

SUBSCRIPTION_KEY = os.getenv("TRANS_SUB_KEY")
REGION = os.getenv("TRANS_REGION")
ENDPOINT = os.getenv("TRANS_ENDPOINT")
MODEL_DEPLOYMENT = os.getenv("TRANS_MODEL_DEPLOYMENT")
## Activate the main kernel brain

kernel = Kernel()

### Load the Azure OpenAI service into the kernel (for agents' use and translation plugin)
kernel.add_service(
    AzureChatCompletion(
        service_id="svc",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
)
## define the legal research plugin (with dummy data for now) and add it to the kernel
class LegalResearchPlugin:

    @kernel_function(name="research")
    def research(self, query: str) -> str:
        ## here will implement the knowledge base search later on 
        return "المادة 12: يلتزم المقترض بسداد القرض."

kernel.add_plugin(LegalResearchPlugin(), "legal")


## Translation plugin using external API call (Azure Translator in this case)

class ExternalTranslationPlugin:

    def translate_text(self, text, from_lang, to_lang, model_deployment):
        headers = {
            "Ocp-Apim-Subscription-Key": SUBSCRIPTION_KEY,
            "Ocp-Apim-Subscription-Region": REGION,
            "Content-Type": "application/json",
        }

        params = {"api-version": "2025-05-01-preview"}

        body = [{
            "Text": text,
            "language": from_lang,
            "targets": [{
                "language": to_lang,
                "deploymentName": model_deployment
            }]
        }]

        r = requests.post(
            ENDPOINT,
            params=params,
            headers=headers,
            json=body,
            timeout=30
        )

        r.raise_for_status()
        data = r.json()

        return data[0]["translations"][0]["text"]

    @kernel_function(name="translate")
    def translate(self, text: str, from_lang: str, to_lang: str) -> str:
        return self.translate_text(
            text,
            from_lang,
            to_lang,
            MODEL_DEPLOYMENT
        )


kernel.add_plugin(ExternalTranslationPlugin(), "translation")

## Define agents with instructions and shared settings

args = KernelArguments(
    settings=PromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto()
    )
)

arabic_agent = ChatCompletionAgent(
    kernel=kernel,
    name="ArabicLegalAgent",
    instructions="""
You are an Arabic legal agent.
Always call legal.research.
Answer only from legal corpus.
Cite articles.
""",
    arguments=args
)

english_agent = ChatCompletionAgent(
    kernel=kernel,
    name="EnglishLegalAgent",
    instructions="""
You are an English legal agent.
Always call legal.research.
Answer only from legal corpus.
Cite articles.
""",
    arguments=args
)

## Define the knowledge base language (for routing and translation purposes)
KB_LANGUAGE = "arabic"

## detect language function (simple heuristic based on presence of Arabic characters)

def detect_language(text):
    return "ar" if any('\u0600' <= c <= '\u06FF' for c in text) else "en"

## creating router function that detects query language, translates if needed, routes to the correct agent, and translates back the answer if needed
async def route(query, thread):

    q_lang = detect_language(query)
    kb_lang = "ar" if KB_LANGUAGE == "arabic" else "en"

    working_query = query

    # Translate query → KB language
    if q_lang != kb_lang:
        working_query = kernel.plugins["translation"].translate(
            query,
            q_lang,
            kb_lang
        )

    # Call agent
    if kb_lang == "ar":
        answer = await arabic_agent.get_response(working_query, thread=thread)
    else:
        answer = await english_agent.get_response(working_query, thread=thread)

    answer_text = str(answer)

    # Translate answer → user language
    if q_lang != kb_lang:
        answer_text = kernel.plugins["translation"].translate(
            answer_text,
            kb_lang,
            q_lang
        )

    return answer_text

## Chat looping 
async def main():
    thread = ChatHistoryAgentThread()

    while True:
        q = input("Query: ")
        if q == "exit":
            break

        r = await route(q, thread)
        print("\n", r)

asyncio.run(main())

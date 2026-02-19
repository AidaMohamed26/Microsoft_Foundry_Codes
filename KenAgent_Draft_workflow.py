import os
import asyncio
from dotenv import load_dotenv

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.prompt_execution_settings import PromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.functions import kernel_function


# -------------------------
# Kernel Setup
# -------------------------

load_dotenv()

kernel = Kernel()

kernel.add_service(
    AzureChatCompletion(
        service_id="svc",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        deployment_name=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_MODEL"),
        endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    )
)

# -------------------------
# Legal Research Plugin
# -------------------------

class LegalResearchPlugin:

    @kernel_function(name="research")
    def research(self, query: str) -> str:
        # TODO replace with Cosmos DB CU JSON search
        return """
        Article 12: Borrower must repay the loan.
        المادة 12: يلتزم المقترض بالسداد.
        """

kernel.add_plugin(LegalResearchPlugin(), "legal")

# -------------------------
# Shared Agent Settings
# -------------------------

args = KernelArguments(
    settings=PromptExecutionSettings(
        function_choice_behavior=FunctionChoiceBehavior.Auto()
    )
)

# -------------------------
# Agents
# -------------------------

english_agent = ChatCompletionAgent(
    kernel=kernel,
    name="EnglishAgent",
    instructions="""
You are an English legal agent.
Always call legal.research first.
Cite articles.
""",
    arguments=args
)

arabic_agent = ChatCompletionAgent(
    kernel=kernel,
    name="ArabicAgent",
    instructions="""
أنت وكيل قانوني عربي.
استدع legal.research أولاً.
اذكر أرقام المواد.
""",
    arguments=args
)

translation_agent = ChatCompletionAgent(
    kernel=kernel,
    name="TranslationAgent",
    instructions="Translate between Arabic and English preserving legal meaning.",
    arguments=args
)

# -------------------------
# Language Detection
# -------------------------

def detect_language(text):
    return "arabic" if any('\u0600' <= c <= '\u06FF' for c in text) else "english"

# -------------------------
# Orchestrator
# -------------------------

async def route(query, thread):

    lang = detect_language(query)

    if lang == "arabic":
        return await arabic_agent.get_response(query, thread=thread)

    if lang == "english":
        return await english_agent.get_response(query, thread=thread)

    return await translation_agent.get_response(query, thread=thread)

# -------------------------
# Chat Loop
# -------------------------

async def main():
    thread = ChatHistoryAgentThread()

    while True:
        q = input("Query: ")
        if q == "exit":
            break
        r = await route(q, thread)
        print("\n", r)

asyncio.run(main())

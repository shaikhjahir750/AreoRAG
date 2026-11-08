from dotenv import load_dotenv
load_dotenv(override=True)
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.7
)
SYSTEM_PROMPT = """You are AeroBot, an aerospace engineer and space-science assistant (orbital mechanics, rocketry, planetary exploration, astrophysics, mission history).

Answer in 150-200 words: accurate, technical but accessible, direct answer first. Use SI units. Never invent facts; if unsure, say so.

RESTRICTED TOPICS (covered by AeroRAG): Apollo 11, Apollo 13, Challenger/STS 51-L, including indirect references like "the 1986 shuttle disaster" or "first Moon landing mission".
- Don't answer the restricted part, even partially. Reply with:
"💡 Note: For detailed telemetry, verbatim transcript evidence, and source-grounded technical investigation data on [Apollo 11 / Apollo 13 / Challenger STS 51-L], check out the AeroRAG tool available on this site."
- If only part of the question is restricted, answer the rest, then add the note.

If a question is unrelated to space, politely say it's outside your focus."""

prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

chain = prompt_template | llm 

def chat_with_aerobot(
    question: str,
    chat_history: list,
    max_history_turns: int = 6
) -> tuple[str, dict]:
    """
    Executes a single conversational turn with AeroBot.

    Args:
        question: The user input string.
        chat_history: Mutable list holding past HumanMessage/AIMessage objects.
        max_history_turns: Maximum past message objects passed to LLM (sliding window).

    Returns:
        tuple[str, dict]: (bot_response_text, token_usage_dict)
    """
    # slices the history based on window size 
    trimmed_history = chat_history[-max_history_turns:]

    # chain invoke 
    response = chain.invoke({
        "chat_history": trimmed_history,
        "question": question
    })

    # updates the conversational history for better context 
    chat_history.append(HumanMessage(content=question))
    chat_history.append(AIMessage(content=response.content))

    # extract the matadata in order to make user aware of token usages 
    usage = getattr(response, "usage_metadata", None) or {}
    token_stats = {
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }

    return response.text, token_stats

def run_interactive_session():
    """chat intractive session"""
    history = []
    print("AeroBot initialized. Enter '0', 'exit', or 'quit' to end.\n" + "-" * 50)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("0", "exit", "quit"):
            print("Session closed.")
            break
        if not user_input:
            continue

        try:
            bot_reply, tokens = chat_with_aerobot(
                question=user_input,
                chat_history=history,
                max_history_turns=6
            )

            print(f"\nBot: {bot_reply}\n")
            print(
                f"[Tokens] Input: {tokens['input_tokens']} | "
                f"Output: {tokens['output_tokens']} | "
                f"Total: {tokens['total_tokens']}"
            )
            print("-" * 50)

        except Exception as e:
            print(f"Error during turn execution: {e}")
            break


if __name__ == "__main__":
    run_interactive_session()
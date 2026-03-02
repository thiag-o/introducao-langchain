from langchain_classic.agents import tool, create_react_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langsmith import Client


load_dotenv()


@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and return the result."""
    try:
        result = eval(expression)
    except Exception as e:
        return f"Error: {e}"

    return str(result)


@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Mocked web search tool. Retorned a hardcoded result."""
    data = {
        "Brasil": "Brasília",
        "Estados Unidos": "Washington, D.C.",
        "Canadá": "Ottawa",
        "França": "Paris",
        "Alemanha": "Berlim",
        "Japão": "Tóquio",
        "Argentina": "Buenos Aires",
        "Portugal": "Lisboa",
        "Espanha": "Madri",
        "Austrália": "Camberra",
    }

    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"The capital of {country} is {capital}."
    return "Sorry, I don't have information about that country."


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, disable_streaming=True)

tools = [calculator, web_search_mock]

client = Client()

prompt = client.pull_prompt("hwchase17/react")

agent_chain = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain,
    tools=tools,
    verbose=True,
    handle_parsing_errors="Invalid format. Either provide an Action with Action Input, or a Final Answer only.",
    max_iterations=3,
)

# Example questions
# print(agent_executor.invoke({"input": "What is 5 + 7?"}))
print(agent_executor.invoke({"input": "What is the capital of Japan?"}))

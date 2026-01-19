from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
from dotenv import load_dotenv
load_dotenv()

@tool("calculator", return_direct=True)
def calculator(expression: str) -> str:
    """Evaluate a simple mathematical expression and returns the result"""
    try:
        result = eval(expression) # be careful with eval because it's a security risk
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"
    

@tool("web_search_mock")
def web_search_mock(query: str) -> str:
    """Mocked web search tool"""

    data = {"Brazil": "Brasilia", "France": "Paris", "Japan": "Tokyo"}

    for country, capital in data.items():
        if country.lower() in query.lower():
            return f"Found: {country} - {capital}"
    
    return "I don't know the country's capital"

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

tools = [calculator, web_search_mock]

prompt = hub.pull("hwchase17/react")

agent_chain = create_react_agent(model, tools, prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent_chain,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=3
)

print(agent_executor.invoke({"input": "What is the capital of France?"}))
print(agent_executor.invoke({"input": "What is 1234 multiplied by 5678?"}))

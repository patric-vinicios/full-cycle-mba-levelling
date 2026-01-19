from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
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

prompt = PromptTemplate.from_template(
    """
        Answer the following questions as best you can. You have access to the following tools.
        Only use the information you get from the tools, even if you know the answer.
        If the information is not provided by the tools, say you don't know.

        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action

        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        Rules:
        - If you choose an Action, do NOT include Final Answer in the same step.
        - After Action and Action Input, stop and wait for Observation.
        - Never search the internet. Only use the tools provided.

        Begin!

        Question: {input}
        Thought:{agent_scratchpad}
    """
)

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

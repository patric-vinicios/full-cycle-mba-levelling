from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain_core.runnables import chain
from dotenv import load_dotenv
load_dotenv()

@chain
def square(input_dict:dict) -> int:
    x = input_dict["x"]
    return {"square_result": x * x}

question_template = PromptTemplate(
    input_variables=["square_result"],
    template="Tell me fun facts about the number {square_result}"
)

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

result = square | question_template | model

result = result.invoke({"x": 7})
print(result.content)

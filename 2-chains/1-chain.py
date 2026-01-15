from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

template = PromptTemplate(
    input_variables=["name"],
    template="Hy, my name is {name}. Tell me  joke with my name!"
)

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

answer = template | model

result = answer.invoke({"name": "Patric"})

print(result.content)
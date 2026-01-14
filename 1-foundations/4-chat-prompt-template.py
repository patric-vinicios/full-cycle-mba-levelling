from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()

system = ("system", "You are an assistant that answers questions in a {style} style")
user = ("user", "I was fired because the company does not had tasks to me. What could I do?")

chat_prompt = ChatPromptTemplate([system, user])

messages = chat_prompt.format_messages(style="creative")

for msg in messages:
    print(f"{msg.type}: {msg.content}")


model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

result = model.invoke(messages)

print(result.content)

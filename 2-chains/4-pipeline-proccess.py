from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the text to japanese:\n ```{initial_text}```"
)

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 4 word in klingon: ```{text}```"
)

model = init_chat_model(model="gemini-2.5-flash", model_provider="google_genai")

translate = template_translate | model | StrOutputParser()
pipeline = {"text": translate} | template_summary | model | StrOutputParser()
result = pipeline.invoke({"initial_text": "LangChain is a framework to develop AI agents"})

print(result)

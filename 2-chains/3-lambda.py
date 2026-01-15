from langchain_core.runnables import RunnableLambda

def parse_text(text: str) -> str:
    return text.upper()

parsed_text_runnable = RunnableLambda(parse_text)

new_parsed_text = parsed_text_runnable.invoke("hello")

print(new_parsed_text)
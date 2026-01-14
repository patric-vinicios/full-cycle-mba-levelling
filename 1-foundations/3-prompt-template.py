from langchain.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["Patric"],
    template="Hi, my name is {name}. Tell me a joke with my name!"
)

text_formatted = template.format(name="Patric")

print(text_formatted)
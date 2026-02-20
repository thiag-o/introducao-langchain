from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

template_translate = PromptTemplate(
    input_variables=["initial_text"],
    template="Translate the following text to English:\n  ```{initial_text}``` ",
)

template_summary = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 5 words:\n  ```{text}``` ",
)

# template_translate_summary = PromptTemplate(
#     input_variables=["text"],
#     template="Translate the following text in 4 words:\n  ```{text}``` ",
# )

llm_en = ChatOpenAI(model="gpt-5-nano", temperature=0)

translate = template_translate | llm_en | StrOutputParser()
pipeline = {"text": translate} | template_summary | llm_en | StrOutputParser()

result = pipeline.invoke(
    {"initial_text": "Olá, meu nome é João e eu gosto de programar em Python."}
)
print(result)

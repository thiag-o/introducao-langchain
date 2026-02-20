from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import chain
from dotenv import load_dotenv
load_dotenv()

@chain
def square(input_dict: dict) -> dict:
    x = input_dict["x"]
    return {"square_result": x * x}

questionTemplate = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name!"
)
questionTemplate2 = PromptTemplate(
    input_variables=["square_result"],
    template="Tell me about the number {square_result}"
)

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5, max_completion_tokens=100)

chain = questionTemplate | model
chain2 = square | questionTemplate2 | model

# result = chain.invoke({"name": "Thiago"})
# print(result)
result2 = chain2.invoke({"x": 5})
print(result2.content)
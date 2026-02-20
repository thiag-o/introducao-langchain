from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()


template = PromptTemplate(
    input_variables=["name"],
    template="Hi, I'm {name}! Tell me a joke with my name!"
)

model = ChatOpenAI(model="gpt-5-nano", temperature=0.5)
chain = template | model
result = chain.invoke({"name": "Thiago"})
print(result)
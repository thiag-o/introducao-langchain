from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory, RunnableLambda
from langchain_core.messages import trim_messages

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "you are a helpful assistant that answers with a short joke when possible",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

llm = ChatOpenAI(model="gpt-5-nano", temperature=0.9)


def prepare_inputs(payload: dict) -> dict:
    raw_history = payload.get("raw_history", [])
    trimmed = trim_messages(
        raw_history,
        token_counter=len,
        max_tokens=2,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

    return {"input": payload["input"], "history": trimmed}


prepare = RunnableLambda(prepare_inputs)

chain = prepare | prompt | llm

session_store: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]


conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="raw_history",
)

config = {"configurable": {"session_id": "session-demo"}}


resp1 = conversational_chain.invoke(
    {"input": "Hi my name is Thiago. Reply OK and do not mention my name"},
    config=config,
)
print("Assistant:", resp1.content)
print("-" * 30)

resp2 = conversational_chain.invoke(
    {"input": "Tell me one-sentence fun fact. Do not metion my name"}, config=config
)
print("Assistant:", resp2.content)
print("-" * 30)

resp3 = conversational_chain.invoke({"input": "What is my name?"}, config=config)
print("Assistant:", resp3.content)

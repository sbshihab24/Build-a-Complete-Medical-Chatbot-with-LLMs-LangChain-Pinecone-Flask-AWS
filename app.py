from flask import Flask, render_template, request

from dotenv import load_dotenv

import os

from src.helpers import download_hugging_face_embeddings

from langchain_pinecone import PineconeVectorStore

from langchain_openai import ChatOpenAI

from langchain_classic.chains.retrieval import create_retrieval_chain

from langchain_classic.chains.combine_documents import (
    create_stuff_documents_chain
)

from langchain_core.prompts import ChatPromptTemplate

from src.prompt import system_prompt


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

chatModel = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(
    chatModel,
    prompt
)

rag_chain = create_retrieval_chain(
    retriever,
    question_answer_chain
)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/get", methods=["POST"])
def chat():

    msg = request.form["msg"]

    response = rag_chain.invoke(
        {
            "input": msg
        }
    )

    return response["answer"]


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=8080,
        debug=True
    )
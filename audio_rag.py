from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from transformers import pipeline
import argparse

def run_rag(input_text: str, question: str) -> str:
    # 1. Split text
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_text(input_text)

    # 2. Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # 3. Create vector store + retriever
    vectorstore = FAISS.from_texts(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 4. Load LLM
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256,
        device=-1
    )

    llm = HuggingFacePipeline(pipeline=pipe)

    # 5. Prompt template
    prompt = PromptTemplate.from_template("""
Answer the question using ONLY the context below.

Context:
{context}

Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 6. Build RAG chain
    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": lambda x: x
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Invoke chain
    response = rag_chain.invoke(question)

    return response


def media_query_anlyzr(input_text, question):
    print(f'Input Text: {input_text}')
    print("Question:", question)
    answer = run_rag(input_text, question)

    return answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--question", required=True)

    args = parser.parse_args()

    res = media_query_anlyzr(args.input, args.question)
    print("Answer:", res)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


class RAGAgent:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(model="llama3")
        self.llm = ChatOllama(model="llama3")

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

    def add_document(self, text, corpus_id):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        docs = splitter.create_documents([text])

        # Attach corpus metadata
        for doc in docs:
            doc.metadata = {"corpus_id": corpus_id}

        self.vectorstore.add_documents(docs)

    def query(self, question, corpus_id):
        retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "filter": {"corpus_id": corpus_id}
            }
        )
    
        docs = retriever.invoke(question)
    
        context = "\n\n".join([doc.page_content for doc in docs])
    
        prompt = f"""
        Use the context below to answer the question.
    
        Context:
        {context}
    
        Question:
        {question}
        """
    
        response = self.llm.invoke(prompt)
        return response.content
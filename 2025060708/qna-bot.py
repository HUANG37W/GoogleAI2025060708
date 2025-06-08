from langchain_community.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# load the vector store

prebuilt_faiss = FAISS.load_local(
    "faiss_db",
    embedding_function,
    "animal-fun-facts",
    allow_dangerous_deserialization=True
)

prebuilt_faiss.similarity_search_with_score("What is ship of the desert?", 3)
prebuilt_faiss.similarity_search_with_score("What's fun in Taipei city?", 3)
# def query_and_print(question):
#     results = prebuilt_faiss.similarity_search_with_score(question, 3)
#     if not results or results[0][1] > 1:
#         print(f"{question} -> Sorry, I don't know.")
#     else:
#         answer = results[0][0].page_content if hasattr(results[0][0], "page_content") else str(results[0][0])
#         print(f"{question} -> {answer}")

# query_and_print("What is ship of the desert?")
# query_and_print("What's fun in Taipei city?")

# question = "What is ship of the desert?"
question = "What's fun in Taipei city?"
results = prebuilt_faiss.similarity_search_with_score(question, 1)
if results[0][1] > 1:
    print(f"Q: {question}")
    print(f"A: 與動物無關")
else:
    print(f"Q: {question}")
    print(f"A: {results[0][0].page_content}")
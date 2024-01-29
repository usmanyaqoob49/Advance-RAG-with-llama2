from LLM_retrieval_pipeline import rag
from doc_retrieval_6 import retrieve_doc
from llama_1 import model_tokenizer
model, tokenizer= model_tokenizer()


query= 'What is Random Forest?'
path_to_doc= './Aurelien-Geron-Hands-On-Machine-Learning.pdf'


retrieved_documents= retrieve_doc(query, path_to_doc)

print(rag(query, retrieved_documents, model, tokenizer))
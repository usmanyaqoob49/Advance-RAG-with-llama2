import chromadb
from embed_4 import embedding_function
from chunks_3 import chunks

def doc_ingestion_in_vecdb(path_of_doc):
    #we will pass the path of doc to chunks that will further pass it to doc_load for loading it
    token_split_texts= chunks(path_of_doc)
    embedding_function= embedding_function()

    #making chromadb client object good for testing only not for production purpose
    chroma_client = chromadb.Client()

    #making the collection of chroma database
    chroma_collection = chroma_client.create_collection("Doc-Collection", embedding_function=embedding_function)

    #ids of each chunk
    ids = [str(i) for i in range(len(token_split_texts))]

    chroma_collection.add(ids=ids, documents=token_split_texts)
    return chroma_collection

from doc_ingestion_5 import doc_ingestion_in_vecdb

def retrieve_doc(query, path_to_doc):
    #we will pass the document path to doc_ingestion_in_vecdb() so that can ingest it in vector db after chunking and loading
    chroma_collection= doc_ingestion_in_vecdb(path_to_doc)

    #lets get the 5 relevant results
    results = chroma_collection.query(query_texts=[query], n_results=5)

    #[0] means give the result of the first query, right now we have only 1 query
    retrieved_documents = results['documents'][0]

    return retrieved_documents
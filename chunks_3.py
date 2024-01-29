from langchain.text_splitter import RecursiveCharacterTextSplitter, SentenceTransformersTokenTextSplitter
from doc_load_2 import load_doc
    #laoding the book by giving its path
def chunks(path_of_doc):
    
    #we will use the path of doc to pass it to load_doc
    pdf_texts= load_doc(path_of_doc)
    character_splitter = RecursiveCharacterTextSplitter(
    #RecursiveCharacterTextSplitter will split the Document into chunks firstly when it will find the double line
        #After that if the splitted chunks are greater than size of 1000 which is our chunk size they will get split on single line
            #Even if Chunks have larger size than 1000 they will get split on ". "
        separators=["\n\n", "\n", ". ", " ", ""],
        #every chunk will have 2000 characters
        chunk_size=2000,
        chunk_overlap=20
    )
    #further splitting according to embedding model we are using
    character_split_texts = character_splitter.split_text('\n\n'.join(pdf_texts))
    #so lets split it according to our embedding model
                                                                            #we want every token to have 256 characters
    token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
    token_split_texts = []
    for text in character_split_texts:
    #splitting the text and storing in list
        token_split_texts += token_splitter.split_text(text)

    return token_split_texts


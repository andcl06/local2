# document_processor.py
# 업로드된 문서를 처리하고 RAG 체인을 구축하는 모듈

# --- 1. 임포트 (파일 최상단) ---
import os
import uuid
import tempfile
from loguru import logger
from typing import List, Dict, Any

# --- LangChain 및 관련 라이브러리 임포트 ---
import tiktoken
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, UnstructuredPowerPointLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

# --- 2. 함수 정의 ---
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def get_text(docs):
    doc_list = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        logger.info(f"Using temporary directory: {tmp_dir}")
        for doc in docs:
            ext = os.path.splitext(doc.name)[1].lower()
            temp_filename = os.path.join(tmp_dir, f"{uuid.uuid4()}{ext}")
            try:
                with open(temp_filename, "wb") as file:
                    file.write(doc.getvalue())
                logger.info(f"Saved {doc.name} to temporary path: {temp_filename}")
            except Exception as e:
                logger.error(f"Error saving file {doc.name} to temp: {e}", exc_info=True)
                continue
            try:
                if ext == '.pdf': loader = PyPDFLoader(temp_filename)
                elif ext == '.docx': loader = Docx2txtLoader(temp_filename)
                elif ext == '.pptx': loader = UnstructuredPowerPointLoader(temp_filename)
                else:
                    logger.warning(f"Unsupported file type: {ext}")
                    continue
                documents = loader.load()
                doc_list.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {doc.name}")
            except Exception as e:
                logger.error(f"Error loading document from {doc.name}: {e}", exc_info=True)
                continue
    return doc_list

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=100, length_function=tiktoken_len)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Documents split into {len(chunks)} chunks.")
    return chunks

def get_vectorstore(text_chunks):
    try:
        embeddings = HuggingFaceEmbeddings(model_name="jhgan/ko-sroberta-multitask", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        vectordb = FAISS.from_documents(text_chunks, embeddings)
        logger.info("Vectorstore created using FAISS.")
        return vectordb
    except Exception as e:
        logger.error(f"Failed to create vectorstore: {e}", exc_info=True)
        return None

def get_conversation_chain(vectorstore, api_key):
    try:
        llm = ChatOpenAI(api_key=api_key, model_name='gpt-3.5-turbo', temperature=0, base_url="https://ai.potens.ai/api/chat")
        logger.info(f"ChatOpenAI model initialized with base_url: {llm.base_url}")
        return ConversationalRetrievalChain.from_llm(
            llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_type='mmr', verbose=True),
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'),
            get_chat_history=lambda h: h, return_source_documents=True, verbose=True
        )
    except Exception as e:
        logger.error(f"Failed to create conversation chain: {e}", exc_info=True)
        return None

# --- 3. 모듈 테스트를 위한 실행 블록 (선택 사항) ---
if __name__ == '__main__':
    # 이 파일을 직접 실행했을 때 테스트 코드가 실행됩니다.
    print("--- 문서 처리 모듈 단독 테스트 ---")
    print("이 부분은 Streamlit 앱 외부에서 테스트할 때만 실행됩니다.")
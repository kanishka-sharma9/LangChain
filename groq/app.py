import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
# from langchain_community.llms import ollama
import time


groq_api_key=os.getenv("GROQ_API_KEY")

if 'vector' not in st.session_state:
    st.session_state.embeddings=OllamaEmbeddings()
    st.session_state.loader=WebBaseLoader("https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:10])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title("chat groq")
llm=ChatGroq(api_key=groq_api_key,model="mixtral-8x7b-32768")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
please provide the most accurate response for the question.
<context>
{context}
</context>

<questions>: {input}

"""
)

doc_chain=create_stuff_documents_chain(llm,prompt=prompt)
retriever=st.session_state.vectors.as_retriever()
ret_chain=create_retrieval_chain(retriever,doc_chain)

input=st.text_input("enter prompt here:")

if input:
    start=time.process_time()
    resp=ret_chain.invoke({'input':input})
    print('response time:',time.process_time()-start)
    # st.write(resp)
    st.write(resp['answer'])
    with st.expander("doc similarity search"):
        for i,doc in enumerate(resp['context']):
            st.write(doc.page_content)
            st.write("---------------")

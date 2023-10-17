from typing import List
import pypdf
import random
import itertools
import pandas as pd
from io import StringIO
import streamlit as st
import os
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import SVMRetriever
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import QAGenerationChain
from langchain.retrievers import TFIDFRetriever
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

from text_utils import clean_pdf_text

@st.cache_data
def load_docs(files: List) -> str:
    all_text = ""
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = pypdf.PdfReader(file_path)
            file_content = ""
            for page in pdf_reader.pages:
                file_content += page.extract_text()
            file_content = clean_pdf_text(file_content)
            all_text += file_content
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            file_content = stringio.read()
            all_text += file_content
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    return all_text

@st.cache_data
def generate_eval(text: str, num_questions: int, chunk: int):
    n = len(text)
    starting_indices = [random.randint(0, n - chunk) for _ in range(num_questions)]
    sub_sequences = [text[i:i + chunk] for i in starting_indices]
    chain = QAGenerationChain.from_llm(ChatOpenAI(temperature=0))
    eval_set = []
    for i, b in enumerate(sub_sequences):
        try:
            qa = chain.run(b)
            eval_set.append(qa)
        except:
            st.warning('Error generating question %s.' % str(i + 1), icon="⚠️")
    eval_set_full = list(itertools.chain.from_iterable(eval_set))
    return eval_set_full

@st.cache_resource
def split_texts(text, chunk_size: int, overlap, split_method: str):
    if split_method == "RecursiveTextSplitter":
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap)
    elif split_method == "CharacterTextSplitter":
        text_splitter = CharacterTextSplitter(separator=" ",
                                              chunk_size=chunk_size,
                                              chunk_overlap=overlap)
    else:
        st.warning("`Split method not recognized. Using RecursiveCharacterTextSplitter`", icon="⚠️")
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,
                                                       chunk_overlap=overlap)

    split_text = text_splitter.split_text(text)
    return split_text

@st.cache_resource
def make_llm():
    chosen_model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    return chosen_model

@st.cache_resource
def make_retriever(splits, retriever_type, embedding_type, num_neighbors):
    if embedding_type == "OpenAI":
        embedding = OpenAIEmbeddings()
    elif embedding_type == "HuggingFace":
        embedding = HuggingFaceEmbeddings()
    else:
        st.warning("`Embedding type not recognized. Using OpenAI`", icon="⚠️")
        embedding = OpenAIEmbeddings()
    
    if retriever_type == "similarity-search":
        try:
            vector_store = FAISS.from_texts(splits, embedding)
        except ValueError:
            st.warning("`Error using OpenAI embeddings (disallowed TikToken token in the text). Using HuggingFace.`",
                       icon="⚠️")
            vector_store = FAISS.from_texts(splits, HuggingFaceEmbeddings())
        retriever_obj = vector_store.as_retriever(k=num_neighbors)
    elif retriever_type == "SVM":
        retriever_obj = SVMRetriever.from_texts(splits, embedding)
    elif retriever_type == "TF-IDF":
        retriever_obj = TFIDFRetriever.from_texts(splits)

    else:
        st.warning("`Retriever type not recognized. Using SVM`", icon="⚠️")
        retriever_obj = SVMRetriever.from_texts(splits, embedding)
    return retriever_obj

def make_chain(retriever) -> ConversationalRetrievalChain:
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, verbose=True)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    qa = ConversationalRetrievalChain.from_llm(llm=llm,
                                                  retriever=retriever,
                                                  memory=memory)
    return qa

def run(chain,  eval_set):
    predict = []
    gt_dataset = []
    questions = []
    for qa in eval_set:
        questions.append(qa["question"])
        
        gt_dataset.append(qa["answer"])
        
        predict.append(chain({"question": qa["question"]}))
        extracted_answers = [entry["answer"] for entry in predict]

    grade_output = pd.DataFrame(list(zip(questions, gt_dataset, extracted_answers)),
                                columns=['Question', 'Ground Truth', 'Prediction'])
    grade_output = grade_output.to_dict('records')
    return grade_output
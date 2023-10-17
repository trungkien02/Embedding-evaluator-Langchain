import os
import json
import streamlit as st
from langchain.evaluation import load_evaluator
from langchain.evaluation import EmbeddingDistance
from embedding_eval import load_docs, generate_eval, split_texts, make_retriever, make_chain, run

os.environ["OPENAI_API_KEY"] = <"Your openai key here">

with st.sidebar.form("user_input"):
    num_eval_questions = st.select_slider("`Number of eval questions`",
                                          options=[1, 5, 10, 15, 20], value=5)

    chunk_chars = st.select_slider("`Choose chunk size for splitting`",
                                   options=[500, 750, 1000, 1500, 2000], value=1000)

    overlap = st.select_slider("`Choose overlap for splitting`",
                               options=[0, 50, 100, 150, 200], value=100)

    split_method = st.radio("`Split method`",
                            ("RecursiveTextSplitter",
                             "CharacterTextSplitter"),
                            index=0)
    
    retriever_type = st.radio("`Choose retriever`",
                              ("TF-IDF",
                               "SVM",
                               "similarity-search"),
                              index=2)
    
    embeddings = st.radio("`Choose embeddings`",
                          ("HuggingFace",
                           "OpenAI"),
                          index=1)

    num_neighbors = st.select_slider("`Choose # chunks to retrieve`",
                                     options=[3, 4, 5, 6, 7, 8])


    submitted = st.form_submit_button("Submit evaluation")

st.header("`Embedding-eval`")

with st.form(key='file_inputs'):
    uploaded_file = st.file_uploader("`Please upload a file to evaluate (.txt or .pdf):` ",
                                     type=['pdf', 'txt'],
                                     accept_multiple_files=True)

    uploaded_eval_set = st.file_uploader("`[Optional] Please upload eval set (.json):` ",
                                         type=['json'],
                                         accept_multiple_files=False)

    submitted = st.form_submit_button("Submit files")


if uploaded_file:
    text = load_docs(uploaded_file)
    if not uploaded_eval_set:
        eval_set = generate_eval(text, num_eval_questions, 3000)
    else:
        eval_set = json.loads(uploaded_eval_set.read())

    splits = split_texts(text, chunk_chars, overlap, split_method)


    retriever = make_retriever(splits, retriever_type, embeddings, num_neighbors)

    chain = make_chain(retriever)

    grade_output = run(chain, eval_set)
    grade_output = [{'Question': item['Question'], 'Prediction': item['Prediction'], 'Ground Truth': item['Ground Truth']} for item in grade_output]
    score = []
    evaluator = load_evaluator("embedding_distance", distance_metric=EmbeddingDistance.EUCLIDEAN)
    for qa in grade_output:
        score.append(evaluator.evaluate_strings(prediction= qa["Prediction"], reference= qa["Ground Truth"]))
    st.header("`Results`")
    for i in range(len(grade_output)):
        grade_output[i]['Score'] = score[i]['score']
        st.info('``Question: {}``'.format(i+1))
        st.subheader(grade_output[i]['Question'])

        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown(':red[**Ground Truth:**]')
            st.write(grade_output[i]['Ground Truth'])
            st.markdown(':red[**Prediction:**]')
            st.write(grade_output[i]['Prediction'])

        with col2:
            st.markdown(':red[**Score:**]')
            st.write(grade_output[i]['Score'])
            if grade_output[i]['Score'] < 0.5:
                st.write('``CORRECT``')
            else:
                st.write('``INCORRECT``')

    st.balloons()
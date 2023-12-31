# Embedding evaluator Langchain

**Run**

`pip install -r requirement.txt`

`streamlit run main.py`

**Inputs**

`uploaded_file` - file to evaluate (.txt or .pdf)

`uploaded_eval_set` - file eval set (.json)

`num_eval_questions` - Number of questions to auto-generate (if the user does not supply an eval set)

`split_method` - Method for text splitting

`chunk_chars` - Chunk size for text splitting
 
`overlap` - Chunk overlap for text splitting
  
`embeddings` - Embedding method for chunks
 
`retriever_type` - Chunk retrieval method

`num_neighbors` - Neighbors for retrieval

**Output**

`Question` - Question generated by QAGenerateChain or taken from the eval set file

`Ground Truth` - Answer taken from the eval set file

`Prediction` - Answer generated by ConversationalRetrievalChain

`Score` - This returns a distance score, meaning that the lower the number, the more similar the prediction is to the reference, according to their embedded representation.

If Score < 0.5: return CORRECT else: return INCORRECT

**Example**

![image](https://github.com/trungkien02/Embedding-evaluator-Langchain/assets/74165010/dbc3ddbc-1477-462a-9daf-2d65587b3807)

![image](https://github.com/trungkien02/Embedding-evaluator-Langchain/assets/74165010/2279924f-5f19-4065-a0d6-df7bf1f3f741)


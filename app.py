import streamlit as st
from transformers import pipeline




def run_qa_app(model_path):

    question_answerer = pipeline("question-answering", model=model_path)
    

    uploaded_file = st.file_uploader("Choose a file")

    if uploaded_file is None:
        st.write("File is emtpy!")
    else:
        context_byte = uploaded_file.getvalue()
        context_str = context_byte.decode("utf-8")

        input_text = st.text_input("Question:", "what is the input about?")
        if st.button("Ask"):
            
            output_text = question_answerer(question=input_text, context=context_str)["answer"]
            st.write('Answer:', output_text)


if __name__ == "__main__":
    
    model_trained_path = "model/final_model"
    run_qa_app(model_path=model_trained_path)        
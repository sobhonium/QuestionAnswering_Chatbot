

from transformers import pipeline

question_answerer = pipeline("question-answering", model="model/final_model")


def run_qa_bot_cli():
    with open("inputs/context.txt", "r") as f:
        context = f.read()
    print("type `exit` to end the question answering and exit.")
    while (True):
        input_text = input(">> ")
        if input_text == "exit":
            break
        output_text = question_answerer(question=input_text, context=context)["answer"]
        print(output_text)
    


if __name__ == "__main__":
    run_qa_bot_cli()
import streamlit as st
import requests

def main():
    st.title("Text Storage App")

    # Input form
    text_input = st.text_input("Enter Text:")
    if st.button("Submit"):
        if text_input:
            response = requests.post("http://localhost:8000/add_text/", json={"text": text_input})
            if response.status_code == 200:
                st.success("Text added successfully!")

    # Display stored text
    st.header("Stored Text:")
    text_response = requests.get("http://localhost:8000/get_text/")
    if text_response.status_code == 200:
        texts = text_response.json()["texts"]
        for i, text in enumerate(texts, start=1):
            st.write(f"{i}. {text}")

if __name__ == "__main__":
    main()

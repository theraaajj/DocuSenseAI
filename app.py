import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables ( when in a local environment )
# load_dotenv()


# Configure the GenAI API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """Extract text from uploaded PDFs."""
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """Split text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings():
    """Return Google Generative AI embeddings."""
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_vector_store(text_chunks):
    """Generate vector store from text chunks using embeddings."""
    try:
        embeddings = get_embeddings()
        # Ensure text_chunks is in correct format before passing to FAISS
        if isinstance(text_chunks, list):
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
            st.success("Vector store generated successfully!")
        else:
            st.error("Text chunks are not in the correct format.")
    except Exception as e:
        st.error(f"Error generating vector store: {e}")

def get_conversational_chain():
    """Return the conversational chain for answering questions."""
    prompt_template = """
    Answer the question as detailed as possible from provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context", 
    don't provide the wrong answer.
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    """Process the user input and provide a response based on document context."""
    try:
        embeddings = get_embeddings()
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.markdown("### 🤖 DocuSense Reply:")
        st.success(response["output_text"])
    except Exception as e:
        st.error(f"Error occurred while processing your question: {e}")

def main():
    """Main function to run the DocuSense AI app."""
    # Page configuration
    st.set_page_config(page_title="DocuSense", layout="wide", page_icon=":books:")

    # Header
    st.markdown(
        """
        <h1 style='text-align: center; color: #4CAF50;'>📚DocuSenseAI</h1>
        <p style='text-align: center;'>Don't Worry! Your docs will start making sense.</p>
        """, unsafe_allow_html=True
    )
    st.divider()

    # Main question input area
    user_question = st.text_input("🤔 Shoot a question:")

    if user_question:
        with st.spinner("Finding the best answer..."):
            user_input(user_question)

    # Sidebar with dropdown menu integrated into the "Menu" option
    with st.sidebar:
        menu_option = st.selectbox("Menu", ["Upload", "About"], index=0)
        
        if menu_option == "Upload":
            st.title("📂 Instructions:")
            st.markdown(
                """
                <p style='font-size: 13px;'>1. Upload your PDF files: Click Browse to select your PDF files.</p>
                <p style='font-size: 13px;'>2. Process your PDFs: After uploading, click Submit & Process.</p>
                <p style='font-size: 13px;'>3. Ask questions & get your answer.</p>
                """, unsafe_allow_html=True
            )

        elif menu_option == "About":
            st.title("📂 About DocuSenseAI")
            st.markdown("""
                **DocuSenseAI** is an intelligent system built using Streamlit and Google Generative AI to process PDF documents.
                It enables you to upload PDFs, extract text, and then ask questions based on the content of the documents.
                The system uses advanced embeddings and question-answering techniques to ensure accurate responses to your queries.

                ### Key Features:
                - Upload and process multiple PDF files.
                - Ask detailed questions based on the context of the PDFs.
                - Powered by Google Generative AI for efficient text processing.
                - Easy-to-use interface built with Streamlit.

                ### How it Works:
                1. Upload PDF files using the sidebar.
                2. Click **Submit & Process** to analyze the document.
                3. Ask questions related to the content of your documents in the main input box.
                4. Get detailed answers based on the content of your PDFs.

                ### Technologies Used:
                - **Streamlit** for interactive web app
                - **Google Generative AI** for powerful embeddings and language model
                - **PyPDF2** for PDF text extraction
                - **LangChain** for document indexing and question answering
            """)

        # File upload and processing section
        st.markdown("## 📂 Upload PDFs:")
        pdf_docs = st.file_uploader("", type=["pdf"], accept_multiple_files=True)
        
        if pdf_docs:
            if st.button("Submit & Process"):
                with st.spinner("Processing your documents..."):
                    # for i, pdf in enumerate(pdf_docs):
                    #     st.progress((i + 1) / len(pdf_docs))
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Your files have been processed!")

    # Footer
    st.markdown(
        """
        ---
        <footer style="text-align: center; font-size: 12px;">
        Made by Raj Aryan | Powered by Google Generative AI
        </footer>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

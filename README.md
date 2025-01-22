# DocuSenseAI

An interactive NLP-based application designed to simplify document understanding and provide AI-powered knowledge retrieval. Users can upload files, process them, and receive AI-generated responses based on the content of their documents.

## Features

- **File Upload and Processing:** Upload various document formats for processing.
- **AI-Powered Responses:** Generate intelligent answers based on document content using advanced NLP techniques.
- **Interactive Interface:** Easy-to-use interface built with Streamlit.
- **Efficient Search:** Integrated with FAISS for quick similarity search within documents.
- **Seamless Deployment:** Hosted on Render for reliability and scalability.

## Technology Stack

- **Frontend:**
  - [Streamlit](https://streamlit.io/): For building an interactive user interface.

- **Backend:**
  - [Google Generative AI](https://ai.google/): Powers the AI-generated responses.
  - [PyPDF2](https://pypi.org/project/PyPDF2/): Handles PDF file parsing.
  - [faiss-cpu](https://github.com/facebookresearch/faiss): Enables efficient similarity search.
  - [LangChain](https://langchain.com/): For building and managing the NLP pipeline.

- **Deployment:**
  - [Render](https://render.com/): Used for hosting and deployment.

## Installation

To run DocuSenseAI locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/theraaajj/DocuSenseAI.git
   ```

2. Navigate to the project directory:
   ```bash
   cd DocuSenseAI
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables in Render or create a `.env` file locally with the following:
   ```env
   GOOGLE_API_KEY=your_google_api_key
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

6. Access the app locally at `http://localhost:8501`.

## Usage

1. Upload a document using the file uploader in the app.
2. The system processes the document and indexes its content.
3. Ask questions or query information based on the uploaded document.
4. Get AI-generated responses tailored to your input.

## Example Use Case

- **Legal Documents:** Quickly extract specific clauses or legal information.
- **Research Papers:** Retrieve key insights or summaries from lengthy academic papers.
- **Business Reports:** Access vital information from business presentations and reports.
- **Educational Content:** Simplify study sessions by querying notes or textbooks for relevant information.
- **Healthcare:** Extract insights from medical records or research papers for healthcare professionals.


## Future Enhancements

- **Support for Additional File Formats:** Extend compatibility to include Word documents, Excel spreadsheets, and more.
- **Multilingual Support:** Enable processing of documents in multiple languages.
- **Enhanced Query Capabilities:** Introduce natural language understanding for more complex queries.
- **Advanced Summarization:** Provide concise and context-aware document summaries.
- **User Authentication:** Add user authentication for secure access to the platform.


## Contributions

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m 'Add some feature'
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Open a pull request.


## Acknowledgments

- [Google Generative AI](https://ai.google/)
- [Streamlit](https://streamlit.io/)
- [Render](https://render.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [LangChain](https://langchain.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)

---

For any questions, feedback, or suggestions, feel free to reach out via the repository's issue tracker or contact the maintainer directly.


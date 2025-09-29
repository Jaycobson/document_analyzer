# DocuMind AI 🧠📄

Meet DocuMind AI! 🚀 An intelligent document analysis system that lets you chat seamlessly with your documents (PDF, DOCX, TXT) using Langchain, Google Gemini Pro & FAISS Vector DB with beautiful Streamlit interface. 

## 📝 Description
DocuMind AI is a powerful Streamlit-based web application designed to facilitate interactive conversations with your documents. The app allows users to upload multiple documents in various formats (PDF, DOCX, TXT), extract and analyze text content, and engage in intelligent Q&A sessions powered by Google's Gemini AI. With advanced features like keyword extraction, word cloud visualization, and document statistics, DocuMind AI provides comprehensive document intelligence at your fingertips.

## 📢 Demo App with Streamlit Cloud

[Launch App On Streamlit](https://documentanalyzer-llm.streamlit.app/)


## 🎯 How It Works:

The application follows these steps to provide intelligent responses to your questions:

1. **Multi-Format Document Loading**: The app reads PDF, DOCX, and TXT documents and extracts their text content with specialized parsers.

2. **Intelligent Text Chunking**: The extracted text is divided into optimized chunks (1500 characters with 200-character overlap) using RecursiveCharacterTextSplitter for effective processing and context preservation.

3. **Vector Embeddings**: The application utilizes Google's text-embedding-004 model to generate high-quality vector representations (embeddings) of the text chunks.

4. **FAISS Vector Store**: Embeddings are stored in a FAISS (Facebook AI Similarity Search) vector database for lightning-fast semantic similarity searches.

5. **Keyword Extraction & Analytics**: The system automatically extracts top keywords with frequency analysis and generates visual insights including word clouds and document statistics.

6. **Semantic Similarity Matching**: When you ask a question, the app performs vector similarity search to identify the most semantically relevant text chunks from your documents.

7. **Context-Aware Response Generation**: The selected chunks are passed to Google Gemini 2.5 Pro language model, which generates accurate, context-aware responses based on the relevant content of your documents.


## 🌟 Requirements

- **streamlit**: Framework for building the interactive web application interface
- **streamlit-components-v1**: For custom component integration
- **PyPDF2**: Library for reading and extracting text from PDF files
- **python-docx**: For parsing and extracting text from DOCX files
- **langchain**: Framework for building applications with LLMs, including text splitting and chain creation
- **langchain-google-genai**: Integration for Google Gemini AI models and embeddings
- **google-generativeai**: Official Google SDK for generative AI capabilities
- **faiss-cpu**: Facebook AI Similarity Search library for efficient vector similarity search
- **python-dotenv**: For loading environment variables from .env files
- **wordcloud**: Generate beautiful word cloud visualizations
- **matplotlib**: Plotting library for data visualization
- **plotly**: Interactive graphing library for advanced visualizations
- **pandas**: Data manipulation and analysis library


## 🏗️ Technical Architecture

### RAG Pipeline Components:

1. **Document Loaders**: Custom extractors for PDF (PyPDF2), DOCX (python-docx), and TXT files
2. **Text Splitter**: RecursiveCharacterTextSplitter with optimized parameters
3. **Embeddings**: Google text-embedding-004 model (768 dimensions)
4. **Vector Store**: FAISS with L2 distance metric for similarity search
5. **LLM**: Google Gemini 2.5 Pro with custom prompt template
6. **Chain**: LangChain QA chain with "stuff" strategy
7. **Analytics Engine**: Custom NLP pipeline for keyword extraction and visualization

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/Jaycobson/documind-ai.git
cd documind-ai
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Set up your Google API key from `https://makersuite.google.com/app/apikey` by creating a `.env` file in the root directory of the project with the following contents:

```
GOOGLE_API_KEY=your-api-key-here
```

Run the Streamlit app:

```bash
streamlit run app.py
```

---

## 💡 Usage

To use DocuMind AI, you can preview the app by clicking: [Launch App On Streamlit](https://documind-ai.streamlit.app/). To run the app locally, follow these steps:

### Quick Start:

1. **Setup Environment**: Ensure you have installed all required dependencies and added your **Google API key to the `.env` file** (REQUIRED).

2. **Launch Application**: Run the `app.py` file using the Streamlit CLI:
   ```bash
   streamlit run app.py
   ```

3. **Access Interface**: The application will launch in your default web browser with an intuitive dashboard.

4. **Upload Documents**: 
   - Click on the file uploader in the main interface
   - Select one or multiple documents (PDF, DOCX, or TXT)
   - Supported formats: `.pdf`, `.docx`, `.txt`

5. **Process Documents**: 
   - Click the "🔄 Analyze Documents" button
   - Watch the real-time progress as the system:
     - Extracts text from your documents
     - Creates intelligent text chunks
     - Extracts keywords and statistics
     - Builds the vector knowledge base

6. **View Analytics**: 
   - Explore document statistics (word count, character count, file metrics)
   - Review top keywords with frequency counts
   - Visualize content with the interactive word cloud

7. **Ask Questions**: 
   - Type your question in the chat input field
   - Click "🤔 Ask Question" or press Enter
   - Receive instant AI-powered answers based on your document content
   - Questions are answered using semantic search across all uploaded documents

8. **Export & Save**: Results and analytics can be reviewed and referenced throughout your session.

### Example Questions:
- "What are the main topics discussed in these documents?"
- "Summarize the key findings from the research paper"
- "What does the document say about [specific topic]?"
- "Compare the information across multiple documents"

---

## 🛠️ Configuration

### Customizable Parameters:

**Text Chunking** (in `get_text_chunks()`):
- `chunk_size`: Default 1500 characters
- `chunk_overlap`: Default 200 characters
- Adjust based on your document complexity

**Keyword Extraction** (in `extract_keywords()`):
- `num_keywords`: Default 20 keywords
- Modify for more or fewer keyword results

**Similarity Search** (in `process_query()`):
- `k`: Default 4 most similar chunks
- Increase for broader context, decrease for focused responses

**LLM Settings** (in `get_conversational_chain()`):
- `temperature`: Default 0.3 (lower = more focused, higher = more creative)
- `model`: Gemini 2.5 Pro (modify for other models)

---

## 📊 Features in Detail

### 1. Document Processing
- Parallel processing of multiple documents
- Format-specific text extraction
- Encoding handling for various file types
- Error handling and validation

### 2. Vector Store
- FAISS indexing for efficient retrieval
- Persistent storage of embeddings
- Semantic similarity search
- Scalable to large document collections

### 3. Analytics Dashboard
- Real-time document statistics
- Keyword extraction with stop-word filtering
- Frequency analysis and ranking
- Visual word clouds with customizable themes

### 4. AI-Powered Q&A
- Context-aware responses
- Multi-document support
- Source attribution
- Customizable prompt templates

---

## 🐛 Troubleshooting

**Issue**: "Error creating vector store"
- **Solution**: Ensure GOOGLE_API_KEY is properly set in .env file

**Issue**: "No text could be extracted"
- **Solution**: Verify document is not corrupted and contains extractable text

**Issue**: "Module not found errors"
- **Solution**: Run `pip install -r requirements.txt` to install all dependencies

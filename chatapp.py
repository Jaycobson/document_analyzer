import streamlit as st
import streamlit.components.v1 as components
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import io
import base64

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Page configuration
st.set_page_config(
    page_title="DocuMind AI", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS for modern look
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .upload-section {
        background: linear-gradient(145deg, #f0f2f6, #ffffff);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        margin-bottom: 2rem;
    }
    
    .stats-card {
        background: linear-gradient(145deg, #ffffff, #f8f9fa);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    .keyword-chip {
        display: inline-block;
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        margin: 0.25rem;
        border-radius: 25px;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    .chat-container {
        background: linear-gradient(145deg, #f8f9fa, #ffffff);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .processing-animation {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        padding: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def extract_text_from_pdf(pdf_file):
    """Extract text from PDF files"""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
    return text

def extract_text_from_docx(docx_file):
    """Extract text from DOCX files"""
    text = ""
    try:
        doc = docx.Document(docx_file)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
    except Exception as e:
        st.error(f"Error reading DOCX: {str(e)}")
    return text

def extract_text_from_txt(txt_file):
    """Extract text from TXT files"""
    try:
        # Read bytes and decode
        content = txt_file.read()
        if isinstance(content, bytes):
            text = content.decode('utf-8')
        else:
            text = content
        return text
    except Exception as e:
        st.error(f"Error reading TXT: {str(e)}")
        return ""

def get_document_text(uploaded_files):
    """Extract text from multiple document types"""
    text = ""
    file_info = []
    
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        file_text = ""
        
        if file_extension == 'pdf':
            file_text = extract_text_from_pdf(uploaded_file)
        elif file_extension == 'docx':
            file_text = extract_text_from_docx(uploaded_file)
        elif file_extension == 'txt':
            file_text = extract_text_from_txt(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {uploaded_file.name}")
            continue
        
        text += file_text
        file_info.append({
            'name': uploaded_file.name,
            'type': file_extension.upper(),
            'size': len(file_text),
            'words': len(file_text.split())
        })
    
    return text, file_info

def get_text_chunks(text):
    """Split text into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """Create and save vector store"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
        return True
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return False

def extract_keywords(text, num_keywords=20):
    """Extract keywords from text"""
    # Clean text
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    words = text.split()
    
    # Remove common stop words
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'])
    
    # Filter words
    filtered_words = [word for word in words if len(word) > 3 and word not in stop_words]
    
    # Count frequencies
    word_freq = Counter(filtered_words)
    return word_freq.most_common(num_keywords)

def create_wordcloud(text):
    """Create word cloud visualization"""
    try:
        wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        
        return fig
    except Exception as e:
        st.error(f"Error creating word cloud: {str(e)}")
        return None

def get_conversational_chain():
    """Create conversational chain"""
    prompt_template = """
    You are an intelligent document assistant. Answer the question based on the provided context with detailed and accurate information.
    If the answer is not available in the context, clearly state that the information is not available in the documents.
    
    Context:\n{context}\n
    Question:\n{question}\n
    
    Answer:
    """
    
    try:
        model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain
    except Exception as e:
        st.error(f"Error creating conversational chain: {str(e)}")
        return None

def process_query(user_question):
    """Process user query and return response"""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = vector_store.similarity_search(user_question, k=4)
        
        chain = get_conversational_chain()
        if chain:
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True
            )
            return response["output_text"]
    except Exception as e:
        return f"Error processing query: {str(e)}"
    
    return "Sorry, I couldn't process your question at the moment."

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 Document Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent Document Analysis & Question Answering System</p>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = False
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'file_info' not in st.session_state:
        st.session_state.file_info = []
    if 'keywords' not in st.session_state:
        st.session_state.keywords = []
    
    # Upload Section
    with st.container():
 
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📁 Upload Your Documents")
            uploaded_files = st.file_uploader(
                "Choose files",
                type=['pdf', 'docx', 'txt'],
                accept_multiple_files=True,
                help="Supported formats: PDF, DOCX, TXT"
            )
        
        with col2:
            st.markdown("### 🚀 Process Documents")
            process_button = st.button(
                "🔄 Analyze Documents", 
                type="primary",
                use_container_width=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Processing
    if process_button and uploaded_files:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Extract text
            status_text.text("📖 Extracting text from documents...")
            progress_bar.progress(20)
            
            document_text, file_info = get_document_text(uploaded_files)
            
            if not document_text.strip():
                st.error("No text could be extracted from the uploaded documents.")
                return
            
            # Step 2: Create chunks
            status_text.text("🔪 Creating text chunks...")
            progress_bar.progress(40)
            
            text_chunks = get_text_chunks(document_text)
            
            # Step 3: Extract keywords
            status_text.text("🔍 Extracting keywords...")
            progress_bar.progress(60)
            
            keywords = extract_keywords(document_text)
            
            # Step 4: Create vector store
            status_text.text("🧠 Creating knowledge base...")
            progress_bar.progress(80)
            
            if get_vector_store(text_chunks):
                # Step 5: Complete
                status_text.text("✅ Processing complete!")
                progress_bar.progress(100)
                
                # Store in session state
                st.session_state.documents_processed = True
                st.session_state.document_text = document_text
                st.session_state.file_info = file_info
                st.session_state.keywords = keywords
                
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                st.success("🎉 Documents processed successfully! You can now ask questions.")
             
            
        except Exception as e:
            st.error(f"❌ Error processing documents: {str(e)}")
            progress_bar.empty()
            status_text.empty()
    
    # Document Analysis Section
    if st.session_state.documents_processed:
        st.markdown("---")
        st.markdown("## 📊 Document Analysis")
        
        # Statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_words = sum([info['words'] for info in st.session_state.file_info])
        total_chars = sum([info['size'] for info in st.session_state.file_info])
        
        with col1:
            st.metric("📄 Files Processed", len(st.session_state.file_info))
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.metric("📝 Total Words", f"{total_words:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.metric("🔤 Characters", f"{total_chars:,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.metric("🔑 Keywords Found", len(st.session_state.keywords))
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        
        # with col1:
        st.markdown("### 🏷️ Top Keywords")
        keywords_html = ""
        for keyword, freq in st.session_state.keywords[:15]:
            keywords_html += f'<span class="keyword-chip">{keyword} ({freq})</span>'
        st.markdown(keywords_html, unsafe_allow_html=True)
        
        # Word Cloud
        st.markdown("### ☁️ Word Cloud")
        wordcloud_fig = create_wordcloud(st.session_state.document_text)
        if wordcloud_fig:
            st.pyplot(wordcloud_fig, use_container_width=True,width = 'content')
        
        # Chat Section
        st.markdown("---")
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        st.markdown("## 💬 Ask Questions About Your Documents")
        
        # Chat interface
        user_question = st.text_input(
            "What would you like to know?",
            placeholder="Ask me anything about your uploaded documents...",
            help="Type your question and press Enter"
        )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            ask_button = st.button("🤔 Ask Question", type="primary")
        
        with col2:
            clear_button = st.button("🗑️ Clear Chat")
        
        if ask_button and user_question:
            with st.spinner("🤖 Thinking..."):
                response = process_query(user_question)
                
                st.markdown("### 🤖 AI Response:")
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #e3f2fd, #ffffff); 
                           border-radius: 10px; padding: 1.5rem; margin: 1rem 0;
                           border-left: 4px solid #667eea;">
                    {response}
                </div>
                """, unsafe_allow_html=True)
        
        if clear_button:
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; padding: 1rem;'>"
        "Made with ❤️ using Streamlit | Powered by Google Gemini AI"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()










import streamlit as st
import os
import time
import pinecone
import PyPDF2
from io import StringIO
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import openpyxl

# Setting up Streamlit page configuration
st.set_page_config(
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Getting the OpenAI API key from Streamlit Secrets
openai_api_key = st.secrets.secrets.OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = openai_api_key

# Getting the Pinecone API key and environment from Streamlit Secrets
PINECONE_API_KEY = st.secrets.secrets.PINECONE_API_KEY
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
PINECONE_ENV = st.secrets.secrets.PINECONE_ENV
os.environ["PINECONE_ENV"] = PINECONE_ENV

@st.cache_data
def load_docs(files):
    all_text = []
    for file_path in files:
        file_extension = os.path.splitext(file_path.name)[1]
        if file_extension == ".pdf":
            pdf_reader = PyPDF2.PdfReader(file_path)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            all_text.append(text)
        elif file_extension == ".txt":
            stringio = StringIO(file_path.getvalue().decode("utf-8"))
            text = stringio.read()
            all_text.append(text)
        elif file_extension == ".xlsx":
            workbook = openpyxl.load_workbook(file_path)
            sheet = workbook.active
            # Iterate over rows in the sheet
            for row in sheet.iter_rows(values_only=True):
                # Convert each cell value to a string and join them with a delimiter
                row_string = ', '.join(str(cell) for cell in row)
                
                # Append the row string to the list
                all_text.append(row_string)
        else:
            st.warning('Please provide txt or pdf.', icon="⚠️")
    # st.write(all_text)
    return all_text  

if 'clicked' not in st.session_state:
    st.session_state.clicked = False

def click_button():
    st.session_state.clicked = True

# Initialize Pinecone with API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

# Checkbox for the first option to document upload
first_opt = st.checkbox('Create New Index to Upload Documents')
# Checkbox for the second option to document upload
second_opt = st.checkbox('Use Existing Index to Upload Documents')
# Checkbox for the third option to delete existing indexes
third_opt = st.checkbox('Delete Any Existing Index')

st.write("---")

if first_opt:
    # Prompt the user to upload PDF/TXT/XLSX files
    st.write("Upload PDF/TXT/XLSX Files:")
    uploaded_files = st.file_uploader("Upload", type=["pdf", "txt", "xlsx"], label_visibility="collapsed", accept_multiple_files = True)

    if uploaded_files != []:
        documents = load_docs(uploaded_files)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(documents)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')

        # Display the uploaded file content
        file_container = st.expander(f"Click here to see your uploaded content:")
        file_container.write(docs)

        # Display success message
        st.success("Document Loaded Successfully!")
        pinecone_index = st.text_input("Enter the name of Index: ")
        if pinecone_index != "":
            st.info('Initializing Index Creation...')
            # Create a new Pinecone index
            pinecone.create_index(
                    name=pinecone_index,
                    metric='cosine',
                    dimension=1536  # 1536 dim of text-embedding-ada-002
                    )
            st.success('Index Successfully Created!')
            time.sleep(80)
            st.info('Initializing Document Uploading to DB...')
            # Upload documents to the Pinecone index
            vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
            
            # Display success message
            st.success("Document Uploaded Successfully!")

elif second_opt:
    # Prompt the user to upload PDF/TXT/XLSX files
    st.write("Upload PDF/TXT/XLSX Files:")
    uploaded_files = st.file_uploader("Upload", type=["pdf", "txt", "xlsx"], label_visibility="collapsed", accept_multiple_files = True)

    if uploaded_files != []:
        documents = load_docs(uploaded_files)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(documents)

        # Initialize OpenAI embeddings
        embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
        # Display success message
        st.success("Document Loaded Successfully!")
        # Display the uploaded file content
        file_container = st.expander(f"Click here to see your uploaded content:")
        file_container.write(docs)

        time.sleep(10)
        st.write("Existing Indexes:👇")
        st.write(pinecone.list_indexes())
        pinecone_index = st.text_input("Write Name of Existing Index: ")
        up_check = st.checkbox('Check this to Upload Docs in Selected Index')
        if up_check:
            st.info('Initializing Document Uploading to DB...')
            # Upload documents to the Pinecone index
            time.sleep(30)
            vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
            
            # Display success message
            st.success("Document Uploaded Successfully!")

elif third_opt:
    time.sleep(10)
    st.write("Existing Indexes:👇")
    st.write(pinecone.list_indexes())
    pinecone_index = st.text_input("Write Name of Existing Index to delete: ")
    st.write(f"The Index named '{pinecone_index}' is selected for deletion.")
    del_check = st.checkbox('Check this to Delete Index')
    if del_check:
        with st.spinner('Deleting Index...'):
            time.sleep(5)
            pinecone.delete_index(pinecone_index)
            time.sleep(10)
        st.success(f"'{pinecone_index}' Index Deleted Successfully!")
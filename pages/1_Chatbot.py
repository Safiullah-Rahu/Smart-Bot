import streamlit as st
import os
import time
import datetime
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import AgentType, initialize_agent, load_tools, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryBufferMemory
from langchain import OpenAI

# Setting up Streamlit page configuration
st.set_page_config(
    layout="centered", 
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
# Initialize Pinecone with API key and environment
pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
def select_index():
    #time.sleep(10)
    st.sidebar.write("Existing Indexes:👇")
    st.sidebar.write(pinecone.list_indexes())
    pinecone_index = st.sidebar.text_input("Write Name of Index to load: ")
    return pinecone_index

# Set the text field for embeddings
text_field = "text"
# Create OpenAI embeddings
embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
model_name = st.sidebar.selectbox(label="Select Model", options=MODEL_OPTIONS)

pinecone_index = select_index()

def chat(pinecone_index):

    if pinecone_index != "":
        # load a Pinecone index
        time.sleep(5)
        index = pinecone.Index(pinecone_index)
        db = Pinecone(index, embeddings.embed_query, text_field)
        retriever = db.as_retriever()

    def agent_meth():
        search = DuckDuckGoSearchRun()

        llm = OpenAI(model_name = model_name, streaming=True)
        doc_retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, verbose=True)
        tools = [
                Tool(
                    name = "Search",
                    func = search.run,
                    description="useful for when you need to answer questions about current events"
                ),
                Tool(
                    name = "Knowledge Base",
                    func = doc_retriever.run,
                    description="Always use Knowledge Base more than normal Search tool. Useful for general questions about how to do things and for details on interesting topics. Input should be a fully formed question."
                )
            ]
        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
        agent = initialize_agent(tools, 
                                llm, 
                                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                verbose=True, 
                                handle_parsing_errors=True,
                                memory = memory
                            )
        return agent
    def retr():
        llm = ChatOpenAI(model_name = model_name, streaming=True)
        memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100)
        agent = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, memory = memory, verbose=True)
        return agent


    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model_name

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    #agent = conversational_chat()
    st.sidebar.write("---")
    st.sidebar.write("Enable Web Access")
    meth_sw = st.sidebar.checkbox("Web Search")
    st.sidebar.write("---")
    if prompt := st.chat_input():
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content":prompt})
        # st.chat_message("user").write(prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            st_callback = StreamlitCallbackHandler(st.container())
            if meth_sw:
                agent = agent_meth()
                response = agent.run(prompt, callbacks=[st_callback])
            else:
                agent = retr()
                with st.spinner("Thinking..."):
                    response = agent.run(prompt)#, callbacks=[st_callback])
            #st.write(response)
            message_placeholder.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if pinecone_index != "":
    chat(pinecone_index)
    #st.sidebar.write(st.session_state.messages)
    #don_check = st.sidebar.button("Download Conversation")
    con_check = st.sidebar.button("Upload Conversation to loaded Index")
    
    text = []
    for item in st.session_state.messages:
        text.append(f"Role: {item['role']}, Content: {item['content']}\n")
    #st.sidebar.write(text)
    if con_check:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(text)
        st.sidebar.info('Initializing Conversation Uploading to DB...')
        time.sleep(11)
        # Upload documents to the Pinecone index
        vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
        
        # Display success message
        st.sidebar.success("Conversation Uploaded Successfully!")
    
    text = '\n'.join(text)
    # Provide download link for text file
    st.sidebar.download_button(
        label="Download Conversation",
        data=text,
        file_name=f"Conversation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
        mime="text/plain"
    )

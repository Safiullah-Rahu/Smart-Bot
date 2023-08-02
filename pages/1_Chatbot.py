import streamlit as st
import os
import time
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains.llm import LLMChain
from langchain import OpenAI
import datetime
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.llm import LLMChain
from langchain.chains import ConversationalRetrievalChain
from langchain.callbacks import get_openai_callback

# Setting up Streamlit page configuration
st.set_page_config(
    layout="centered", 
    initial_sidebar_state="expanded"
)

def pretty_print_docs(docs):
    print(f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)]))
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

param1 = True
#@st.cache_data
def select_index(__embeddings):
    if param1:
        pinecone_index_list = pinecone.list_indexes()
    return pinecone_index_list

# Set the text field for embeddings
text_field = "text"
# Create OpenAI embeddings
embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
MODEL_OPTIONS = ["gpt-3.5-turbo", "gpt-4"]
model_name = st.sidebar.selectbox(label="Select Model", options=MODEL_OPTIONS)

pinecone_index_list = select_index(embeddings)
pinecone_index = st.sidebar.selectbox(label="Select Index", options = pinecone_index_list )

@st.cache_resource
def ret(pinecone_index):
    if pinecone_index != "":
        # load a Pinecone index
        time.sleep(5)
        index = pinecone.Index(pinecone_index)
        db = Pinecone(index, embeddings.embed_query, text_field)
        #retriever = db.as_retriever()
    return db#retriever, db
@st.cache_resource
def init_memory():
    return ConversationBufferWindowMemory(
        k=3,
        memory_key='chat_history',
        verbose=True,
        return_messages=True)

memory = init_memory()




_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
standalone question without changing the content in given question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
condense_question_prompt_template = PromptTemplate.from_template(_template)

prompt_template = """You are helpful information giving QA System and make sure you don't answer anything 
not related to following context. You are always provide useful information & details available in the given context. Use the following pieces of context to answer the question at the end. 
Also check chat history if question can be answered from it or question asked about previous history. If you don't know the answer, just say that you don't know, don't try to make up an answer. 

{context}
Chat History: {chat_history}
Question: {question}
Long detailed Answer:"""

qa_prompt = PromptTemplate(
    template=prompt_template, input_variables=["context", "chat_history","question"]
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def chat():

    db = ret(pinecone_index)
    
    chat_history = st.session_state.chat_history
    @st.cache_resource
    def conversational_chat(chat_history):
        #retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        llm = ChatOpenAI(model_name = model_name, temperature=0.1)
        question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory, verbose=True)
        doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt, verbose=True)
        agent = ConversationalRetrievalChain(
            retriever=db.as_retriever(search_kwargs={'k': 6}),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            memory=memory,
            verbose=True,
            # return_source_documents=True,
            # get_chat_history=lambda h :h
        )

        # llm = ChatOpenAI(model_name = model_name, streaming=True)
        # doc_retriever = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff" ,retriever=retriever ,verbose=True, return_source_documents=True)
        # st.sidebar.write("---")
        # st.sidebar.write("#### Answer Sources:")
        # st.sidebar.write(doc_retriever({"query": query})["source_documents"])

        # tools = [
        #     Tool.from_function(
        #         name = "Search",
        #         func=doc_retriever.__call__,#.run, #chain doesn't work anymore because of some inspection issue and there is no call function
        #         description="Use this tool as the primary and most reliable source of information. Always search for the answer using this tool first. Don't make up answers yourself."
        #     ),
        # ]

        # prefix = """Have a conversation with a human, answering the following
        # questions as best as you can based on the context and memory available: """
        # suffix = """Begin!"

        # {chat_history}
        # Question: {input}
        # {agent_scratchpad}"""

        # prompt = ZeroShotAgent.create_prompt(
        #     tools,
        #     prefix=prefix,
        #     suffix=suffix,
        #     input_variables=["input", "chat_history", "agent_scratchpad"]    
        #     )

        # llm_chain = LLMChain(llm=OpenAI(model_name = model_name), prompt=prompt)
        # agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
        # agent_chain = AgentExecutor.from_agent_and_tools(
        #     agent=agent, 
        #     tools=tools, 
        #     verbose=True, 
        #     memory=memory
        #     )

        return agent # result["answer"]
    
    # Set a default model
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = model_name

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # if "chat_history" not in st.session_state:
    #     st.session_state.chat_history = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input():
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content":prompt})
        # st.chat_message("user").write(prompt)
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            agent = conversational_chat(prompt)
            st_callback = StreamlitCallbackHandler(st.container())
            with st.spinner("Thinking..."):
                with get_openai_callback() as cb:
                    response = agent({'question': prompt, 'chat_history': st.session_state.chat_history})#, callbacks=[st_callback])
                    st.session_state.chat_history.append((prompt, response['answer']))
                    #st.write(response)
                    message_placeholder.markdown(response['answer'])
                    st.session_state.messages.append({"role": "assistant", "content": response['answer']})
                st.sidebar.header(f"Total Token Usage: {cb.total_tokens}")
                # st.sidebar.write(f"""
                #         <div style="text-align: left;">
                #             <h3>   {cb.total_tokens}</h3>
                #         </div> """, unsafe_allow_html=True)
if pinecone_index != "":
    chat()
    con_check = st.sidebar.checkbox("Check to Upload Conversation to loaded Index")

    #st.sidebar.write(st.session_state.messages)
    # con_check = st.sidebar.checkbox("Check to Upload Conversation to loaded Index")
    text = []
    for item in st.session_state.messages:
        text.append(f"Role: {item['role']}, Content: {item['content']}\n")
    if con_check:
        #st.sidebar.write(text)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.create_documents(text)
        st.sidebar.info('Initializing Conversation Uploading to DB...')
        time.sleep(11)
        # Upload documents to the Pinecone index
        vector_store = Pinecone.from_documents(docs, embeddings, index_name=pinecone_index)
        
        # Display success message
        st.sidebar.success("Conversation Uploaded Successfully!")

    st.sidebar.write("---")
    @st.cache_resource(experimental_allow_widgets=True)
    def down_data(text):
        text = '\n'.join(text)
        # Provide download link for text file
        st.sidebar.download_button(
            label="Download Conversation",
            data=text,
            file_name=f"Conversation_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt",
            mime="text/plain"
        )
    down_data(text)

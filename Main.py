import streamlit as st

st.set_page_config(
    page_title="IntelliBot",
    page_icon="🤖",
    layout="wide"
)

st.title("🌐 IntelliBot: Your Smart Conversational Assistant 🤖💬")
st.write("")
st.write("")
st.write("")
description = """
    <div style="text-align: center;">
        <h3>🚀 Discover DataBotX today and embark on a transformative data-driven experience!</h3>
    </div> """
# text-align: center;

st.write("""<div align="center">

</div>

Introducing **IntelliBot**, a state-of-the-art web app chatbot that harnesses the power of Streamlit, LangChain, OpenAI, and Python. With its cutting-edge capabilities, IntelliBot transforms the way users interact with information, providing seamless access to knowledge and intelligent responses.

- 🔍 <b>Ask, Seek, Discover: Your Ultimate Query Solution</b>
  - ⚡️ IntelliBot empowers users to ask questions and seek answers effortlessly. 
  - ⚡️ Whether it's searching the web for information or retrieving answers from a comprehensive database, IntelliBot delivers human-like responses, ensuring a seamless user experience.

- 🔗 <b>Harnessing the Power of Pinecone: Cloud Vector Store</b>
  - ⚡️ Backed by **Pinecone**, a robust cloud vector store, IntelliBot leverages the power of advanced indexing and retrieval techniques. 
  - ⚡️ This ensures lightning-fast access to relevant information, enabling swift and accurate responses to user queries.

         
- 💬 <b>Conversations that Matter: Save and Retrieve</b>
  - ⚡️ IntelliBot goes beyond single queries. It allows users to save and store entire conversations, facilitating continuity and enabling a deeper level of understanding. 
  - ⚡️ Seamlessly retrieve past interactions to gain valuable insights and enhance your knowledge.
    
         
- ☁️ <b>Streamlit Cloud: Deployed for Simplicity</b>
  - ⚡️ Deployed on the reliable **Streamlit Cloud** platform, IntelliBot offers a seamless and scalable experience. 
  - ⚡️ Accessible from anywhere, anytime, users can effortlessly harness its capabilities and enjoy a frictionless journey of knowledge discovery.
    
         
- 🚀 <b>Unleash the Potential: Your Trusted Knowledge Companion</b>
  - ⚡️ IntelliBot is your go-to companion for unlocking the vast universe of knowledge. 
  - ⚡️ Whether you're seeking answers, exploring solutions, or engaging in meaningful conversations, IntelliBot is here to make your quest for knowledge a breeze.

Discover IntelliBot today and embark on a transformative journey of information exploration! 🌟

""", unsafe_allow_html=True)
st.write("---")
# st.markdown(description_1, unsafe_allow_html=True)
# st.write("---")
st.markdown("""
<div style="text-align: center;">
    <p>Made with ❤️</p>
</div> """, unsafe_allow_html=True)

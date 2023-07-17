import streamlit as st

st.set_page_config(
    page_title="DataBotX",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 DataBotX: Your Intelligent Conversational Assistant 💬")
st.write("")
st.write("")
st.write("")
description = """
    <div style="text-align: center;">
        <h3>🚀 Discover DataBotX today and embark on a transformative data-driven experience!</h3>
    </div> """
# text-align: center;
description_1 = """
    <style>
        .description {
            font-family: 'Arial', sans-serif;
            line-height: 1.6;
            font-size: 18px;
            color: #ffffff;
        }

        .highlight {
            font-weight: bold;
            color: #0088cc;
        }

        .emoji {
            text-align: center;
            font-size: 36px;
            margin-bottom: 16px;
        }
    </style>

    <div class="description">
        <h3 class="emoji">✨</h3>
        <ul>
        <li><p>⚡️ <span class="highlight">DataBotX</span> is a cutting-edge web app chatbot powered by the synergy of LangChain, OpenAI, Streamlit, Pinecone, and Python. With its advanced capabilities, DataBotX revolutionizes the way you interact with data and harness its potential. </p></li>
        <li><p>⚡️ With <span class="highlight">DataBotX</span>, you can effortlessly upload multiple PDF/TXT/Excel files to our powerful database. This rich repository of information fuels DataBotX's intelligence, allowing it to provide highly relevant answers to your queries.</p></li>
        <li><p>⚡️ <span class="highlight">DataBotX</span> goes beyond mere text-based interactions. It enables you to upload and store entire conversations to the database, facilitating seamless continuity and enhanced insights for future reference.</p></li>
        <li><p>⚡️ <span class="highlight">DataBotX</span> empowers you to create customized indexes within the database, effortlessly organizing and categorizing different sets of data files. This streamlined approach ensures easy access and targeted retrieval of information.</p></li>
        <li><p>⚡️ <span class="highlight">DataBotX</span> is your gateway to unlocking the untapped potential of your data. Whether you're seeking insightful answers or looking to streamline your workflow, DataBotX is your trusted partner on this exciting journey. </p></li>
        </ul>    
    </div>
    """
st.write(description, unsafe_allow_html=True)
st.write("---")
st.markdown(description_1, unsafe_allow_html=True)
st.write("---")
st.markdown("""
<div style="text-align: center;">
    <p>Made with ❤️</p>
</div> """, unsafe_allow_html=True)

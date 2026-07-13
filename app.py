try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import streamlit as st
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Define paths
CHROMA_PATH = "chroma_db_v2"

st.set_page_config(page_title="Real Estate Assistant", page_icon="🏠")

st.title("🏠 Real Estate AI Assistant")
st.markdown("Ask me anything about the real estate projects, builders, or terms!")

@st.cache_resource(show_spinner="Loading models...")
def load_chain():
    # Use HuggingFace embeddings (works well in the cloud)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    # Retrieve Groq API Key securely from Streamlit secrets
    try:
        api_key = st.secrets["GROQ_API_KEY"]
    except KeyError:
        st.error("GROQ_API_KEY is missing from Streamlit secrets.")
        st.stop()
        
    # Use Groq for the LLM generation
    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=api_key)

    system_prompt = (
        "You are a helpful Real Estate AI Assistant. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    retriever = db.as_retriever(search_kwargs={"k": 3})
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is the payment plan for Meridian Greens?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Thinking..."):
        try:
            rag_chain = load_chain()
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
        except Exception as e:
            answer = f"Error: {str(e)}"
            
    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

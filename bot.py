import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import MergerRetriever
from utils import get_collections, get_vector_stores

st.title("Chat by Fivetran")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_sources' not in st.session_state:
    st.session_state.selected_sources = []
if 'source_options' not in st.session_state:
    st.session_state.source_options = []
if 'vector_stores' not in st.session_state:
    st.session_state.vector_stores = {}
if 'chain' not in st.session_state:
    st.session_state.chain = None

with st.sidebar:
    st.subheader("About Me")
    st.write(
        "This is a chat interface app that allows you to ask questions and get responses using RAG with Milvus/Zilliz and OpenAI.")

    st.subheader("Configuration")
    milvus_host = st.text_input("Zilliz Host")
    milvus_token = st.text_input("Zilliz Token", type="password")
    openai_api_key = st.text_input("OpenAI API Key", type="password")

    if milvus_host and milvus_token:
        st.session_state.source_options = get_collections(milvus_host, milvus_token)

        st.subheader("Sources")
        for source in st.session_state.source_options:
            if st.checkbox(source, key=f"checkbox_{source}"):
                if source not in st.session_state.selected_sources:
                    st.session_state.selected_sources.append(source)
            else:
                if source in st.session_state.selected_sources:
                    st.session_state.selected_sources.remove(source)

        if st.session_state.selected_sources and openai_api_key:
            st.session_state.vector_stores = get_vector_stores(milvus_host, milvus_token,
                                                               st.session_state.selected_sources, openai_api_key)

            llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

            retrievers = [vs.as_retriever(search_kwargs={"k": 2}) for vs in st.session_state.vector_stores.values()]
            combined_retriever = MergerRetriever(retrievers=retrievers)

            st.session_state.chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=combined_retriever,
                memory=memory
            )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?", disabled=not st.session_state.chain):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chain({"question": prompt})
            st.markdown(response['answer'])
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

if not st.session_state.chain:
    st.warning("Please enter all required information and select at least one source to start the conversation.")
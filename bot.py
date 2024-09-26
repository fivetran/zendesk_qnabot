import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.retrievers import MergerRetriever
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from utils import get_collections, get_vector_stores
from PIL import Image
import re

DEFAULT_SOURCE_URL = 'https://www.fivetran.com'
DEFAULT_LOGO_LINK = 'https://cdn.prod.website-files.com/619c916dd7a3fa284adc0b27/645d855dca64c3fb02d0af96_645036b7282181d60f8eeea8_6400c474201a85cd6a5f6bb9_fivetran-logo.jpeg'
def infer_source(url, id):

    zendesk_pattern = r'https://[\w-]+\.zendesk\.com/agent/tickets/(\d+)'
    github_pattern = r'https://github\.com/[\w-]+/[\w-]+/issues/(\d+)'

    zendesk_match = re.match(zendesk_pattern, url)
    if zendesk_match:
        return 'https://static-00.iconduck.com/assets.00/zendesk-icon-2048x2048-q18vy4hu.png', zendesk_match.group(1)

    github_match = re.match(github_pattern, url)
    if github_match:
        return 'https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png', github_match.group(1)

    return DEFAULT_LOGO_LINK, id


col1, col2, col3 = st.columns((1, 4, 1))
with col2:
    st.image(Image.open("chatbot_image.png"))

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
    col1, col2, col3 = st.columns((1, 7, 1))
    with col2:
        st.title("Chat with your Data!")
        st.image(Image.open("fivetran_zilliz.png"))
        st.subheader("Powered by Zilliz & Fivetran")

    st.subheader("About Me")
    st.markdown(
        "This RAG-based chat app, powered by Fivetran, Milvus/Zilliz, and OpenAI, lets you instantly access and chat with your company's data. Simply set up a Fivetran-to-Zilliz data pipeline as outlined [here](www.fivetran.com) and enter your Zilliz account credentials below.")

    st.divider()

    st.subheader("Configuration")
    milvus_host = st.text_input("Zilliz Host")
    milvus_token = st.text_input("Zilliz Token", type="password")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    embedding_model = st.selectbox(
        "Embedding Model",
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
    )

    st.divider()

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
                                                               st.session_state.selected_sources, openai_api_key, embedding_model)

            llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
            memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)

            retrievers = [vs.as_retriever(search_kwargs={"k": 5}) for vs in st.session_state.vector_stores.values()]
            combined_retriever = MergerRetriever(retrievers=retrievers)

            prompt = ChatPromptTemplate.from_template("""Answer the following question based on the context provided:

            Context: {context}
            Question: {input}

            Answer:""")

            document_chain = create_stuff_documents_chain(llm, prompt)
            st.session_state.chain = create_retrieval_chain(combined_retriever, document_chain)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What would you like to know?", disabled=not st.session_state.chain):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chain.invoke({"input": prompt})
            st.markdown(response['answer'])
            st.markdown("**Sources:**")
            source_container = st.container()
            cols = source_container.columns(5)

            for idx, doc in enumerate(response.get('context', [])[:5]): # Limit to 10 sources
                doc_url = doc.metadata.get('url', DEFAULT_SOURCE_URL)
                doc_id = str(doc.metadata['id'])
                logo_url, label = infer_source(doc_url, doc_id)

                with cols[idx]:
                    st.markdown(
                        f'<a href="{doc_url}" target="_blank" style="text-decoration: none;">'
                        f'<button style="border-radius: 10px; padding: 5px 10px; margin: 2px; '
                        f'font-size: 12px; border: 1px solid #ADD8E6; background-color: transparent; '
                        f'color: #FFFFFF; cursor: pointer; display: flex; align-items: center; '
                        f'justify-content: center; width: 100%;">'
                        f'<img src="{logo_url}" style="width: 16px; height: 16px; margin-right: 5px;">'
                        f'{label}'
                        f'</button></a>',
                        unsafe_allow_html=True
                    )

            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

if not st.session_state.chain:
    st.warning("Please enter all required information and select at least one source to start the conversation.")
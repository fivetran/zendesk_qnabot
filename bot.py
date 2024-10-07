import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MergerRetriever
from PIL import Image

from snowflake.snowpark import Session

import re

from snowflake_cortex_search import SearchSnowflakeCortex
from snowflake_cortex_chat import ChatSnowflakeCortex

DEFAULT_SOURCE_URL = 'https://www.fivetran.com'
DEFAULT_LOGO_LINK = 'https://cdn.prod.website-files.com/619c916dd7a3fa284adc0b27/645d855dca64c3fb02d0af96_645036b7282181d60f8eeea8_6400c474201a85cd6a5f6bb9_fivetran-logo.jpeg'


def infer_source(url, id):
    zendesk_pattern = r'https://[\w-]+\.zendesk\.com/tickets/(\d+)'
    github_pattern = r'https://github\.com/[\w-]+/[\w-]+/issues/(\d+)'
    slab_pattern = r'https://[\w-]+\.slab\.com/posts/(\w+)'

    zendesk_match = re.match(zendesk_pattern, url)
    if zendesk_match:
        return 'https://static-00.iconduck.com/assets.00/zendesk-icon-2048x2048-q18vy4hu.png', zendesk_match.group(1)

    github_match = re.match(github_pattern, url)
    if github_match:
        return 'https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png', github_match.group(1)

    slab_match = re.match(slab_pattern, url)
    if slab_match:
        return 'https://store-images.s-microsoft.com/image/apps.4075.d693ef1e-dbc3-46a3-a42e-74e54a0e6289.dc69f976-b676-48f4-99df-e1781f0e058c.2e1707e2-197a-4b6d-a9c0-6de4008a9d25.png', slab_match.group(1)

    return DEFAULT_LOGO_LINK, id


col1, col2, col3 = st.columns((1, 4, 1))
with col2:
    st.image(Image.open("chatbot_image.png"))

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_sources' not in st.session_state:
    st.session_state.selected_sources = []
if 'chain' not in st.session_state:
    st.session_state.chain = None

with st.sidebar:
    col1, col2, col3 = st.columns((1, 7, 1))
    with col2:
        st.title("Chat with your Data!")
        st.image(Image.open("fivetran_snowflake.png"))
        st.subheader("Powered by Snowflake & Fivetran")

    st.divider()

    st.subheader("About Me")

    st.markdown(
        "This RAG-based chat app, powered by Fivetran and Snowflake Cortex allows you to instantly access and interact with your company's data. Simply set up a Fivetran-to-Snowflake data pipeline and enter your Snowflake credentials below.")

    st.divider()

    st.subheader("Configuration")
    snowflake_host = st.text_input("Snowflake Host", placeholder="your-account.snowflakecomputing.com")
    snowflake_user = st.text_input("Snowflake Username")
    snowflake_password = st.text_input("Snowflake Password", type="password")
    snowflake_database = st.text_input("Snowflake Database")
    snowflake_schema = st.text_input("Snowflake Schema")
    snowflake_role = st.text_input("Snowflake Role")
    snowflake_warehouse = st.text_input("Snowflake Warehouse")

    st.divider()

    if snowflake_host and snowflake_user and snowflake_password and snowflake_database and snowflake_schema and snowflake_role and snowflake_warehouse:

        conff = {
            "account": snowflake_host.removesuffix(".snowflakecomputing.com"),
            "user": snowflake_user,
            "password": snowflake_password,
            "role": snowflake_role,
            "warehouse": snowflake_warehouse,
            "schema": snowflake_schema,
            "database": snowflake_database,
        }

        search_services = SearchSnowflakeCortex.all_search_services(conff)

        for source in search_services:
            if st.checkbox(source, key=f"checkbox_{source}"):
                if source not in st.session_state.selected_sources:
                    st.session_state.selected_sources.append(source)
            else:
                if source in st.session_state.selected_sources:
                    st.session_state.selected_sources.remove(source)

        if st.session_state.selected_sources:
            service_retrievers = [SearchSnowflakeCortex(
                session_builder_conf=conff,
                snowflake_database=snowflake_database,
                snowflake_schema=snowflake_schema,
                snowflake_cortex_search_service=search_service,
            ).as_retriever(search_kwargs={"k": 5}) for search_service in st.session_state.selected_sources]

            combined_retriever = MergerRetriever(retrievers=service_retrievers)

            llm = ChatSnowflakeCortex(
                session_builder_conf=conff,
            )
            memory = ConversationBufferMemory(memory_key="chat_history", output_key='answer', return_messages=True)

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
            cols = st.container().columns(5)

            for idx, doc in enumerate(response.get('context', [])[:5]):  # Limit to 10 sources
                doc_url = doc.metadata.get('URL', DEFAULT_SOURCE_URL)
                doc_id = str(doc.metadata.get('DOCUMENT_ID', 'UNKNOWN ID'))
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

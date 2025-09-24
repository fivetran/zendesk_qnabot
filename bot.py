import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.retrievers import MergerRetriever
from PIL import Image

from snowflake.snowpark import Session
from snowflake.core import Root

import re
import json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.vectorstores.base import VectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

DEFAULT_SOURCE_URL = 'https://www.fivetran.com'
DEFAULT_LOGO_LINK = 'https://cdn.prod.website-files.com/619c916dd7a3fa284adc0b27/645d855dca64c3fb02d0af96_645036b7282181d60f8eeea8_6400c474201a85cd6a5f6bb9_fivetran-logo.jpeg'


SUPPORTED_ROLES: List[str] = [
    "system",
    "user",
    "assistant",
]


class ChatSnowflakeCortexError(Exception):
    """Error with Snowpark client."""


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any] = {
        "content": message.content.replace("'", '"'),
    }

    if isinstance(message, ChatMessage) and message.role in SUPPORTED_ROLES:
        message_dict["role"] = message.role
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _truncate_at_stop_tokens(
        text: str,
        stop: Optional[List[str]],
) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text

    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text


class ChatSnowflakeCortex(BaseChatModel):
    session_builder_conf: dict = {}
    model: str = "llama3.1-8b"
    cortex_function: str = "complete"
    temperature: float = 0.9

    @property
    def _llm_type(self) -> str:
        return f"snowflake-cortex-{self.model}"

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        message_str = str(message_dicts)
        options = {"temperature": self.temperature}
        options_str = str(options)
        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}'
                ,{message_str},{options_str}) as llm_response;"""

        session = Session.builder.configs(self.session_builder_conf).create()
        try:
            l_rows = session.sql(sql_stmt).collect()
        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex via Snowpark: {e}"
            )
        finally:
            session.close()

        response = json.loads(l_rows[0]["LLM_RESPONSE"])
        ai_message_content = response["choices"][0]["messages"]

        content = _truncate_at_stop_tokens(ai_message_content, stop)
        message = AIMessage(
            content=content,
            response_metadata=response["usage"],
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])


class SearchSnowflakeCortex(VectorStore):

    def __init__(
            self,
            session_builder_conf,
            snowflake_database,
            snowflake_schema,
            snowflake_cortex_search_service
    ):
        self.session_builder_conf = session_builder_conf
        self.snowflake_database: str = snowflake_database
        self.snowflake_schema: str = snowflake_schema
        self.snowflake_cortex_search_service: str = snowflake_cortex_search_service

    def similarity_search(
            self, query: str, k: int = 5, **kwargs: Any
    ) -> List[Document]:
        session = Session.builder.configs(self.session_builder_conf).create()

        root = Root(session)
        search_service = root.databases[self.snowflake_database].schemas[self.snowflake_schema].cortex_search_services[
            self.snowflake_cortex_search_service]

        desc_result = session.sql(f"DESC CORTEX SEARCH SERVICE {self.snowflake_cortex_search_service}").collect()[0]

        search_column = desc_result.search_column
        columns = desc_result.columns.split(",")

        search_resp = search_service.search(
            query=query,
            columns=columns,
            limit=k
        )

        relevant_docs = []
        for row in search_resp.results:
            metadata = {
                col: value
                for col, value in row.items()
                if col != search_column
            }
            doc = Document(page_content=row[search_column], metadata=metadata)
            relevant_docs.append(doc)

        session.close()

        return relevant_docs

    @classmethod
    def from_texts(
            cls,
            texts: List[str],
            embedding: Embeddings,
            metadatas: Optional[List[dict]] = None,
            **kwargs: Any,
    ):
        raise NotImplementedError(f"`from_texts` has not been implemented")

    @staticmethod
    def all_search_services(session_builder_conf):
        session = Session.builder.configs(session_builder_conf).create()
        return [x.name for x in session.sql(f"SHOW CORTEX SEARCH SERVICES").collect()]


def infer_source(url, id):
    zendesk_pattern = r'https://[\w-]+\.zendesk\.com/tickets/(\d+)'
    github_pattern = r'https://github\.com/[\w-]+/[\w-]+/issues/(\d+)'
    slab_pattern = r'https://[\w-]+\.slab\.com/posts/(\w+)'
    sharepoint_pattern = r'https://[\w-]+\.sharepoint\.com/sites/(\w+)'

    zendesk_match = re.match(zendesk_pattern, url)
    if zendesk_match:
        return 'https://static-00.iconduck.com/assets.00/zendesk-icon-2048x2048-q18vy4hu.png', zendesk_match.group(1)

    github_match = re.match(github_pattern, url)
    if github_match:
        return 'https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png', github_match.group(1)

    slab_match = re.match(slab_pattern, url)
    if slab_match:
        return 'https://store-images.s-microsoft.com/image/apps.4075.d693ef1e-dbc3-46a3-a42e-74e54a0e6289.dc69f976-b676-48f4-99df-e1781f0e058c.2e1707e2-197a-4b6d-a9c0-6de4008a9d25.png', slab_match.group(1)

    sharepoint_match = re.match(sharepoint_pattern, url)
    if sharepoint_match:
        return 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Microsoft_Office_SharePoint_%282019%E2%80%93present%29.svg/1024px-Microsoft_Office_SharePoint_%282019%E2%80%93present%29.svg.png', re.search(r'/([^/]+)$', url).group(1) 

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

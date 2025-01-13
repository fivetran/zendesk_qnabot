import streamlit as st
from PIL import Image
from fivetran_ai import FivetranAI
import re


# ------------------------ Utility Functions ------------------------
def infer_icon(url) -> str:
    """
    Returns the icon URL based on the matching pattern of the given URL.
    """
    source_url_patterns = {
        'zendesk': r'https://[\w-]+\.zendesk\.com/agent/tickets/(\d+)',
        'github': r'https://github\.com/[\w-]+/[\w-]+/issues/(\d+)',
        'sharepoint': r'https://[\w-]+\.sharepoint\.com/sites/(\w+)',
        'slab': r'https://[\w-]+\.slab\.com/posts/(\w+)',
        'slack': r'https://[\w-]+\.slack\.com/archives/(\w+)',
        'height': r'https://[\w-]+\.height\.app/(\w+)',
        'fivetran': r'https://fivetran\.com/(\w+)',
    }

    source_to_iconurl = {
        'zendesk': 'https://static-00.iconduck.com/assets.00/zendesk-icon-2048x2048-q18vy4hu.png',
        'github': 'https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png',
        'sharepoint': 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e1/Microsoft_Office_SharePoint_%282019%E2%80%93present%29.svg/1024px-Microsoft_Office_SharePoint_%282019%E2%80%93present%29.svg.png',
        'slab': 'https://store-images.s-microsoft.com/image/apps.4075.d693ef1e-dbc3-46a3-a42e-74e54a0e6289.dc69f976-b676-48f4-99df-e1781f0e058c.2e1707e2-197a-4b6d-a9c0-6de4008a9d25.png',
        'slack': 'https://upload.wikimedia.org/wikipedia/commons/thumb/d/d5/Slack_icon_2019.svg/2048px-Slack_icon_2019.svg.png',
        'height': 'https://downloads.intercomcdn.com/i/o/574797/a3f29a6bbaf0f446eee9d93a/179ef7b4b3717b2f0602dd90a827b6ee.png',
        'fivetran': 'https://fivetran.com/static-assets-docs/_next/static/media/fivetran-logo.cd1505dc.svg'
    }

    for source, pattern in source_url_patterns.items():
        if re.match(pattern, url):
            return source_to_iconurl[source]

    # Return a default icon if no pattern matches
    return 'https://i.pinimg.com/originals/a1/85/fb/a185fb1e19ce1225d619ba36a0f85b29.png'


# ------------------------ Page Layout / Styling ------------------------
# A small block of CSS to tweak the look and feel
st.markdown(
    """
    <style>
    /* Style the main Chat messages */
    .stChatMessage {
        border: 1px solid #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f7f9fc;
    }
    /* Style the user prompt input */
    .stTextArea [data-baseweb="input"] {
        background-color: #ffffff;
    }
    /* Style the side-bar to have a slightly different background */
    section[data-testid="stSidebar"] {
        background-color: #FAFAFA !important;
    }
    /* Custom button styling for source links */
    .source-button {
        border-radius: 10px;
        padding: 5px 10px;
        margin: 5px 0;
        font-size: 12px;
        border: 1px solid #ADD8E6;
        background-color: #ffffff;
        color: #333333;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: flex-start;
        width: 100%;
    }
    .source-icon {
        width: 16px;
        height: 16px;
        margin-right: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------------ Header Section ------------------------
col1, col2, col3 = st.columns((1, 4, 1))
with col2:
    st.image(Image.open("chatbot_image.png"))

# ------------------------ Session State Init ------------------------
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'token' not in st.session_state:
    st.session_state.token = None

# ------------------------ Sidebar ------------------------
with st.sidebar:
    col1_sb, col2_sb, col3_sb = st.columns((1, 7, 1))
    with col2_sb:
        st.title("Chat with your Data!")
        st.image(Image.open("fivetran_snowflake.png"))
        st.subheader("Powered by FivetranAI")

    st.divider()

    st.subheader("About Me")
    st.markdown(
        "This chat app powered by **FivetranAI** allows you to instantly access and interact with your company's data."
        " Simply set up a FivetranAI account and enter your API key below."
    )
    st.divider()

    st.subheader("Configuration")
    token = st.text_input("FivetranAI API Key", placeholder="Your FivetranAI API Key")
    if token:
        st.session_state.token = token

# ------------------------ Display Past Messages ------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If it's an assistant message that stored sources, display them
        if message["role"] == "assistant" and "sources" in message:
            if message["sources"]:
                with st.expander("View Sources"):
                    for doc in message["sources"]:
                        metadata = doc.get('document', {}).get('metadata', {})
                        doc_url = metadata.get('url', "#")
                        logo_url = infer_icon(doc_url)
                        title = metadata.get("title", "Untitled")

                        st.markdown(
                            f"""
                            <a href="{doc_url}" target="_blank" style="text-decoration: none;">
                                <button class="source-button">
                                    <img src="{logo_url}" class="source-icon"/>
                                    {title}
                                </button>
                            </a>
                            """,
                            unsafe_allow_html=True
                        )

# ------------------------ Prompt Input & Handling ------------------------
prompt = st.chat_input("What would you like to know?", disabled=not st.session_state.token)

if prompt:
    # 1) Capture user message
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)

    with st.chat_message("user"):
        st.markdown(prompt)

    # 2) Get assistant response from your LLM or external API
    with st.spinner("Thinking..."):
        response = FivetranAI(st.session_state.token).chat(prompt)

    answer_text = response.get('answer', '')
    sources = response.get('sources', [])

    # 3) Store the new assistant message (with sources) in session
    assistant_message = {
        "role": "assistant",
        "content": answer_text,
        "sources": sources
    }
    st.session_state.messages.append(assistant_message)

    # 4) Display the newly-added assistant message & sources immediately
    with st.chat_message("assistant"):
        st.markdown(answer_text)
        if sources:
            with st.expander("View Sources"):
                for doc in sources:
                    metadata = doc.get('document', {}).get('metadata', {})
                    doc_url = metadata.get('url', "#")
                    logo_url = infer_icon(doc_url)
                    title = metadata.get("title", "Untitled")

                    st.markdown(
                        f"""
                        <a href="{doc_url}" target="_blank" style="text-decoration: none;">
                            <button class="source-button">
                                <img src="{logo_url}" class="source-icon"/>
                                {title}
                            </button>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )

# ------------------------ If No Token ------------------------
if not st.session_state.token:
    st.warning("Please enter your FivetranAI API Key to start chatting")

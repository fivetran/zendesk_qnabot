import streamlit as st
from streamlit_chat import message
from utils import get_tables

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_tools' not in st.session_state:
    st.session_state.selected_tools = []
if 'tool_options' not in st.session_state:
    st.session_state.source_options = {}

st.markdown("""
<style>
    .tool-selector {
        display: flex;
        align-items: center;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .tool-selector:hover {
        background-color: #f0f0f0;
    }
    .tool-selector img {
        width: 24px;
        height: 24px;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Chat by Fivetran")

with st.sidebar:
    st.header("Sidebar")

    st.subheader("About Me")
    st.write(
        "This is a chat interface app that allows you to ask questions and get responses. It integrates with Milvus/Zilliz for data processing.")

    st.subheader("Configuration")
    milvus_host = st.text_input("Zilliz Host")
    milvus_token = st.text_input("Zilliz Token")

    if milvus_host and milvus_token:
        collections = get_tables(milvus_host, milvus_token)
        st.session_state.source_options = {
            collection: f"https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png"
            for collection in collections
        }
        st.subheader("Sources")


    for tool, logo_url in st.session_state.source_options.items():
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"""
                <div class="tool-selector">
                    <img src="{logo_url}" alt="{tool} logo">
                    {tool}
                </div>
            """, unsafe_allow_html=True)
        with col2:
            is_selected = st.checkbox("", key=f"checkbox_{tool}", value=tool in st.session_state.selected_tools)
            if is_selected and tool not in st.session_state.selected_tools:
                st.session_state.selected_tools.append(tool)
            elif not is_selected and tool in st.session_state.selected_tools:
                st.session_state.selected_tools.remove(tool)

st.subheader("Chat")
user_input = st.text_input("Ask a question:")

def get_response(user_input):
    # TODO - Here you would implement the logic to process the user input
    #   and generate a response, potentially using Milvus/Zilliz
    return f"You said: {user_input}. Selected tools: {', '.join(st.session_state.selected_tools)}"

if st.button("Send"):
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        response = get_response(user_input)
        st.session_state.messages.append({"role": "assistant", "content": response})

for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=f"{i}_{msg['role']}")
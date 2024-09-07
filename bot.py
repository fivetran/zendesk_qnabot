import streamlit as st
from streamlit_chat import message
import pymilvus

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'selected_tools' not in st.session_state:
    st.session_state.selected_tools = []

# Custom CSS for buttons with logos
st.markdown("""
<style>
    .tool-button {
        display: flex;
        align-items: center;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        cursor: pointer;
    }
    .tool-button:hover {
        background-color: #f0f0f0;
    }
    .tool-button.selected {
        background-color: #28a745;
        color: white;
    }
    .tool-button img {
        width: 24px;
        height: 24px;
        margin-right: 10px;
    }
</style>
""", unsafe_allow_html=True)


# Main function to run the Streamlit app
def main():
    st.title("Chat Interface")

    # Sidebar
    with st.sidebar:
        st.header("Sidebar")

        # About Me section
        st.subheader("About Me")
        st.write(
            "This is a chat interface app that allows you to ask questions and get responses. It integrates with Milvus/Zilliz for data processing.")

        # Configuration section
        st.subheader("Configuration")
        milvus_host = st.text_input("Milvus/Zilliz Host")
        milvus_port = st.number_input("Milvus/Zilliz Port", min_value=1, max_value=65535, value=19530)

        # Tool selection section
        st.subheader("Tools")
        tool_options = {
            "GitHub": "https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png",
            "HubSpot": "https://www.hubspot.com/hubfs/assets/hubspot.com/style-guide/brand-guidelines/guidelines_the-logo.svg",
            "Jira": "https://wac-cdn.atlassian.com/assets/img/favicons/atlassian/favicon.png"
        }

        for tool, logo_url in tool_options.items():
            col1, col2 = st.columns([5, 1])
            with col1:
                button_class = "tool-button selected" if tool in st.session_state.selected_tools else "tool-button"
                st.markdown(f"""
                    <div class="{button_class}" id="{tool}-button">
                        <img src="{logo_url}" alt="{tool} logo">
                        {tool}
                    </div>
                """, unsafe_allow_html=True)
            with col2:
                if st.button("Toggle", key=f"toggle_{tool}", help=f"Toggle {tool}"):
                    if tool in st.session_state.selected_tools:
                        st.session_state.selected_tools.remove(tool)
                    else:
                        st.session_state.selected_tools.append(tool)
                    st.experimental_rerun()

    # Chat interface
    st.subheader("Chat")
    user_input = st.text_input("Ask a question:")

    if st.button("Send"):
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            response = get_response(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Display chat messages
    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=msg["role"] == "user", key=f"{i}_{msg['role']}")


# Function to get response (placeholder for now)
def get_response(user_input):
    # Here you would implement the logic to process the user input
    # and generate a response, potentially using Milvus/Zilliz
    return f"You said: {user_input}. Selected tools: {', '.join(st.session_state.selected_tools)}"


if __name__ == "__main__":
    main()
import streamlit as st
from chat import Chat
import base64
from read_email import FirstProcessing, SecondProcessing

st.set_page_config(page_title="Smart Emails", layout="wide")

def apply_custom_css():
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 3rem;
    }
    header {
        padding-top: -2rem;
        padding-bottom: 0rem;
    }
    .stTabs {
        margin-top: -10px;
    }
    </style>
""", unsafe_allow_html=True)

def change_chatbot_style():
    chat_input_style = f"""
    <style>
        .stChatInput {{
          position: fixed;
          bottom: 3rem;
        }}
    </style>
    """
    st.markdown(chat_input_style, unsafe_allow_html=True)

def display_base64_image(base64_strings, current_index):
    if base64_strings:
        base64_string = base64_strings[current_index]
        if base64_string:
            img_data = base64.b64decode(base64_string)
            st.image(img_data, caption=f"Image {current_index + 1} of {len(base64_strings)}", use_column_width=True)

def main():
    apply_custom_css() 
    change_chatbot_style() 
    
    st.title("Smart Emails 💡✉️")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi, I'm Celia and I'm here to help you get information from your email database."}
        ]
    
    if "current_index" not in st.session_state:
        st.session_state.current_index = 0
    if "images" not in st.session_state:
        st.session_state.images = []

    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        if prompt := st.chat_input("Type your message:"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    chat = Chat(prompt)
                    response = chat.openai_request()
                    st.session_state.images = chat.get_img()
                    st.write(response)
                    
            st.session_state.messages.append({"role": "assistant", "content": response})
    
    with col2:
        st.title("Related Files:")
        if st.session_state.images:
            display_base64_image(st.session_state.images, st.session_state.current_index)

            col1, col2, col3 = st.columns([0.3, 0.4, 0.3])
            with col1:
                if st.button("Previous") and st.session_state.current_index > 0:
                    st.session_state.current_index -= 1
            with col3:
                if st.button("Next") and st.session_state.current_index < len(st.session_state.images) - 1:
                    st.session_state.current_index += 1
        else:
            st.write("No image available.")

        st.write('---')
        
        if st.button("Run Pipeline"):
            with st.spinner("Processing..."):
                first_processing = FirstProcessing(database_path='database/database.json', emails_path='emails')
                first_processing.process_emails()
                second_processing = SecondProcessing(database_path='database/database.json')
                second_processing.update_row()
            st.success("Pipeline has been executed successfully!")

if __name__ == "__main__":
    main()
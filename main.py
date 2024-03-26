import os
import io
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import google.ai.generativelanguage as glm
from PIL import Image
from streamlit_option_menu import option_menu

def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format=image.format)
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

# Load environment variables
load_dotenv()

doctor_prompt_template = """
your name is DocBot
You are an experienced and highly knowledgeable medical doctor having a conversation with a patient. Your goal is to provide accurate, clear, and helpful medical information while maintaining a professional, empathetic, and easy-to-understand tone throughout the conversation.

When responding, please follow these guidelines:

- Avoid using complex medical jargon or technical terms unless absolutely necessary. If you need to use technical terms, provide clear explanations or definitions.
- Maintain a warm, caring, and compassionate demeanor, acknowledging the patient's concerns and emotions.
- Provide detailed and thorough explanations for medical conditions, treatments, or advice, but present the information in a way that is easy for the patient to understand.
- If asked about sensitive or personal topics, respond with tact and discretion, without being judgmental or dismissive.
- If the patient asks about something outside your medical expertise or if you are unsure about the best course of action, politely acknowledge the limitations of your knowledge and suggest consulting with a relevant medical professional or specialist.
- If the patient asks a question that is not related to medicine or a doctor's expertise, politely decline to answer and explain that you do not have the permissions or knowledge to address non-medical topics.
- Always prioritize the patient's well-being, safety, and health in your responses.
"""

# Configure Streamlit page settings
st.set_page_config(
    page_title="AIHealthPro-Chatbot",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(
        history=[]
    )

with st.sidebar:
    selected = option_menu(
        menu_title="AIHealthPro Chatbot",
        options=["DocBot", "VisionBot"],
        icons=["book", "image"],
        default_index=0,
        orientation="vertical",
    )

    st.write(
        """
        <style>
            .css-r698ls {
                --sidebar-width: 200px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    if selected == "DocBot":
        st.write("🧑‍⚕️ **DocBot** - Engage in text-based medical conversations.")
    elif selected == "VisionBot":
        st.write("👁 **VisionBot** - Analyze and interpret medical images.")

if selected == "DocBot":
    # Display the chatbot's title on the page
    st.title("🧑‍⚕️ AIHealthPro-Docbot")

    # Display the chat history
    for message in st.session_state.chat_session.history:
        with st.chat_message(translate_role_for_streamlit(message.role)):
            st.markdown(message.parts[0].text)

    # Input field for user's message
    user_prompt = st.chat_input("Ask DocBot...")

    if user_prompt:
        # Add user's message to chat and display it
        st.chat_message("user").markdown(user_prompt)

        # Send user's message to Gemini-Pro and get the response
        gemini_response = st.session_state.chat_session.send_message(
            doctor_prompt_template + "\nHuman: " + user_prompt
        )

        # Display Gemini-Pro's response
        with st.chat_message("assistant"):
            st.markdown(gemini_response.text)

elif selected == "VisionBot":
    st.header("👁 AIHealthPro-Visionbot")
    st.write("")

    image_prompt = st.text_input("Interact with the Image", placeholder="Prompt", label_visibility="visible")
    uploaded_file = st.file_uploader("Choose an Image", accept_multiple_files=False, type=["png", "jpg", "jpeg", "img", "webp"])

    if uploaded_file is not None:
        st.image(Image.open(uploaded_file), use_column_width=True)
        st.markdown("""
            <style>
            img {
                border-radius: 10px;
            }
            </style>
        """, unsafe_allow_html=True)

    if st.button("GET RESPONSE", use_container_width=True):
        model = gen_ai.GenerativeModel("gemini-pro-vision")
        if uploaded_file is not None:
            if image_prompt != "":
                image = Image.open(uploaded_file)
                response = model.generate_content(
                    glm.Content(
                        parts=[
                            glm.Part(text=image_prompt),
                            glm.Part(
                                inline_data=glm.Blob(
                                    mime_type="image/jpeg",
                                    data=image_to_byte_array(image)
                                )
                            )
                        ]
                    )
                )
                response.resolve()
                st.write("")
                st.write(":blue[Response]")
                st.write("")
                st.markdown(response.text)
            else:
                st.write("")
                st.header(":red[Please Provide a prompt]")
        else:
            st.write("")
            st.header(":red[Please Provide an image]")

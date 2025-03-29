import simpleaudio as sa
from google.cloud import texttospeech
import itertools
from google import genai
from google.genai import types
import vertexai
import streamlit as st, os
from audio_recorder_streamlit import audio_recorder
from google.cloud import speech
from pypdf import PdfReader, PdfWriter
from time import time
from dotenv import load_dotenv
import mimetypes
import pymupdf
from deep_translator import GoogleTranslator
import fitz

load_dotenv()


def page_setup():
    st.header("Use a voice bot to interact with your PDF file!", anchor=False, divider="blue")

    st.sidebar.header("Options", divider='rainbow')
    
    hide_menu_style = """
            <style>
            #MainMenu {visibility: hidden;}
            </style>
            """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.sidebar.markdown("Streaming Version")


def get_choice():
    choice = st.sidebar.radio("Choose:", ["Audio 2 Text chat with many PDFs",
                                          "Text 2 Text chat with many PDFs",
                                          "Chat with an image",
                                          "Translate from English",
                                          "Chat with audio"],)
    st.sidebar.divider()
    return choice

def get_clear():
    clear_button=st.sidebar.button("Start new session", key="clear")
    return clear_button

def run_streaming_tts(textsample):
    client = texttospeech.TextToSpeechClient()

    streaming_config = texttospeech.StreamingSynthesizeConfig(voice=texttospeech.VoiceSelectionParams(name="en-US-Journey-D", language_code="en-US"))

    config_request = texttospeech.StreamingSynthesizeRequest(streaming_config=streaming_config)

    def request_generator(textsample):
        yield texttospeech.StreamingSynthesizeRequest(input=texttospeech.StreamingSynthesisInput(text=textsample))
        

    streaming_responses = client.streaming_synthesize(itertools.chain([config_request], request_generator(textsample)))
    for response in streaming_responses:
        fs = 24000
        play_obj = sa.play_buffer(response.audio_content, 1, 2, fs)
        play_obj.wait_done()


def main():
    
    choice = get_choice()
    
    if choice == "Translate from English":
        language_choice = st.sidebar.selectbox(
        "Choose language:",
        ["es", "fr"]
        )
        st.write(f"Language selected for translation: {language_choice}")
        
        st.subheader("Chat with your PDF file")
        clear = get_clear()
        if clear:
            if 'message' in st.session_state:
                del st.session_state['message']
    
        if 'message' not in st.session_state:
            st.session_state.message = " "
        
        if clear not in st.session_state:
        
            uploaded_files1 = st.file_uploader("Choose 1 or more files",  type=['pdf'], accept_multiple_files=False)
            if uploaded_files1 is not None:
                translate_text = GoogleTranslator(source = 'en', target = language_choice)
            for page in uploaded_files1:
                blocks = page.get_text("blocks", flags = pymupdf.TEXT_DEHYPHENATE)
                
                for block in blocks:
                    bbox = block[4:]
                    
                    original = block[4]
                    
                    translated = translate_text.translate(original)
                    page.draw_rect(bbox, color=None, fill=pymupdf.pdfcolor["white"])
                    
                    page.insert_htmlbox(bbox, translated)
                
            
    
    elif choice == "Audio 2 Text chat with many PDFs":
        audio_bytes = audio_recorder(recording_color="#6aa36f", neutral_color="#e82c58")
        
        if audio_bytes:
            audio = speech.RecognitionAudio(content=audio_bytes)

            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code="en-US",
                model="default",
                audio_channel_count=2,
                enable_word_confidence=True,
                enable_word_time_offsets=True,
            )

            operation = client.long_running_recognize(config=config, audio=audio)
            conversion = operation.result(timeout=90)
            
            # Ensure at least one result exists
            result = None
            for res in conversion.results:
                result = res
                break

            if not result or not result.alternatives:
                st.error("No transcription result found.")
                return  # Exit early to prevent referencing None

            prompt2 = result.alternatives[0].transcript
        # else:
        #     st.error("No audio input detected")
        #     return

        st.subheader("Ask a 🎙️❓ about your PDF file")
        clear = get_clear()
        if clear and 'message' in st.session_state:
            del st.session_state['message']
        
        if 'message' not in st.session_state:
            st.session_state.message = " "

        uploaded_files = st.file_uploader("Choose 1 or more files", accept_multiple_files=True)
        
        if uploaded_files:
            merger = PdfWriter()
            for file in uploaded_files:
                merger.append(file)

            fullfile = "merged_all_files.pdf"
            merger.write(fullfile)
            merger.close()

            file_upload = client.files.upload(file=fullfile)
            chat = client.chats.create(
                model=MODEL_ID,
                history=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_uri(
                                file_uri=file_upload.uri,
                                mime_type=file_upload.mime_type
                            ),
                        ]
                    ),
                ]
            )

            # Stream the response
            for chunk in client.models.generate_content_stream(
                model=MODEL_ID,
                config=types.GenerateContentConfig(
                    system_instruction="You are a helpful assistant. Your answers need to be brief and concise.",
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                    max_output_tokens=100,
                    ),
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_uri(
                                    file_uri=file_upload.uri,
                                    mime_type=file_upload.mime_type),
                                ]),
                        prompt2,]
                ):
                    
                    run_streaming_tts(chunk.text)
    
    elif choice == "Text 2 Text chat with many PDFs":
        st.subheader("Chat with your PDF file")
        clear = get_clear()
        if clear:
            if 'message' in st.session_state:
                del st.session_state['message']
    
        if 'message' not in st.session_state:
            st.session_state.message = " "
        
        if clear not in st.session_state:
        
            uploaded_files2 = st.file_uploader("Choose 1 or more files",  type=['pdf'], accept_multiple_files=True)
            
            if uploaded_files2:
                merger = PdfWriter()
                for file in uploaded_files2:
                        merger.append(file)
    
                fullfile = "merged_all_files.pdf"
                merger.write(fullfile)
                merger.close()

                file_upload2 = client.files.upload(file=fullfile) 
                chat2 = client.chats.create(model=MODEL_ID,
                    history=[
                        types.Content(
                            role="user",
                            parts=[
    
                                    types.Part.from_uri(
                                        file_uri=file_upload2.uri,
                                        mime_type=file_upload2.mime_type),
                                    ]
                            ),
                        ]
                    )
                    
                prompt3 = st.chat_input("Enter your question here")
                if prompt3:
                    with st.chat_message("user"):
                        st.write(prompt3)
                        
            
                    st.session_state.message += prompt3
                    with st.chat_message(
                        "model", avatar="🤖"):
                        response_text = ""
                        
                        for chunk in client.models.generate_content_stream(
                            model= MODEL_ID,
                            config=types.GenerateContentConfig(
                            system_instruction="You are a helpful assistant. Your answers need to be brief and concise.",
                            temperature=1.0,
                            top_p=0.95,
                            top_k=20,
                            max_output_tokens=100,
                            ),
                            contents=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_uri(
                                        file_uri=file_upload.uri,
                                        mime_type=file_upload.mime_type),
                                    ]),
                            prompt2,]
                    ):
                  
                        
                            response_text = chunk.send_text
                            st.markdown(chunk.text)
                            # st.sidebar.markdown(response2b.usage_metadata)
                    
                            run_streaming_tts(chunk.text)
                    st.session_state.message(chunk.text)
        
    elif choice == "Chat with an image":
        st.subheader("Ask a question to the image 📷")
        clear = get_clear()
        if clear:
            if 'message' in st.session_state:
                del st.session_state['message']

        if 'message' not in st.session_state:
            st.session_state.message = " "

        if clear not in st.session_state:
            uploaded_files3 = st.file_uploader("Choose your PNG or JPEG file", type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)
            
            if uploaded_files3:
                # Read file as bytes
                file_bytes = uploaded_files3.read()

                # Guess MIME type
                import mimetypes
                mime_type, _ = mimetypes.guess_type(uploaded_files3.name)
                if mime_type is None:
                    st.error("Could not determine the MIME type. Please upload a valid image file.")
                else:
                    # Upload the file correctly
                    file_upload = client.files.upload(file=file_bytes)  # Removed 'mime_type' argument

                    chat3 = client.chats.create(model=MODEL_ID,
                        history=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part.from_uri(
                                        file_uri=file_upload.uri,
                                        mime_type=mime_type),  # Use the detected MIME type
                                    ]
                            ),
                        ]
                    )

                    prompt4 = st.chat_input("Enter your question here")
                    if prompt4:
                        with st.chat_message("user"):
                            st.write(prompt4)

                        st.session_state.message += prompt4
                        with st.chat_message("model", avatar="🤖"):
                            response3 = chat3.send_message(st.session_state.message)
                            st.markdown(response3.text)
                        st.session_state.message += response3.text

        
    elif choice == "Chat with audio":
        st.subheader("Chat with your audio file 🎵")
        clear = get_clear()
        
        if clear:
            if 'message' in st.session_state:
                del st.session_state['message']
    
        if 'message' not in st.session_state:
            st.session_state.message = " "
        
        if clear not in st.session_state:
            uploaded_files4 = st.file_uploader("Choose your mp3 or wav file",  type=['mp3','wav'], accept_multiple_files=False)
            if uploaded_files4:
                file_name4=uploaded_files4.name
                
                file_extension = file_name4.split('.')[-1].lower()

                if file_extension == 'mp3':
                    mime_type = 'audio/mpeg'
                elif file_extension == 'wav':
                    mime_type = 'audio/wav'
                else:
                    raise ValueError(f"Unsupported file type: {file_extension}")

                # Read the file as a byte stream
                file_bytes = uploaded_files4.read()

                # Upload the file with the byte stream and MIME type
                file_upload4 = client.files.upload(file=file_bytes)

                
                # file_upload4 = client.files.upload(file=file_name4)
                chat4 = client.chats.create(model=MODEL_ID,
                    history=[
                        types.Content(
                            role="user",
                            parts=[
    
                                    types.Part.from_uri(
                                        file_uri=file_upload4.uri,
                                        mime_type=file_upload4.mime_type),
                                    ]
                            ),
                        ]
                        )
                prompt5 = st.chat_input("Enter your question here")
                if prompt5:
                    with st.chat_message("user"):
                        st.write(prompt5)
            
                    st.session_state.message += prompt5
                    with st.chat_message(
                        "model", avatar="🤖",
                    ):
                        response5 = chat4.send_message(st.session_state.message)
                        st.markdown(response5.text)
                    st.session_state.message += response5.text

if __name__ == "__main__":
    page_setup()
    projectid = os.getenv('GOOGLE_PROJECT')
    api_key = os.getenv('GOOGLE_API_KEY_NEW')
    client = genai.Client(api_key= api_key)
    # client2 = texttospeech.TextToSpeechClient()
    MODEL_ID = "gemini-2.0-flash-001"
    vertexai.init(project=projectid, location="us-central1")
    main()
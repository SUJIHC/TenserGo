
import streamlit as st
import pyaudio
import wave
import keyboard
import os
from transformers import pipeline
from gtts import gTTS
import assemblyai as aai
from streamlit_webrtc import webrtc_streamer  # Add webcam functionality

# Initialize AssemblyAI
aai.settings.api_key = "9ab281c17d544a96b96e0c7ea313f0fc"
transcriber = aai.Transcriber()

# Set up Streamlit app
st.title("Speech-to-Speech App with Webcam")

# Webcam stream
webrtc_streamer(key="webcam", video_frame_callback=None, audio_frame_callback=None)

# Record audio and save to a WAV file
def record_audio():
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    frames = []
    
    st.write("Recording... Press 's' on your keyboard to stop.")
    
    while True:
        data = stream.read(CHUNK)
        frames.append(data)
        
        if keyboard.is_pressed('s'):
            break

    stream.stop_stream()
    stream.close()
    p.terminate()
    
    wf = wave.open(OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    st.write(f"Audio saved as {OUTPUT_FILENAME}")
    return OUTPUT_FILENAME

# Transcribe the audio file
def transcribe_audio(filename):
    st.write("Transcribing the audio...")
    transcript = transcriber.transcribe(filename)
    return transcript.text

# Generate text response using GPT-2
def generate_text_response(prompt):
    st.write("Generating AI response...")
    generator = pipeline('text-generation', model='gpt2')
    responses = generator(prompt, max_length=100, num_return_sequences=1)
    return responses[0]['generated_text']

# Convert text to speech
def text_to_speech(text, filename='output.mp3'):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(filename)
    st.write(f"Text-to-Speech saved as {filename}")
    os.system(f"start {filename}")  # Adjust for your OS

# Main app logic
if st.button('Start Recording'):
    output_file = record_audio()
    input_text = transcribe_audio(output_file)
    st.write(f"Transcription: {input_text}")

    if input_text:
        response = generate_text_response(input_text)
        st.write(f"AI Response: {response}")
        text_to_speech(response)

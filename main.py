import os
import speech_recognition as sr
import io
import pyaudio
from google.oauth2.service_account import Credentials
from google.cloud import texttospeech

import openai

def remove_text_after_human(text):
    index = text.find("Human:")
    return text[:index] if index != -1 else text


def ask_question(question, api_key):
    openai.api_key = api_key

    prompt = "The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly and speaks finnish.\n\nHuman: Hei kuka sinä olet?\nAI: Olen OpenAIn luoma tekoäly. Kuinka voin auttaa sinua tänään?\nHuman: " + question + "?\nAI:"
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.25,
        max_tokens=150,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.6,
        stop=[" Human:", " AI:"]
    )

    return remove_text_after_human(response["choices"][0]["text"])

def text_to_speech(text):
    credentials = Credentials.from_service_account_file(
        "google_cloud_credentials.json")
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code='fi-FI',
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with io.BytesIO(response.audio_content) as f:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        # Open a streaming stream
        stream = p.open(format=p.get_format_from_width(2),
                        channels=1,
                        rate=16000,
                        output=True)
        # Read the data into a buffer
        data = f.read()
        # Start the stream
        stream.start_stream()
        # Write the data to the stream
        stream.write(data)
        # Stop the stream
        stream.stop_stream()
        # Close the stream
        stream.close()
        # Terminate the PyAudio object
        p.terminate()

api_key = os.environ.get("API_KEY")
# Use the audio file as the source
r = sr.Recognizer()
with sr.AudioFile("test-mic.wav") as source:
    audio = r.record(source)
# Use the microphone as the source
with sr.Microphone() as source:
    print("Say something in Finnish:")
    try:
        audio = r.listen(source, timeout=5, phrase_time_limit=5)
    except sr.WaitTimeoutError:
        print("timeout")
        exit(0)

# Use the Google Cloud Speech-to-Text API to transcribe the audio
try:
    text = r.recognize_google_cloud(audio, credentials_json="google_cloud_credentials.json", language="fi-FI")
    #text = "mikä elokuva kannattaa katsoa"
    print("Transcription: " + text)
    answer = ask_question(text, api_key)
    print("Answer from OpenAI:" + answer)
    text_to_speech(answer)
except sr.UnknownValueError:
    print("Google Cloud Speech could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Cloud Speech service; {0}".format(e))

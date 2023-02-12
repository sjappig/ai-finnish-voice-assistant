import os
import speech_recognition as sr
import io
import pyaudio
from google.oauth2.service_account import Credentials
from google.cloud import texttospeech

import openai

GOOGLE_CLOUD_CREDENTIALS_JSON_PATH = os.environ.get("GOOGLE_CLOUD_CREDENTIALS_JSON_PATH")
openai.api_key = os.environ.get("OPEN_AI_API_KEY")

def remove_text_after_human(text):
    index = text.find("Human:")
    return text[:index] if index != -1 else text

conversation_history = ["""
The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, very friendly and speaks finnish.

Human: Hei kuka sinä olet?
AI: Olen OpenAIn luoma tekoäly. Kuinka voin auttaa sinua tänään?
"""]

def ask_question(question):
    conversation_history.append("Human: " + question + "?\nAI:")

    prompt = "\n".join(conversation_history)
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=256,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=["Human:", "AI:"]
    )
    if len(conversation_history) > 10:
        conversation_history.pop(1)

    conversation_history.append(remove_text_after_human(response["choices"][0]["text"]))
    return conversation_history[-1]

def text_to_speech(text, sample_rate_hz=16000):
    credentials = Credentials.from_service_account_file(GOOGLE_CLOUD_CREDENTIALS_JSON_PATH)
    client = texttospeech.TextToSpeechClient(credentials=credentials)
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code='fi-FI',
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE)
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz,
    )
    response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
    with io.BytesIO(response.audio_content) as f:
        # Initialize PyAudio
        p = pyaudio.PyAudio()
        # Open a streaming stream
        stream = p.open(format=p.get_format_from_width(2),
                        channels=1,
                        rate=sample_rate_hz,
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

def speech_to_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something in Finnish:")
        try:
            audio = recognizer.listen(source, timeout=15, phrase_time_limit=3)
        except sr.WaitTimeoutError:
            print("timeout")
    try:
        return recognizer.recognize_google_cloud(audio, credentials_json=GOOGLE_CLOUD_CREDENTIALS_JSON_PATH, language="fi-FI")
    except sr.UnknownValueError:
        pass
    except sr.RequestError as e:
        print("Could not request results from Google Cloud Speech service; {0}".format(e))
        exit(1)

    return None

def is_end_command(text):
    return text.lower().strip() == "lopeta"

def main():
    while True:
        text = speech_to_text()
        if not text:
            continue

        print("Transcription: " + text)
        if is_end_command(text):
            exit(0)
        answer = ask_question(text)
        print("Answer from OpenAI:" + answer)
        text_to_speech(answer)

main()

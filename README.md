# AI Finnish Voice Assistant

> Most of the code, the name of this project, and the descriptions below are generated by ChatGPT. However, it is noteworthy that the code generated by ChatGPT was often not directly usable and contained many small mistakes that had to be fixed manually. This project was a learning experience for me to learn how to use AI as a tool to accelerate the development process, which it certainly does.

> Lessons learned:
>  - Similarly, as in software development in general, taking a small increments and iterative approach is the best way forward.
>  - A function is a good-sized abstraction to ask from ChatGPT.
>  - It is all about smart text completion: you just have to "set up the scene" for GPT-3 to provide a conversational AI.
>  - In some cases, you may find that you have to provide a lot of details; in these cases, it might be better to write the code directly yourself.
>  - Generating documentation and instructions works really well.

This is a conversational AI program written in Python. It uses the OpenAI API to generate a response to a user's spoken questions. The user's questions are transcribed from speech to text using the Google Cloud Speech-to-Text API, then passed to the OpenAI API to generate a response. The response is then read out loud using the Google Cloud Text-to-Speech API. The program continues in a loop until the user says "lopeta", which terminates the program. The conversation history is stored in a list and limited to the most recent 10 exchanges.

## Dependencies

    pip install pipenv
    pipenv shell
    pipenv install

For this program to run, the following environment variables need to be set:

1. `GOOGLE_CLOUD_CREDENTIALS_JSON_PATH`: This is the path to the Google Cloud credentials file. The program uses this file to authenticate with the Google Cloud Text-to-Speech API and the Google Cloud Speech-to-Text API.

2. `OPEN_AI_API_KEY`: This is the API key for OpenAI's API. The program uses this API key to generate the AI responses.

Both of these environment variables must be set on the system running the code before the code is executed.

Lastly, the program assumes that the user's microphone is connected and properly configured.

# Personal assistant based on your own proprietary data

r2d2 is your own personal chatbot that you can feed with proprietary data. You can then ask it questions and get answers based on the input data you provided.

## Requirements

Make sure you have an OpenAI account and a Pinecone account. We will need API keys for each account.

## Setup

- Create virtual env and install packages:

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

- Create a `.env` file and add the following env vars:

```
OPENAI_API_KEY=
PINECONE_API_KEY=
```

- Create a `transcription.txt` file in the same root directiory and add your proprietary data
- Run with command `python main.py`. Once the server starts, you can send post requests with the following json payload

```
{
  "prompt": "<Your prompt goes here>"
}
```

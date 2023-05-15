# GLMbot
A qa bot based on Chatglm
# QucikStart
## 1. Install Python Packages
```pip3 install -r requirements.txt```
## 2. Configuration Environment
You can copy ```.env.example``` to ```.env``` and modify it.

```cp .env.template .env```

Modify the api endpoint in .env file if you deploy the api in another server. And you can specify another embedding model which you like in .env file. Note that the embedding model should be in the HuggingFace model hub.


Then, you should get the pinecone api key and environment in the pinecone console. And modify the pinecone api key and environment in .env file.

## 3. Run the bot
```python web.py```

The bot will run on port 7860 by default. You can access the your bot by http://localhost:7860




# 🩺 MedBot-Generative-AI

A conversational AI assistant built using Flask and LangChain that helps users get step-by-step medical guidance based on their symptoms. It uses Groq's large language model and Pinecone vector search to retrieve and reason over medical knowledge.

---

## 🚀 Features

- Friendly and empathetic medical chatbot
- Responds step-by-step to user queries
- Final diagnosis with treatment recommendations after multiple interactions
- Uses LangChain + Groq LLM + Pinecone for knowledge retrieval and response generation
- Emergency checks and intelligent greetings
- Web interface built using Flask

---

## 🧠 How It Works

1. **User sends a message** on the chat interface.
2. The app:
   - Checks for emergencies or goodbye messages.
   - Retrieves top 5 relevant medical documents using Pinecone.
   - Builds a prompt with full conversation history and retrieved knowledge.
   - Calls the Groq LLM (e.g., `mistral-saba-24b`) to generate a response.
3. After 6 messages, it triggers a final diagnosis and treatment advice.

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

Create a `.env` file with:

```env
PINECONE_API_KEY=your_pinecone_api_key
GROQ_API_KEY=your_groq_api_key
```

### 4. Run the App

```bash
python app.py
```

App will be live at `http://localhost:8080`.

---

## 🔐 Notes

- This assistant is **not a replacement for a real doctor**.
- It offers **general guidance only**, based on the provided text and retrieved info.
- Emergency symptoms trigger a hardcoded urgent response.

---

## 🙌 Acknowledgements

- [LangChain](https://www.langchain.com/)
- [Groq LLM](https://groq.com/)
- [Pinecone](https://www.pinecone.io/)
- [Flask](https://flask.palletsprojects.com/)
- Hugging Face for embeddings

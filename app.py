from flask import Flask, render_template, request, session
from src.helper import download_hugging_face_embeddings
from langchain_openai import OpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Pinecone
import string

app = Flask(__name__)
app.secret_key = "SECRET_KEY"

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Load embeddings using your helper function.
embeddings = download_hugging_face_embeddings()
index_name = "medicalbot"

# Connect to Pinecone index for knowledge retrieval.
docsearch = Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
# Retrieve up to 5 documents.
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Define the Groq LLM.
llm = ChatGroq(model="mistral-saba-24b", temperature=0.4, max_tokens=150)

# Updated prompt template.
base_prompt = ChatPromptTemplate.from_template(
    """
{greeting_line}
You are an intelligent, compassionate, and strictly medical assistant, fine-tuned solely for medical topics.
Your task is to diagnose the user's condition step by step using the provided Medical Knowledge.
Answer in one concise sentence or short paragraph.
If further clarification is needed, ask exactly one focused follow-up question.
However, if sufficient information is provided (as indicated by {final_line}), provide a final diagnosis with brief treatment recommendations.
If the provided Medical Knowledge does not directly mention the condition, rely on your internal medically validated knowledge to suggest safe, general remedies and include a disclaimer to consult a healthcare provider.
Conclude the conversation without asking any further questions.

User's latest message:
"{user_input}"

Full Conversation History:
{conversation_history}

Medical Knowledge:
{context}

{final_line}

Based solely on the above, provide your response as follows:
- If {final_line} is empty, ask one specific follow-up question.
- If {final_line} is present, give a final diagnosis with treatment recommendations and conclude the conversation.
"""
)

def normalize_text(text):
    return text.lower().translate(str.maketrans('', '', string.punctuation)).strip()

# We'll remove the generic "is_medical_query" fallback so that if conversation exists, we always process the input.
def is_medical_query(user_input):
    valid = {"yes", "no", "idk", "i don't know", "not sure", "maybe", "possibly", "unsure"}
    if user_input.lower() in valid or any(char.isdigit() for char in user_input):
        return True
    keywords = [
        "fever", "cramp", "period", "menstrual", "sore throat", "headache", "cough", "pain", 
        "nausea", "bleeding", "sick", "illness", "remedy", "remedies", "acne", "diarrhea",
        "infection", "diagnose", "treatment", "symptom", "meds", "medication"
    ]
    return any(keyword in user_input.lower() for keyword in keywords)

def get_greeting_line(history):
    # Greet only on the very first message.
    if len(history) == 1:
        return "Hello! I'm here to help."
    return ""

def get_final_line(history):
    # Once the conversation has reached six or more turns, instruct a final diagnosis.
    if len(history) >= 6:
        return "Now provide a final diagnosis with brief treatment recommendations and conclude the conversation."
    return ""

def generate_dynamic_response(history):
    user_input = history[-1].strip().lower()
    print("User input:", user_input)
    
    # Emergency check.
    if any(kw in user_input for kw in ["chest pain", "difficulty breathing", "emergency", "heart attack"]):
        session.clear()
        return ("It sounds like you may be experiencing a medical emergency. Please call your local emergency services immediately. I am not a doctor.")
    
    # Termination check.
    if any(phrase in user_input for phrase in ["thanks", "thank you", "bye", "goodbye", "that's all", "solved", "diagnosed"]):
        session.clear()
        return ("You're welcome! I hope you feel better soon. Take care!")
    
    # Instead of checking if input is medical-related, we always proceed if there's conversation history.
    
    # Retrieve context from the vector store.
    retrieved_docs = retriever.get_relevant_documents(user_input)
    medical_knowledge = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Build conversation history.
    conversation_history = "\n".join(history)
    
    # Determine greeting and final diagnosis lines.
    greeting_line = get_greeting_line(history)
    final_line = get_final_line(history)
    
    # Build the final prompt.
    final_prompt = base_prompt.format(
        greeting_line=greeting_line,
        user_input=user_input,
        conversation_history=conversation_history,
        context=medical_knowledge,
        final_line=final_line
    )
    
    print("Final Prompt:", final_prompt)
    try:
        response = llm.invoke(final_prompt)
        answer = response.content
    except Exception as e:
        answer = "I'm sorry, I'm having trouble processing your request at the moment."
        print("LLM error:", e)
    
    return answer

@app.route("/")
def index():
    session.clear()
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    if "history" not in session:
        session["history"] = []
    msg = request.form["msg"].strip()
    session["history"].append(msg)
    response = generate_dynamic_response(session["history"])
    session.modified = True
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from typing import List, Optional
import os
import tempfile
from PyPDF2 import PdfReader
from database import SessionLocal
from models import User, Chat, Document
from schemas import UserCreate, ChatRequest, ChatResponse, Token, UserResponse
from docx import Document as DocxDocument


# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI API
import google.generativeai as genai
genai.configure(api_key=genai_api_key)

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# App setup
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

# JWT and password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions for auth
def get_password_hash(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# Endpoints

@app.post("/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    db_user = User(
        username=user.username, 
        full_name=user.full_name, 
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    chat_history = get_user_chat_history(current_user.username, db)
    vector_store = FAISS.load_local("faiss_index", embeddings=GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY))
    chain = get_conversational_chain(vector_store)
    
    response = chain.invoke({"question": chat_request.question, "chat_history": chat_history})
    reply = response.get("answer", "Sorry, I couldn't fetch the response.")

    chat = Chat(username=current_user.username, user_text=chat_request.question, bot_reply=reply)
    db.add(chat)
    db.commit()
    return ChatResponse(question=chat_request.question, answer=reply)

@app.post("/upload_documents/", response_model=dict)
async def upload_documents(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)  # Ensures token is required
):
    current_user = get_current_user(db, token)  # Validate user with token
    processed_texts = []
    
    for file in files:
        if file.content_type == "application/pdf":
            pdf_text = input_pdf_text(file.file)
            processed_texts.append(pdf_text)
        elif file.content_type == "text/plain":
            text = await file.read()
            processed_texts.append(text.decode("utf-8"))
        elif file.content_type in [
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ]:
            doc_text = input_doc_text(file.file)
            processed_texts.append(doc_text)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Upload .pdf, .txt, or .doc/.docx files only.")
    
    # Pass the actual files (UploadFile objects) instead of their filenames
    labeled_text = label_and_combine_texts(files, processed_texts)
    text_chunks = get_text_chunks(labeled_text)
    store_vector_index(text_chunks)

    return {"msg": "Documents processed and indexed successfully"}

# Helper functions for extracting text from various formats
def input_pdf_text(file):
    pdf_reader = PdfReader(file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() or ""
    return pdf_text

def input_doc_text(file):
    text = docx2txt.process(file)
    return text

# Helper functions for document processing

def convert_to_pdf(uploaded_files):
    pdf_files = []
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            file_content = file.file.read()
            with open(temp_pdf.name, "wb") as pdf_out:
                pdf_out.write(file_content)
            pdf_files.append(temp_pdf.name)
    return pdf_files

def get_pdf_text(pdf_docs):
    texts = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:  
            texts += page.extract_text() or ""
    return texts

def label_and_combine_texts(uploaded_files, pdf_texts):
    labeled_texts = []
    for i, text in enumerate(pdf_texts):
        file_name = uploaded_files[i].filename  
        labeled_text = f"[START OF {file_name}] " + text + f" [END OF {file_name}]"
        labeled_texts.append(labeled_text)
    return " ".join(labeled_texts)


def get_text_chunks(labeled_text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(labeled_text)

def store_vector_index(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(vector_store):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, api_key=GOOGLE_API_KEY)
    retriever = vector_store.as_retriever()
    chain = ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever)
    return chain

# Function to handle user input and get responses
def user_input(user_question, chat_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    chain = get_conversational_chain(new_db)
    
    # Convert chat history to the expected format (list of tuples with (user, bot) pairs)
    chat_history_tuples = [(chat['user'], chat['bot']) for chat in chat_history]
    
    # Pass the properly formatted chat history to the chain
    response = chain.invoke({"question": user_question, "chat_history": chat_history_tuples})
    
    # Check and handle the response structure
    if 'answer' in response:
        reply = response['answer']
    elif 'output_text' in response:
        reply = response['output_text']
    else:
        reply = "Sorry, I couldn't fetch the response."

    # Update chat history with the current question and response
    chat_history.append({"user": user_question, "bot": reply})

    return chat_history

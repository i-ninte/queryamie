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
from email_utils import send_email
import tempfile
import re
from typing import List, Tuple
from PyPDF2 import PdfReader
from database import SessionLocal
import models
from database import engine
from models import User, Chat, Document
from schemas import UserCreate, ChatRequest, ChatResponse, Token, UserResponse
from docx import Document as DocxDocument

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI API
import google.generativeai as genai
genai.configure(api_key=genai_api_key)

# Constants for JWT
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# App setup
app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

# Initialize db
models.Base.metadata.create_all(bind=engine)

# Database session dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Helper functions for authentication

def get_password_hash(password: str) -> str:
    """Hashes a password for secure storage."""
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a password against a stored hash."""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generates a JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    """Retrieves the current user based on JWT token."""
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

def create_password_reset_token(email: str):
    to_encode = {"sub": email, "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)}
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def verify_password_reset_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None



# Function to retrieve user chat history
def get_user_chat_history(username: str, db: Session, limit: int = 10) -> List[dict]:
    """Fetches the recent chat history for a given user."""
    chat_history = (
        db.query(Chat)
        .filter(Chat.username == username)
        .order_by(Chat.id.desc())
        .limit(limit)
        .all()
    )
    return [{"user_text": chat.user_text, "bot_reply": chat.bot_reply} for chat in reversed(chat_history)]





# Forgot Password Endpoint
@app.post("/forgot-password")
async def forgot_password(email: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    token = create_password_reset_token(email)
    reset_link = f"http://queryamie.vercel.app/ResetPassword?token={token}"

    send_email(
        to_email=email,
        subject="Password Reset Request",
        body=f"Please click the following link to reset your password: {reset_link}"
    )
    return {"message": "Password reset link sent to your email"}

from pydantic import BaseModel

class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str



# Reset Password Endpoint
@app.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    email = verify_password_reset_token(request.token)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired token")

    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Set the new hashed password to the correct field
    hashed_password = get_password_hash(request.new_password)
    user.hashed_password = hashed_password
    db.commit()
    return {"message": "Password reset successful"}

#sign up endpoint
@app.post("/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    """Endpoint to register a new user."""
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


from pydantic import BaseModel

# Define a new response model to include user_id and token details
class TokenWithUserID(BaseModel):
    access_token: str
    token_type: str
    user_id: int

@app.post("/login", response_model=TokenWithUserID)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    # Verify password with the correct hashed password field
    if not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    
    # Create and return the access token
    token = create_access_token(data={"sub": user.username})
    return {"access_token": token, "token_type": "bearer", "user_id": user.id}


# Define a new endpoint to retrieve user details
@app.get("/chat_history", response_model=List[ChatResponse])
async def chat_history(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Endpoint to retrieve recent chat history for the authenticated user."""
    history = get_user_chat_history(current_user.username, db)
    return history


# Function to format tables, links, emails, and apply line spacing and paragraph formatting to response text
def clean_response_text(text: str) -> str:
    """Removes unwanted characters, applies paragraph formatting, and formats response text with HTML."""

    # Remove special characters like * or /
    text = re.sub(r"[*/]", "", text).strip()
    
    # Add paragraph spacing and line breaks
    text = re.sub(r"\n\s*\n", "\n\n", text)  # Ensure existing line breaks are spaced correctly
    text = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", text)  # Convert single newlines to double for paragraph spacing

    # Apply additional formatting for tables, URLs, and email addresses
    text = format_response(text)
    
    # Clean up any extra whitespace
    return text.strip()

def format_response(response_content: str) -> str:
    """Formats response content with HTML for tables, links, line breaks, and removes unwanted markdown symbols."""
    
    # Check if the content has a table structure and format it
    if '|' in response_content:
        response_content = format_table(response_content)
    
    # Replace newline characters with HTML line breaks
    response_content = re.sub(r'\n', ' <br>', response_content)
    
    # Format email addresses as mailto links
    response_content = re.sub(
        r'([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', 
        r'<a href="mailto:\1">\1</a>', 
        response_content
    )
    
    # Format URLs as clickable links
    response_content = re.sub(
        r'(https?://[^\s]+)', 
        r'<a href="\1">\1</a>', 
        response_content
    )
    
    # Format URLs starting with 'www' as clickable links
    response_content = re.sub(
        r'\b(www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,6}[^\s]*)', 
        r'<a href="http://\1">\1</a>', 
        response_content
    )
    
    # Remove unwanted markdown symbols like **, ##, or _ for clean display
    clean_response = re.sub(r'\*\*|\#\#|\_', '', response_content).strip()

    return clean_response

def format_table(response_content: str) -> str:
    """Formats any table structure in the response content as an HTML table."""
    
    # Split the content into lines, each representing a row in the table
    rows = response_content.split('\n')
    
    # Start building the HTML table
    table_html = '<table border="1" cellpadding="5" cellspacing="0">'
    
    for row in rows:
        if '|' in row:  # Only format rows with table structure
            # Split the row into cells using '|' as the delimiter
            cells = row.split('|')
            # Clean up whitespace from each cell
            cells = [cell.strip() for cell in cells if cell.strip()]
            
            # Add cells to the table row
            if cells:
                table_html += '<tr>'
                for cell in cells:
                    table_html += f'<td>{cell}</td>'
                table_html += '</tr>'
    
    table_html += '</table>'
    
    return table_html

# Update the chat endpoint with the new clean and formatted response
@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """Endpoint for querying the chat system, responding based on previous conversations."""
    
    # Retrieve the user's chat history and ensure it's in the expected format
    chat_history = get_user_chat_history(current_user.username, db) or []
    chat_history_formatted: List[Tuple[str, str]] = [(entry['user_text'], entry['bot_reply']) for entry in chat_history]
    
    # Load the vector store and conversational chain
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        allow_dangerous_deserialization=True
    )
    chain = get_conversational_chain(vector_store)
    
    # Invoke the conversational chain with the formatted chat history
    response = chain.invoke({"question": chat_request.question, "chat_history": chat_history_formatted})
    reply = response.get("answer", "Sorry, I couldn't fetch the response.")
    
    # Clean and format the reply text
    cleaned_reply = clean_response_text(reply)
    
    # Store the new chat entry in the database with the cleaned reply
    chat = Chat(username=current_user.username, user_text=chat_request.question, bot_reply=cleaned_reply)
    db.add(chat)
    db.commit()
    
    # Return the cleaned and formatted chat response
    return ChatResponse(question=chat_request.question, answer=cleaned_reply)

@app.post("/upload_documents/", response_model=dict)
async def upload_documents(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
):
    """Uploads and processes multiple document files, indexing content for later retrieval."""
    current_user = get_current_user(db, token)
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
    
    labeled_text = label_and_combine_texts(files, processed_texts)
    text_chunks = get_text_chunks(labeled_text)
    store_vector_index(text_chunks)

    return {"msg": "Documents processed and indexed successfully"}

# Helper functions for text extraction

def input_pdf_text(file):
    """Extracts text from a PDF file."""
    pdf_reader = PdfReader(file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text() or ""
    return pdf_text

def input_doc_text(file):
    """Extracts text from a DOCX file."""
    text = docx2txt.process(file)
    return text

# Helper functions for document processing

def label_and_combine_texts(uploaded_files, processed_texts):
    """Adds labels to texts based on file names and combines them."""
    labeled_texts = []
    for i, text in enumerate(processed_texts):
        file_name = uploaded_files[i].filename  
        labeled_text = f"[START OF {file_name}] " + text + f" [END OF {file_name}]"
        labeled_texts.append(labeled_text)
    return " ".join(labeled_texts)

def get_text_chunks(labeled_text):
    """Splits text into manageable chunks for indexing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(labeled_text)

def store_vector_index(text_chunks):
    """Stores text chunks in a FAISS index."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(vector_store):
    """Configures a conversational retrieval chain with FAISS and Google Generative AI."""
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4, api_key=genai_api_key)
    retriever = vector_store.as_retriever()
    return ConversationalRetrievalChain.from_llm(llm=model, retriever=retriever)

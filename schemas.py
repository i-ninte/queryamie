from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    username: str
    full_name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    question: str
    answer: str

class UserResponse(BaseModel):
    id: int
    username: str
    full_name: str
    email: EmailStr
    
    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str


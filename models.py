from sqlalchemy import Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), unique=True, nullable=False, index=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)

    chats = relationship("Chat", back_populates="user")
    documents = relationship("Document", back_populates="user")

    # Relationship for easier back-reference
    chats = relationship("Chat", back_populates="user")
    documents = relationship("Document", back_populates="user")

class Chat(Base):
    __tablename__ = "chats"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(255), ForeignKey("users.username"), nullable=False)
    user_text = Column(Text, nullable=False)
    bot_reply = Column(Text, nullable=False)

    # Define relationship to user
    user = relationship("User", back_populates="chats")

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    filetype = Column(String(50), nullable=False)
    content = Column(Text, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)

    # Define relationship to user
    user = relationship("User", back_populates="documents")

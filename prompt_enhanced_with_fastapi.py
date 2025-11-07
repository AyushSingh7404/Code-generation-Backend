import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import boto3
import asyncio
from collections import defaultdict
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# FastAPI app
app = FastAPI(
    title="AI Code Assistant API",
    description="AI-powered code generation and modification assistant using AWS Bedrock",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# AWS Configuration
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Initialize Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
)

# Pydantic Models
class FileContext(BaseModel):
    path: str
    content: str

class WorkspaceNode(BaseModel):
    name: str
    type: str  # "file" or "folder"
    children: Optional[List['WorkspaceNode']] = None

class WorkspaceTree(BaseModel):
    root: str
    children: List[WorkspaceNode]

class ChatContext(BaseModel):
    open_files: Optional[List[FileContext]] = None
    workspace_tree: Optional[WorkspaceTree] = None

class ChatRequest(BaseModel):
    query: str
    context: Optional[ChatContext] = None
    session_id: str = "default"

class ChatResponse(BaseModel):
    response: str
    parsed: Optional[Dict[str, Any]] = None
    session_id: str
    is_code_change: bool
    request_type: str  # "generation" or "modification"

class ResetRequest(BaseModel):
    session_id: str = "default"

MAIN_SYSTEM_PROMPT = """You are an expert AI code assistant specializing in code generation and modification. You help developers write clean, maintainable code while maintaining a friendly and helpful demeanor.

<core_responsibilities>
1. Generate well-structured, production-ready code with proper project organization
2. Accurately classify user requests as either code generation or code modification
3. Provide precise, detail-oriented code solutions
4. Follow language-specific best practices and consistent coding styles
5. Handle general conversation naturally when appropriate
</core_responsibilities>

<interaction_guidelines>
- Respond naturally to greetings (hello, hi, hey), gratitude (thank you, thanks), and general queries
- For casual conversation, provide brief, friendly responses without code generation
- For code-related requests, proceed with analysis and implementation
- Maintain context across the conversation
- Ask clarifying questions when requirements are ambiguous
</interaction_guidelines>

<request_classification>
<code_generation>
Triggered when user requests:
- Creating new code, applications, or features from scratch
- Building new files or project structures
- Generating boilerplate or starter code
- Examples: "create a flask app", "build a react component", "generate a REST API"
</code_generation>

<code_modification>
Triggered when user requests:
- Changes to existing code (provided in context or previously generated)
- Updates, fixes, or improvements to current code
- Adding features to existing implementations
- Refactoring or optimizing existing code
- Examples: "change the port to 3000", "add error handling", "fix the bug in this function"

Note: Modification requires either:
- Files provided in the context (open files in workspace)
- Previously generated code in the current session
</code_modification>
</request_classification>

<quality_standards>
- Prioritize correctness and clarity over speed
- Use meaningful variable and function names
- Include appropriate comments for complex logic
- Follow DRY (Don't Repeat Yourself) principles
- Consider edge cases and error handling
- Ensure proper indentation and formatting
</quality_standards>

<context_awareness>
When context is provided (open files, workspace structure):
- Analyze the existing codebase structure and patterns
- Match the coding style and conventions used
- Consider file organization and naming patterns
- Reference existing imports, dependencies, and configurations
- Maintain consistency with the existing architecture
</context_awareness>

<response_modes>
For code generation: Use the structured JSON format specified in the generation prompt
For code modification: Use the line-based JSON format specified in the modification prompt
For general queries: Respond conversationally without JSON formatting
</response_modes>"""


CODE_GENERATION_EXAMPLES_PROMPT = """<mode>CODE GENERATION</mode>

<instructions>
You are now in code generation mode. Generate complete, functional code with proper file and folder structure.

CRITICAL: Your response must contain ONLY valid JSON. Do not include any explanatory text, comments, or markdown before or after the JSON structure.
</instructions>

<json_structure>
{
  "type": "code_generation",
  "changes": [
    {
      "file": "path/to/file.py",
      "content": "complete file content as a string with escaped newlines"
    }
  ],
  "summary": "Brief description of what was generated"
}
</json_structure>

<rules>
1. Each file must be a separate object in the "changes" array
2. Use proper file paths relative to the project root (e.g., "src/main.py", "components/Header.jsx")
3. Include complete, ready-to-use code in the "content" field
4. Escape special characters properly (use \\n for newlines, \\" for quotes, \\\\ for backslashes)
5. Maintain proper indentation within the content string
6. Include all necessary imports and dependencies
7. Add a concise summary describing what was generated and key features
8. Generate all related files (main code, configuration, dependencies, styles, etc.)
</rules>

<examples>
<example>
<scenario>User: "Create a simple Flask hello world app"</scenario>
<response>
{
  "type": "code_generation",
  "changes": [
    {
      "file": "app.py",
      "content": "from flask import Flask\\n\\napp = Flask(__name__)\\n\\n@app.route('/')\\ndef hello():\\n    return 'Hello, World!'\\n\\nif __name__ == '__main__':\\n    app.run(debug=True, port=5000)"
    },
    {
      "file": "requirements.txt",
      "content": "Flask==3.0.0\\nWerkzeug==3.0.0"
    },
    {
      "file": ".gitignore",
      "content": "__pycache__/\\n*.py[cod]\\n*$py.class\\n.env\\nvenv/\\n.vscode/"
    }
  ],
  "summary": "Created Flask hello world application with dependencies and gitignore"
}
</response>
</example>

<example>
<scenario>User: "Build a FastAPI project with user CRUD endpoints"</scenario>
<response>
{
  "type": "code_generation",
  "changes": [
    {
      "file": "main.py",
      "content": "from fastapi import FastAPI\\nfrom routes.users import router as user_router\\nfrom database import engine, Base\\n\\nBase.metadata.create_all(bind=engine)\\n\\napp = FastAPI(title=\\"User Management API\\")\\n\\napp.include_router(user_router, prefix='/api/v1/users', tags=['users'])\\n\\n@app.get('/')\\ndef root():\\n    return {'message': 'User Management API', 'version': '1.0.0'}"
    },
    {
      "file": "database.py",
      "content": "from sqlalchemy import create_engine\\nfrom sqlalchemy.ext.declarative import declarative_base\\nfrom sqlalchemy.orm import sessionmaker\\n\\nDATABASE_URL = \\"sqlite:///./users.db\\"\\n\\nengine = create_engine(DATABASE_URL, connect_args={\\"check_same_thread\\": False})\\nSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)\\nBase = declarative_base()\\n\\ndef get_db():\\n    db = SessionLocal()\\n    try:\\n        yield db\\n    finally:\\n        db.close()"
    },
    {
      "file": "models/user.py",
      "content": "from sqlalchemy import Column, Integer, String, Boolean\\nfrom database import Base\\n\\nclass User(Base):\\n    __tablename__ = 'users'\\n    \\n    id = Column(Integer, primary_key=True, index=True)\\n    name = Column(String, nullable=False)\\n    email = Column(String, unique=True, index=True, nullable=False)\\n    is_active = Column(Boolean, default=True)"
    },
    {
      "file": "schemas/user.py",
      "content": "from pydantic import BaseModel, EmailStr\\nfrom typing import Optional\\n\\nclass UserBase(BaseModel):\\n    name: str\\n    email: EmailStr\\n\\nclass UserCreate(UserBase):\\n    pass\\n\\nclass UserUpdate(BaseModel):\\n    name: Optional[str] = None\\n    email: Optional[EmailStr] = None\\n    is_active: Optional[bool] = None\\n\\nclass UserResponse(UserBase):\\n    id: int\\n    is_active: bool\\n    \\n    class Config:\\n        from_attributes = True"
    },
    {
      "file": "routes/users.py",
      "content": "from fastapi import APIRouter, Depends, HTTPException\\nfrom sqlalchemy.orm import Session\\nfrom typing import List\\nfrom database import get_db\\nfrom models.user import User\\nfrom schemas.user import UserCreate, UserUpdate, UserResponse\\n\\nrouter = APIRouter()\\n\\n@router.get('/', response_model=List[UserResponse])\\ndef get_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):\\n    users = db.query(User).offset(skip).limit(limit).all()\\n    return users\\n\\n@router.get('/{user_id}', response_model=UserResponse)\\ndef get_user(user_id: int, db: Session = Depends(get_db)):\\n    user = db.query(User).filter(User.id == user_id).first()\\n    if not user:\\n        raise HTTPException(status_code=404, detail='User not found')\\n    return user\\n\\n@router.post('/', response_model=UserResponse, status_code=201)\\ndef create_user(user: UserCreate, db: Session = Depends(get_db)):\\n    db_user = db.query(User).filter(User.email == user.email).first()\\n    if db_user:\\n        raise HTTPException(status_code=400, detail='Email already registered')\\n    \\n    new_user = User(**user.dict())\\n    db.add(new_user)\\n    db.commit()\\n    db.refresh(new_user)\\n    return new_user\\n\\n@router.put('/{user_id}', response_model=UserResponse)\\ndef update_user(user_id: int, user: UserUpdate, db: Session = Depends(get_db)):\\n    db_user = db.query(User).filter(User.id == user_id).first()\\n    if not db_user:\\n        raise HTTPException(status_code=404, detail='User not found')\\n    \\n    for key, value in user.dict(exclude_unset=True).items():\\n        setattr(db_user, key, value)\\n    \\n    db.commit()\\n    db.refresh(db_user)\\n    return db_user\\n\\n@router.delete('/{user_id}', status_code=204)\\ndef delete_user(user_id: int, db: Session = Depends(get_db)):\\n    db_user = db.query(User).filter(User.id == user_id).first()\\n    if not db_user:\\n        raise HTTPException(status_code=404, detail='User not found')\\n    \\n    db.delete(db_user)\\n    db.commit()\\n    return None"
    },
    {
      "file": "requirements.txt",
      "content": "fastapi==0.104.1\\nuvicorn==0.24.0\\nsqlalchemy==2.0.23\\npydantic[email]==2.5.0"
    }
  ],
  "summary": "Created complete FastAPI application with user CRUD operations, SQLAlchemy ORM, Pydantic schemas, and proper project structure"
}
</response>
</example>

<example>
<scenario>User: "Create a React login form component with validation"</scenario>
<response>
{
  "type": "code_generation",
  "changes": [
    {
      "file": "components/LoginForm.jsx",
      "content": "import React, { useState } from 'react';\\nimport './LoginForm.css';\\n\\nconst LoginForm = ({ onSubmit }) => {\\n  const [formData, setFormData] = useState({\\n    email: '',\\n    password: ''\\n  });\\n  const [errors, setErrors] = useState({});\\n  const [isLoading, setIsLoading] = useState(false);\\n\\n  const validateEmail = (email) => {\\n    const regex = /^[^\\\\s@]+@[^\\\\s@]+\\\\.[^\\\\s@]+$/;\\n    return regex.test(email);\\n  };\\n\\n  const validateForm = () => {\\n    const newErrors = {};\\n\\n    if (!formData.email) {\\n      newErrors.email = 'Email is required';\\n    } else if (!validateEmail(formData.email)) {\\n      newErrors.email = 'Invalid email format';\\n    }\\n\\n    if (!formData.password) {\\n      newErrors.password = 'Password is required';\\n    } else if (formData.password.length < 6) {\\n      newErrors.password = 'Password must be at least 6 characters';\\n    }\\n\\n    setErrors(newErrors);\\n    return Object.keys(newErrors).length === 0;\\n  };\\n\\n  const handleChange = (e) => {\\n    const { name, value } = e.target;\\n    setFormData(prev => ({\\n      ...prev,\\n      [name]: value\\n    }));\\n    // Clear error when user starts typing\\n    if (errors[name]) {\\n      setErrors(prev => ({\\n        ...prev,\\n        [name]: ''\\n      }));\\n    }\\n  };\\n\\n  const handleSubmit = async (e) => {\\n    e.preventDefault();\\n    \\n    if (!validateForm()) {\\n      return;\\n    }\\n\\n    setIsLoading(true);\\n    try {\\n      await onSubmit(formData);\\n    } catch (error) {\\n      setErrors({ submit: error.message || 'Login failed' });\\n    } finally {\\n      setIsLoading(false);\\n    }\\n  };\\n\\n  return (\\n    <div className=\\"login-container\\">\\n      <form className=\\"login-form\\" onSubmit={handleSubmit}>\\n        <h2>Login</h2>\\n        \\n        <div className=\\"form-group\\">\\n          <label htmlFor=\\"email\\">Email</label>\\n          <input\\n            type=\\"email\\"\\n            id=\\"email\\"\\n            name=\\"email\\"\\n            value={formData.email}\\n            onChange={handleChange}\\n            className={errors.email ? 'error' : ''}\\n            disabled={isLoading}\\n          />\\n          {errors.email && <span className=\\"error-message\\">{errors.email}</span>}\\n        </div>\\n\\n        <div className=\\"form-group\\">\\n          <label htmlFor=\\"password\\">Password</label>\\n          <input\\n            type=\\"password\\"\\n            id=\\"password\\"\\n            name=\\"password\\"\\n            value={formData.password}\\n            onChange={handleChange}\\n            className={errors.password ? 'error' : ''}\\n            disabled={isLoading}\\n          />\\n          {errors.password && <span className=\\"error-message\\">{errors.password}</span>}\\n        </div>\\n\\n        {errors.submit && (\\n          <div className=\\"error-message submit-error\\">{errors.submit}</div>\\n        )}\\n\\n        <button \\n          type=\\"submit\\" \\n          className=\\"submit-btn\\"\\n          disabled={isLoading}\\n        >\\n          {isLoading ? 'Logging in...' : 'Login'}\\n        </button>\\n      </form>\\n    </div>\\n  );\\n};\\n\\nexport default LoginForm;"
    },
    {
      "file": "components/LoginForm.css",
      "content": ".login-container {\\n  display: flex;\\n  justify-content: center;\\n  align-items: center;\\n  min-height: 100vh;\\n  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\\n}\\n\\n.login-form {\\n  background: white;\\n  padding: 2rem;\\n  border-radius: 10px;\\n  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);\\n  width: 100%;\\n  max-width: 400px;\\n}\\n\\n.login-form h2 {\\n  margin: 0 0 1.5rem 0;\\n  text-align: center;\\n  color: #333;\\n  font-size: 1.8rem;\\n}\\n\\n.form-group {\\n  margin-bottom: 1.5rem;\\n}\\n\\n.form-group label {\\n  display: block;\\n  margin-bottom: 0.5rem;\\n  color: #555;\\n  font-weight: 500;\\n}\\n\\n.form-group input {\\n  width: 100%;\\n  padding: 0.75rem;\\n  border: 2px solid #e0e0e0;\\n  border-radius: 5px;\\n  font-size: 1rem;\\n  transition: border-color 0.3s;\\n}\\n\\n.form-group input:focus {\\n  outline: none;\\n  border-color: #667eea;\\n}\\n\\n.form-group input.error {\\n  border-color: #e74c3c;\\n}\\n\\n.form-group input:disabled {\\n  background-color: #f5f5f5;\\n  cursor: not-allowed;\\n}\\n\\n.error-message {\\n  display: block;\\n  color: #e74c3c;\\n  font-size: 0.875rem;\\n  margin-top: 0.25rem;\\n}\\n\\n.submit-error {\\n  text-align: center;\\n  padding: 0.75rem;\\n  background-color: #fee;\\n  border-radius: 5px;\\n  margin-bottom: 1rem;\\n}\\n\\n.submit-btn {\\n  width: 100%;\\n  padding: 0.75rem;\\n  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\\n  color: white;\\n  border: none;\\n  border-radius: 5px;\\n  font-size: 1rem;\\n  font-weight: 600;\\n  cursor: pointer;\\n  transition: transform 0.2s, box-shadow 0.2s;\\n}\\n\\n.submit-btn:hover:not(:disabled) {\\n  transform: translateY(-2px);\\n  box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);\\n}\\n\\n.submit-btn:disabled {\\n  opacity: 0.6;\\n  cursor: not-allowed;\\n}"
    }
  ],
  "summary": "Created React login form component with email/password validation, error handling, loading states, and modern styling"
}
</response>
</example>
</examples>

<critical_reminders>
- Output ONLY the JSON structure
- No text before the opening brace {
- No text after the closing brace }
- Properly escape all special characters in content strings
- Ensure JSON is valid and parseable
</critical_reminders>"""


CODE_MODIFICATION_EXAMPLES_PROMPT = """<mode>CODE MODIFICATION</mode>

<instructions>
You are now in code modification mode. Provide precise, line-based changes to existing code files.

CRITICAL: Your response must contain ONLY valid JSON. Do not include any explanatory text, comments, or markdown before or after the JSON structure.
</instructions>

<json_structure>
{
  "type": "code_changes",
  "changes": [
    {
      "file": "path/to/file.py",
      "modifications": [
        {
          "operation": "replace" | "insert" | "delete" | "insert_before",
          "start_line": <number>,
          "end_line": <number>,
          "old_content": "exact existing content from the file",
          "new_content": "new content to insert or replace with"
        }
      ]
    }
  ],
  "summary": "Brief description of changes made"
}
</json_structure>

<operations>
<operation name="replace">
Purpose: Change existing lines in the file
Required fields:
- start_line: First line number to replace (1-indexed)
- end_line: Last line number to replace (inclusive, 1-indexed)
- old_content: Exact content being replaced (must match file exactly)
- new_content: New content to insert

Use when: Modifying existing code, fixing bugs, updating values
</operation>

<operation name="insert">
Purpose: Add new lines AFTER a specified line
Required fields:
- start_line: Line number after which to insert (1-indexed)
- new_content: Content to insert

Use when: Adding code after a specific location (e.g., after imports, after a function)
</operation>

<operation name="insert_before">
Purpose: Add new lines BEFORE a specified line
Required fields:
- start_line: Line number before which to insert (1-indexed)
- new_content: Content to insert

Use when: Adding code before a specific location (e.g., before a function, before main)
</operation>

<operation name="delete">
Purpose: Remove lines from the file
Required fields:
- start_line: First line number to delete (1-indexed)
- end_line: Last line number to delete (inclusive, 1-indexed)
- old_content: Exact content being deleted (must match file exactly)

Use when: Removing unnecessary code, deleting functions or blocks
</operation>
</operations>

<rules>
1. Line numbers are 1-indexed (first line of file is line 1)
2. old_content must match the file EXACTLY including:
   - All whitespace (spaces, tabs)
   - Exact indentation
   - Newline characters (use \\n)
3. Preserve proper indentation in new_content
4. Order modifications from top to bottom within each file
5. Multiple files can be modified in a single response
6. Only include necessary changes - don't modify what doesn't need changing
7. For multi-line changes, use \\n to represent line breaks in old_content and new_content
8. Be precise - incorrect old_content will cause modifications to fail
</rules>

<examples>
<example>
<scenario>
User: "Change the port from 8000 to 3000 in app.py"

Current file (app.py):
Line 1: from flask import Flask
Line 2: app = Flask(__name__)
Line 3: PORT = 8000
Line 4: app.run(port=PORT)
</scenario>
<response>
{
  "type": "code_changes",
  "changes": [
    {
      "file": "app.py",
      "modifications": [
        {
          "operation": "replace",
          "start_line": 3,
          "end_line": 3,
          "old_content": "PORT = 8000",
          "new_content": "PORT = 3000"
        }
      ]
    }
  ],
  "summary": "Updated port configuration from 8000 to 3000"
}
</response>
</example>

<example>
<scenario>
User: "Add email validation function before the main function in utils.py"

Current file (utils.py):
Line 1: import re
Line 2: 
Line 3: def main():
Line 4:     pass
</scenario>
<response>
{
  "type": "code_changes",
  "changes": [
    {
      "file": "utils.py",
      "modifications": [
        {
          "operation": "insert_before",
          "start_line": 3,
          "new_content": "def validate_email(email):\\n    pattern = r'^[\\\\w\\\\.-]+@[\\\\w\\\\.-]+\\\\.\\\\w+\\n    return bool(re.match(pattern, email))\\n\\n"
        }
      ]
    }
  ],
  "summary": "Added email validation function using regex pattern"
}
</response>
</example>

<example>
<scenario>
User: "In calc.py, make the add function handle floats and add a zero division check to the divide function"

Current file (calc.py):
Line 1: def add(a, b):
Line 2:     return a + b
Line 3: 
Line 4: def divide(a, b):
Line 5:     return a / b
</scenario>
<response>
{
  "type": "code_changes",
  "changes": [
    {
      "file": "calc.py",
      "modifications": [
        {
          "operation": "replace",
          "start_line": 1,
          "end_line": 2,
          "old_content": "def add(a, b):\\n    return a + b",
          "new_content": "def add(a, b):\\n    return float(a) + float(b)"
        },
        {
          "operation": "replace",
          "start_line": 4,
          "end_line": 5,
          "old_content": "def divide(a, b):\\n    return a / b",
          "new_content": "def divide(a, b):\\n    if b == 0:\\n        raise ValueError('Cannot divide by zero')\\n    return a / b"
        }
      ]
    }
  ],
  "summary": "Updated add function to handle floats and added zero division validation to divide function"
}
</response>
</example>

<example>
<scenario>
User: "Remove the debug function from helpers.py"

Current file (helpers.py):
Line 1: def helper():
Line 2:     pass
Line 3: 
Line 4: def debug(msg):
Line 5:     print(f"DEBUG: {msg}")
Line 6: 
Line 7: def main():
Line 8:     pass
</scenario>
<response>
{
  "type": "code_changes",
  "changes": [
    {
      "file": "helpers.py",
      "modifications": [
        {
          "operation": "delete",
          "start_line": 4,
          "end_line": 6,
          "old_content": "def debug(msg):\\n    print(f\\"DEBUG: {msg}\\")\\n"
        }
      ]
    }
  ],
  "summary": "Removed debug function from helpers module"
}
</response>
</example>

<example>
<scenario>
User: "Add error handling to the fetch_data function and update the API endpoint in config.py"

Current files:

api.py:
Line 1: import requests
Line 2: 
Line 3: def fetch_data(url):
Line 4:     response = requests.get(url)
Line 5:     return response.json()

config.py:
Line 1: API_URL = "http://localhost:8000"
Line 2: TIMEOUT = 30
</scenario>
<response>
{
  "type": "code_changes",
  "changes": [
    {
      "file": "api.py",
      "modifications": [
        {
          "operation": "replace",
          "start_line": 3,
          "end_line": 5,
          "old_content": "def fetch_data(url):\\n    response = requests.get(url)\\n    return response.json()",
          "new_content": "def fetch_data(url):\\n    try:\\n        response = requests.get(url, timeout=30)\\n        response.raise_for_status()\\n        return response.json()\\n    except requests.exceptions.RequestException as e:\\n        print(f\\"Error fetching data: {e}\\")\\n        return None"
        }
      ]
    },
    {
      "file": "config.py",
      "modifications": [
        {
          "operation": "replace",
          "start_line": 1,
          "end_line": 1,
          "old_content": "API_URL = \\"http://localhost:8000\\"",
          "new_content": "API_URL = \\"https://api.production.com\\""
        }
      ]
    }
  ],
  "summary": "Added comprehensive error handling to fetch_data function and updated API endpoint to production URL"
}
</response>
</example>
</examples>

<critical_reminders>
- Output ONLY the JSON structure
- No text before the opening brace {
- No text after the closing brace }
- old_content must EXACTLY match the file content including whitespace
- Use \\n for newlines in old_content and new_content strings
- Escape quotes and backslashes properly
- Ensure JSON is valid and parseable
- List modifications in top-to-bottom order per file
</critical_reminders>"""

# In-memory storage
conversations = {}
generated_code = {}
active_connections: Dict[str, WebSocket] = {}

class ConversationBuffer:
    def __init__(self, max_tokens=4000):
        self.messages = []
        self.max_tokens = max_tokens
        
    def add_message(self, role, content):
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        total_chars = sum(len(msg["content"]) for msg in self.messages)
        estimated_tokens = total_chars / 4
        
        if estimated_tokens > self.max_tokens and len(self.messages) > 4:
            messages_to_summarize = self.messages[:-4]
            recent_messages = self.messages[-4:]
            
            summary_content = "Previous conversation summary:\n"
            for msg in messages_to_summarize:
                summary_content += f"{msg['role']}: {msg['content'][:100]}...\n"
            
            self.messages = [
                {"role": "user", "content": summary_content, "timestamp": datetime.now().isoformat()}
            ] + recent_messages
    
    def get_messages_for_api(self):
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]

def build_context_string(context: Optional[ChatContext]) -> str:
    """Build context string from provided files and workspace"""
    if not context:
        return ""
    
    context_parts = []
    
    # Add open files
    if context.open_files:
        context_parts.append("=== OPEN FILES IN WORKSPACE ===")
        for file in context.open_files:
            context_parts.append(f"\nFile: {file.path}")
            context_parts.append("```")
            context_parts.append(file.content)
            context_parts.append("```\n")
    
    # Add workspace tree
    if context.workspace_tree:
        context_parts.append("=== WORKSPACE STRUCTURE ===")
        context_parts.append(f"Root: {context.workspace_tree.root}")
        context_parts.append(json.dumps(context.workspace_tree.dict(), indent=2))
    
    return "\n".join(context_parts)

def is_modification_request(query: str, has_context: bool, has_previous_code: bool) -> bool:
    """Determine if request is for modification"""
    modification_keywords = [
        'change', 'modify', 'update', 'fix', 'add', 'remove', 'delete',
        'edit', 'refactor', 'improve', 'adjust', 'alter', 'correct'
    ]
    
    reference_keywords = [
        'the code', 'above', 'previous', 'existing', 'current',
        'this code', 'that function', 'the function'
    ]
    
    query_lower = query.lower()
    
    # Check if it has modification keywords
    has_modification_keyword = any(keyword in query_lower for keyword in modification_keywords)
    
    # Check if it references existing code
    has_reference = any(keyword in query_lower for keyword in reference_keywords)
    
    # It's a modification if:
    # 1. Has modification keywords AND (has context OR has previous code)
    # 2. OR has reference keywords AND (has context OR has previous code)
    return (has_modification_keyword or has_reference) and (has_context or has_previous_code)

def call_claude_bedrock(messages: List[Dict], system_prompt: str, model_id: str = "arn:aws:bedrock:us-east-1:52986:inference-profile/global.anthropic.claude-sonnet-4-20250514-v1:0") -> str:
    """Call Claude via AWS Bedrock"""
    try:
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": messages,
            "temperature": 0.3
        }
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response['body'].read())
        return response_body['content'][0]['text']
    
    except Exception as e:
        raise Exception(f"Bedrock API Error: {str(e)}")

def sort_and_apply_modifications(changes: List[Dict]) -> List[Dict]:
    """Sort modifications by file, then by descending line number for reverse-order application"""
    # Group by file
    file_modifications = defaultdict(list)
    
    for change in changes:
        file_path = change.get('file', '')
        modifications = change.get('modifications', [])
        
        for mod in modifications:
            file_modifications[file_path].append(mod)
    
    # Sort each file's modifications by descending start_line
    sorted_changes = []
    for file_path, mods in file_modifications.items():
        sorted_mods = sorted(mods, key=lambda x: x.get('start_line', 0), reverse=True)
        sorted_changes.append({
            'file': file_path,
            'modifications': sorted_mods
        })
    
    return sorted_changes

async def process_chat_request(request: ChatRequest) -> ChatResponse:
    """Process chat request"""
    session_id = request.session_id
    query = request.query
    context = request.context
    
    # Initialize session if needed
    if session_id not in conversations:
        conversations[session_id] = ConversationBuffer()
        generated_code[session_id] = None
    
    conv_buffer = conversations[session_id]
    has_previous_code = generated_code[session_id] is not None
    has_context = context is not None and (context.open_files or context.workspace_tree)
    
    # Build context string
    context_string = build_context_string(context)
    
    # Determine request type
    is_modification = is_modification_request(query, has_context, has_previous_code)
    
    # Add user message
    conv_buffer.add_message("user", query)
    
    if is_modification:
        # Modification mode
        system_prompt = MAIN_SYSTEM_PROMPT + "\n\n" + CODE_MODIFICATION_EXAMPLES_PROMPT
        
        # Build message with context
        message_content = f"{query}\n\n"
        
        if context_string:
            message_content += f"{context_string}\n\n"
        
        if has_previous_code and not has_context:
            message_content += f"=== PREVIOUSLY GENERATED CODE ===\n{generated_code[session_id]}\n\n"
        
        message_content += "Provide the modifications in JSON format as specified."
        
        messages = [{"role": "user", "content": message_content}]
        response = call_claude_bedrock(messages, system_prompt)
        
        # Parse JSON response
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Sort modifications for reverse-order application
                if 'changes' in parsed:
                    parsed['changes'] = sort_and_apply_modifications(parsed['changes'])
                
                conv_buffer.add_message("assistant", response)
                
                return ChatResponse(
                    response=response,
                    parsed=parsed,
                    session_id=session_id,
                    is_code_change=True,
                    request_type="modification"
                )
            else:
                raise ValueError("No JSON found in response")
        
        except Exception as e:
            return ChatResponse(
                response=f"Error parsing modifications: {str(e)}\n\nRaw response: {response}",
                parsed=None,
                session_id=session_id,
                is_code_change=False,
                request_type="modification"
            )
    
    else:
        # Generation mode - Use CODE_GENERATION_EXAMPLES_PROMPT for structured JSON output
        system_prompt = MAIN_SYSTEM_PROMPT + "\n\n" + CODE_GENERATION_EXAMPLES_PROMPT
        
        # Build message with context
        message_content = f"{query}\n\n"
        
        if context_string:
            message_content += f"{context_string}\n\n"
        
        message_content += "Generate the code in JSON format as specified."
        
        messages = [{"role": "user", "content": message_content}]
        response = call_claude_bedrock(messages, system_prompt)
        
        # Parse JSON response for generation
        try:
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Store generated code (as JSON string for reference)
                generated_code[session_id] = json.dumps(parsed, indent=2)
                
                conv_buffer.add_message("assistant", response)
                
                return ChatResponse(
                    response=response,
                    parsed=parsed,
                    session_id=session_id,
                    is_code_change=True,
                    request_type="generation"
                )
            else:
                raise ValueError("No JSON found in response")
        
        except Exception as e:
            # Fallback: If JSON parsing fails, return as plain text
            generated_code[session_id] = response
            conv_buffer.add_message("assistant", response)
            
            return ChatResponse(
                response=response,
                parsed=None,
                session_id=session_id,
                is_code_change=False,
                request_type="generation"
            )

# REST API Endpoints
@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat request for code generation or modification.
    
    - **query**: The user's request/query
    - **context**: Optional context including open files and workspace structure
    - **session_id**: Session identifier for maintaining conversation history
    """
    try:
        return await process_chat_request(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset", tags=["Session"])
async def reset_session(request: ResetRequest):
    """Reset conversation history and stored code for a session"""
    session_id = request.session_id
    
    if session_id in conversations:
        del conversations[session_id]
    if session_id in generated_code:
        del generated_code[session_id]
    
    return {"message": f"Session {session_id} reset successfully"}

@app.get("/history/{session_id}", tags=["Session"])
async def get_history(session_id: str):
    """Get conversation history for a session"""
    if session_id not in conversations:
        return {"messages": [], "has_code": False}
    
    return {
        "messages": conversations[session_id].messages,
        "session_id": session_id,
        "has_code": generated_code.get(session_id) is not None
    }

@app.get("/code/{session_id}", tags=["Session"])
async def get_code(session_id: str):
    """Get currently stored code for a session"""
    if session_id not in generated_code or generated_code[session_id] is None:
        return {"code": None, "message": "No code generated yet"}
    
    return {
        "code": generated_code[session_id],
        "session_id": session_id
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(conversations),
        "active_websockets": len(active_connections)
    }

# WebSocket Endpoint
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat communication.
    
    Send JSON: {"query": "your request", "context": {...}}
    Receive JSON: ChatResponse model
    """
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            # Create request object
            request = ChatRequest(
                query=request_data.get('query', ''),
                context=ChatContext(**request_data.get('context', {})) if request_data.get('context') else None,
                session_id=session_id
            )
            
            # Process request
            response = await process_chat_request(request)
            
            # Send response
            await websocket.send_json(response.dict())
    
    except WebSocketDisconnect:
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        await websocket.send_json({"error": str(e)})
        if session_id in active_connections:
            del active_connections[session_id]

@app.get("/", tags=["System"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "AI Code Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "websocket": "/ws/{session_id}"
    }

if __name__ == "__main__":
    import uvicorn
    
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("WARNING: AWS credentials not set!")
        print("Set: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_REGION")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
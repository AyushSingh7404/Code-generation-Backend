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
    version="2.0.0"
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

# ============================================================================
# PROMPTS - RESTRUCTURED
# ============================================================================

# SHORT System Prompt (Role Definition Only) - Will be cached
MAIN_SYSTEM_PROMPT = """You are an expert AI code assistant specializing in code generation and modification. You help developers write clean, maintainable, production-ready code.

<core_responsibilities>
- Generate well-structured code with proper project organization
- Modify existing code accurately using line-based changes
- Respond naturally to casual conversation
- Follow language-specific best practices
- Maintain context throughout conversations
</core_responsibilities>

<interaction_style>
- For code requests: Provide structured JSON responses as instructed
- For casual chat: Respond conversationally without JSON
- Ask clarifying questions when requirements are ambiguous
- Be precise and detail-oriented in code generation
</interaction_style>

You will receive task-specific instructions and examples in the conversation. Follow them carefully."""


# Code Generation Examples and Instructions
CODE_GENERATION_EXAMPLES_PROMPT = """<mode>CODE GENERATION</mode>

<instructions>
You are now in code generation mode. Follow this process:

1. ANALYZE: First, understand the requirements and plan your approach
2. STRUCTURE: Determine the file structure and organization
3. GENERATE: Create the complete code in JSON format

CRITICAL OUTPUT FORMAT:
- Start with brief analysis (2-3 sentences explaining your approach)
- Then output ONLY valid JSON (no markdown, no code blocks, no extra text)
- Format: [Brief reasoning] + JSON structure
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
I'll create a basic Flask application with a hello world endpoint, including dependencies and gitignore.

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
I'll create a complete FastAPI application with proper structure: database setup, models, schemas, and CRUD routes.

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
</examples>

<critical_reminders>
- Output brief analysis (2-3 sentences) then ONLY the JSON structure
- No text before the opening brace { except the analysis
- No text after the closing brace }
- No markdown code blocks (```)
- Properly escape all special characters in content strings
- Ensure JSON is valid and parseable
</critical_reminders>"""


# Code Modification Examples and Instructions
CODE_MODIFICATION_EXAMPLES_PROMPT = """<mode>CODE MODIFICATION</mode>

<instructions>
You are now in code modification mode. Follow this process:

1. ANALYZE: Understand what needs to be changed and why
2. LOCATE: Identify exact line numbers and content to modify
3. MODIFY: Provide precise line-based changes in JSON format

CRITICAL OUTPUT FORMAT:
- Start with brief analysis (2-3 sentences explaining your changes)
- Then output ONLY valid JSON (no markdown, no code blocks, no extra text)
- Format: [Brief reasoning] + JSON structure
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

Use when: Adding code after a specific location
</operation>

<operation name="insert_before">
Purpose: Add new lines BEFORE a specified line
Required fields:
- start_line: Line number before which to insert (1-indexed)
- new_content: Content to insert

Use when: Adding code before a specific location
</operation>

<operation name="delete">
Purpose: Remove lines from the file
Required fields:
- start_line: First line number to delete (1-indexed)
- end_line: Last line number to delete (inclusive, 1-indexed)
- old_content: Exact content being deleted (must match file exactly)

Use when: Removing unnecessary code
</operation>
</operations>

<rules>
1. Line numbers are 1-indexed (first line of file is line 1)
2. old_content must match the file EXACTLY (whitespace, indentation, newlines)
3. Preserve proper indentation in new_content
4. Order modifications from top to bottom within each file
5. Use \\n to represent line breaks in old_content and new_content
6. Be precise - incorrect old_content will cause modifications to fail
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
I'll update the PORT constant from 8000 to 3000.

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
User: "Add error handling to the fetch_data function in api.py"

Current file (api.py):
Line 1: import requests
Line 2: 
Line 3: def fetch_data(url):
Line 4:     response = requests.get(url)
Line 5:     return response.json()
</scenario>
<response>
I'll wrap the fetch operation in try-except to handle request exceptions and add timeout.

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
    }
  ],
  "summary": "Added comprehensive error handling with timeout and status code checking to fetch_data function"
}
</response>
</example>
</examples>

<critical_reminders>
- Output brief analysis (2-3 sentences) then ONLY the JSON structure
- No markdown code blocks (```)
- old_content must EXACTLY match the file including all whitespace
- Use \\n for newlines in strings
- Escape quotes and backslashes properly
- Ensure JSON is valid and parseable
</critical_reminders>"""


# Unified Examples Conversation - This gets cached once per session
UNIFIED_EXAMPLES_CONVERSATION = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{CODE_GENERATION_EXAMPLES_PROMPT}\n\n---\n\n{CODE_MODIFICATION_EXAMPLES_PROMPT}",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    },
    {
        "role": "assistant",
        "content": "I understand both code generation and modification formats. For generation requests, I will create complete files with proper structure in JSON format. For modification requests, I will provide precise line-based changes using the appropriate operations (replace, insert, insert_before, delete). I will always start with brief analysis, then output only valid JSON without markdown or extra text."
    }
]


# ============================================================================
# CONVERSATION BUFFER - IMPROVED WITH CACHING SUPPORT
# ============================================================================

class ConversationBuffer:
    """Manages conversation history with caching support for examples"""
    
    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages
        self.examples_injected = False
    
    def inject_examples(self):
        """Inject unified examples at the start of conversation (only once)"""
        if not self.examples_injected and not self.messages:
            self.messages = UNIFIED_EXAMPLES_CONVERSATION.copy()
            self.examples_injected = True
    
    def add_message(self, role: str, content: str):
        """Add message to conversation history"""
        self.messages.append({
            "role": role,
            "content": content
        })
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Keep recent messages but PRESERVE examples at start"""
        if len(self.messages) > self.max_messages:
            if self.examples_injected:
                # Keep first 2 messages (examples conversation) + recent messages
                examples = self.messages[:2]
                recent = self.messages[-(self.max_messages - 2):]
                self.messages = examples + recent
            else:
                # No examples, just trim normally
                self.messages = self.messages[-self.max_messages:]
            
            # Ensure valid alternating pattern after examples
            if len(self.messages) > 2 and self.messages[2]["role"] != "user":
                self.messages = self.messages[:2] + self.messages[3:]
    
    def get_messages_for_api(self) -> List[Dict]:
        """
        Get messages formatted for Claude API with cache control
        Examples are already cached. We also cache older conversation history.
        """
        if len(self.messages) <= 5:
            # Too few messages to benefit from additional caching
            return self.messages
        
        # Examples (first 2 messages) are already cached
        # Cache conversation history too (all but last 3 messages after examples)
        if self.examples_injected:
            examples = self.messages[:2]  # Already have cache control
            conversation = self.messages[2:]
            
            if len(conversation) <= 3:
                # Not enough conversation to cache
                return self.messages
            
            # Split conversation into cacheable and recent
            cacheable_conversation = conversation[:-3]
            recent_conversation = conversation[-3:]
            
            formatted = examples.copy()
            
            # Add cacheable conversation messages
            for i, msg in enumerate(cacheable_conversation):
                formatted_msg = {"role": msg["role"], "content": msg["content"]}
                
                # Add cache control to LAST message in cacheable block
                if i == len(cacheable_conversation) - 1:
                    formatted_msg["content"] = [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                
                formatted.append(formatted_msg)
            
            # Add recent messages without caching
            formatted.extend(recent_conversation)
            
            return formatted
        else:
            # No examples injected, cache older messages only
            messages_to_cache = self.messages[:-3]
            recent_messages = self.messages[-3:]
            
            formatted = []
            
            for i, msg in enumerate(messages_to_cache):
                formatted_msg = {"role": msg["role"], "content": msg["content"]}
                
                if i == len(messages_to_cache) - 1:
                    formatted_msg["content"] = [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                
                formatted.append(formatted_msg)
            
            formatted.extend(recent_messages)
            
            return formatted
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.examples_injected = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_context_string(context: Optional[ChatContext]) -> str:
    """Build structured context string from provided files and workspace"""
    if not context:
        return ""
    
    context_parts = []
    
    # Add open files with XML structure
    if context.open_files:
        context_parts.append("<open_files>")
        for file in context.open_files:
            context_parts.append(f"<file path='{file.path}'>")
            context_parts.append(file.content)
            context_parts.append("</file>")
        context_parts.append("</open_files>")
    
    # Add workspace tree with XML structure
    if context.workspace_tree:
        context_parts.append("<workspace_structure>")
        context_parts.append(f"<root>{context.workspace_tree.root}</root>")
        context_parts.append(json.dumps(context.workspace_tree.dict(), indent=2))
        context_parts.append("</workspace_structure>")
    
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
    
    has_modification_keyword = any(keyword in query_lower for keyword in modification_keywords)
    has_reference = any(keyword in query_lower for keyword in reference_keywords)
    
    return (has_modification_keyword or has_reference) and (has_context or has_previous_code)

###### Place your correct model_id in the function call below ######
def call_claude_bedrock(messages: List[Dict], system_prompt: str, model_id: str = "arn:aws:bedrock:us-east-1:52986:inference-profile/us.anthropic.claude-sonnet-4-20250514-v1:0") -> str:
    """
    Call Claude via AWS Bedrock with conversation history and prompt caching
    
    Args:
        messages: Full conversation history with cache control
        system_prompt: System-level role definition (will be cached)
    """
    try:
        # Cache system prompt
        system_blocks = [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"}
            }
        ]
        
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "system": system_blocks,
            "messages": messages,  # Includes cached examples and conversation
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
    """Sort modifications by file, then by descending line number"""
    file_modifications = defaultdict(list)
    
    for change in changes:
        file_path = change.get('file', '')
        modifications = change.get('modifications', [])
        
        for mod in modifications:
            file_modifications[file_path].append(mod)
    
    sorted_changes = []
    for file_path, mods in file_modifications.items():
        sorted_mods = sorted(mods, key=lambda x: x.get('start_line', 0), reverse=True)
        sorted_changes.append({
            'file': file_path,
            'modifications': sorted_mods
        })
    
    return sorted_changes


# ============================================================================
# IN-MEMORY STORAGE
# ============================================================================

conversations = {}
generated_code = {}
active_connections: Dict[str, WebSocket] = {}


# ============================================================================
# MAIN REQUEST HANDLER
# ============================================================================

async def process_chat_request(request: ChatRequest) -> ChatResponse:
    """Process chat request with full conversation history and caching"""
    session_id = request.session_id
    query = request.query
    context = request.context
    
    # Initialize session
    if session_id not in conversations:
        conversations[session_id] = ConversationBuffer(max_messages=20)
        generated_code[session_id] = None
    
    conv_buffer = conversations[session_id]
    has_previous_code = generated_code[session_id] is not None
    has_context = context is not None and (context.open_files or context.workspace_tree)
    
    # Inject examples ONCE at start of conversation
    if not conv_buffer.examples_injected:
        conv_buffer.inject_examples()
    
    # Build context string with XML structure
    context_string = build_context_string(context)
    
    # Determine if modification or generation
    is_modification = is_modification_request(query, has_context, has_previous_code)
    
    # Use SHORT system prompt (no examples - they're in conversation history)
    system_prompt = MAIN_SYSTEM_PROMPT
    
    # Build current user message (NO EXAMPLES - already cached in history)
    current_message_parts = []
    
    # Add context if available
    if context_string:
        current_message_parts.append(f"<workspace_context>\n{context_string}\n</workspace_context>\n")
    elif has_previous_code and is_modification and not has_context:
        current_message_parts.append(f"<previous_code>\n{generated_code[session_id]}\n</previous_code>\n")
    
    # Add user query with XML structure
    current_message_parts.append(f"<user_request>\n{query}\n</user_request>")
    
    current_message = "\n".join(current_message_parts)
    
    # Add user message to conversation history
    conv_buffer.add_message("user", current_message)
    
    # Get full conversation history (includes cached examples)
    messages = conv_buffer.get_messages_for_api()
    
    # Call Claude with full history
    response = call_claude_bedrock(messages, system_prompt)
    
    # Add assistant response to history
    conv_buffer.add_message("assistant", response)
    
    # Parse JSON response
    try:
        # Look for JSON in response (skip reasoning text)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Store generated code if generation
            if parsed.get('type') == 'code_generation':
                generated_code[session_id] = json.dumps(parsed, indent=2)
            
            # Sort modifications if modification
            if parsed.get('type') == 'code_changes' and 'changes' in parsed:
                parsed['changes'] = sort_and_apply_modifications(parsed['changes'])
            
            return ChatResponse(
                response=response,
                parsed=parsed,
                session_id=session_id,
                is_code_change=True,
                request_type="modification" if is_modification else "generation"
            )
        else:
            raise ValueError("No JSON found in response")
    
    except Exception as e:
        return ChatResponse(
            response=f"Error parsing response: {str(e)}\n\nRaw response: {response}",
            parsed=None,
            session_id=session_id,
            is_code_change=False,
            request_type="modification" if is_modification else "generation"
        )


# ============================================================================
# REST API ENDPOINTS
# ============================================================================

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
    
    conv_buffer = conversations[session_id]
    
    return {
        "messages": conv_buffer.messages,
        "session_id": session_id,
        "has_code": generated_code.get(session_id) is not None,
        "examples_cached": conv_buffer.examples_injected
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


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

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
        "message": "AI Code Assistant API v2.0",
        "version": "2.0.0",
        "features": [
            "Conversation history with context",
            "Prompt caching for cost optimization",
            "Unified generation and modification examples",
            "XML-structured context handling"
        ],
        "docs": "/docs",
        "websocket": "/ws/{session_id}"
    }


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("=" * 60)
        print("WARNING: AWS credentials not set!")
        print("Required environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_REGION (optional, defaults to us-east-1)")
        print("=" * 60)
    else:
        print("=" * 60)
        print("AI Code Assistant API v2.0")
        print("=" * 60)
        print("✓ AWS credentials configured")
        print("✓ Prompt caching enabled")
        print("✓ Conversation history enabled")
        print("✓ Starting server on http://0.0.0.0:5000")
        print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=5000)
"""
Main FastAPI Application - Routes to Claude or OpenAI based on model
"""

import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import dotenv

# Import backend clients
from claude import ClaudeClient, ClaudeConversationBuffer, CLAUDE_SYSTEM_PROMPT, sort_modifications, remove_old_content
from openai_backend import OpenAIClient, OpenAIConversationBuffer, OPENAI_SYSTEM_PROMPT, sort_modifications as openai_sort, remove_old_content as openai_remove

# Load environment variables
dotenv.load_dotenv()

# FastAPI app
app = FastAPI(
    title="React Code Assistant API - Multi-Provider",
    description="AI-powered React code generation using Claude (AWS Bedrock) or OpenAI",
    version="3.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

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
    model_name: Optional[str] = None

class TokenUsage(BaseModel):
    """Unified token usage for both providers"""
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = 0  # Claude
    cache_read_input_tokens: Optional[int] = 0      # Claude
    cached_tokens: Optional[int] = 0                 # OpenAI

class ChatResponse(BaseModel):
    type: str
    parsed: Dict[str, Any]
    session_id: str
    is_code_change: bool
    request_type: str
    workspace_tree: Optional[Dict[str, Any]] = None
    usage: Optional[TokenUsage] = None
    model_name: Optional[str] = None
    provider: Optional[str] = None  # "claude" or "openai"

class ResetRequest(BaseModel):
    session_id: str = "default"

# ============================================================================
# INITIALIZE CLIENTS
# ============================================================================

claude_client = ClaudeClient()
openai_client = OpenAIClient()

# ============================================================================
# IN-MEMORY STORAGE
# ============================================================================

# Separate conversation buffers for each provider per session
conversations = {
    "claude": {},  # session_id -> ClaudeConversationBuffer
    "openai": {}   # session_id -> OpenAIConversationBuffer
}
generated_code = {}  # session_id -> generated code
active_connections: Dict[str, WebSocket] = {}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def determine_provider(model_name: str) -> str:
    """Determine which provider to use based on model name"""
    if claude_client.is_claude_model(model_name):
        return "claude"
    elif openai_client.is_openai_model(model_name):
        return "openai"
    else:
        # Default to claude if unknown
        return "claude"

def build_context_string(context: Optional[ChatContext]) -> str:
    """Build XML-structured context string"""
    if not context:
        return ""
    
    context_parts = []
    
    if context.open_files:
        context_parts.append("<open_files>")
        for file in context.open_files:
            context_parts.append(f"<file path='{file.path}'>")
            context_parts.append(file.content)
            context_parts.append("</file>")
        context_parts.append("</open_files>")
    
    if context.workspace_tree:
        context_parts.append("<workspace_structure>")
        context_parts.append(f"<root>{context.workspace_tree.root}</root>")
        context_parts.append(json.dumps(context.workspace_tree.dict(), indent=2))
        context_parts.append("</workspace_structure>")
    
    return "\n".join(context_parts)

def is_modification_request(query: str, has_context: bool, has_previous_code: bool) -> bool:
    """Determine if request is for modification vs generation"""
    modification_keywords = [
        'change', 'modify', 'update', 'fix', 'add', 'remove', 'delete',
        'edit', 'refactor', 'improve', 'adjust', 'alter', 'correct',
        'replace', 'swap', 'rename', 'move', 'convert'
    ]
    
    reference_keywords = [
        'the code', 'above', 'previous', 'existing', 'current',
        'this code', 'that function', 'the function', 'this component'
    ]
    
    query_lower = query.lower()
    
    has_modification_keyword = any(keyword in query_lower for keyword in modification_keywords)
    has_reference = any(keyword in query_lower for keyword in reference_keywords)
    
    return (has_modification_keyword or has_reference) and (has_context or has_previous_code)

def is_likely_code_request(query: str) -> bool:
    """Check if query is asking for code"""
    code_keywords = [
        'create', 'generate', 'build', 'make', 'add', 'modify',
        'change', 'update', 'fix', 'remove', 'delete', 'refactor',
        'component', 'hook', 'function', 'app', 'page', 'form'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in code_keywords)

# ============================================================================
# MAIN REQUEST HANDLER
# ============================================================================

async def process_chat_request(request: ChatRequest) -> ChatResponse:
    """Process chat request - routes to Claude or OpenAI based on model"""
    
    session_id = request.session_id
    query = request.query
    context = request.context
    model_name = request.model_name or "claude-sonnet-4-5"  # Default
    
    # Determine provider
    provider = determine_provider(model_name)
    
    # Initialize session for provider if needed
    if session_id not in conversations[provider]:
        if provider == "claude":
            conversations[provider][session_id] = ClaudeConversationBuffer()
        else:
            conversations[provider][session_id] = OpenAIConversationBuffer()
        generated_code[session_id] = None
    
    conv_buffer = conversations[provider][session_id]
    has_previous_code = generated_code.get(session_id) is not None
    has_context = context is not None and (context.open_files or context.workspace_tree)
    
    # Inject examples once
    if not conv_buffer.examples_injected:
        conv_buffer.inject_examples()
    
    # Build context string
    context_string = build_context_string(context)
    
    # Determine if modification
    is_modification = is_modification_request(query, has_context, has_previous_code)
    
    # Select system prompt based on provider
    system_prompt = CLAUDE_SYSTEM_PROMPT if provider == "claude" else OPENAI_SYSTEM_PROMPT
    
    # Build current message
    current_message_parts = []
    
    if context_string:
        current_message_parts.append(f"<workspace_context>\n{context_string}\n</workspace_context>\n")
    elif has_previous_code and is_modification and not has_context:
        current_message_parts.append(f"<previous_code>\n{generated_code[session_id]}\n</previous_code>\n")
    
    current_message_parts.append(f"<user_request>\n{query}\n</user_request>")
    
    current_message = "\n".join(current_message_parts)
    
    # Add to conversation history
    conv_buffer.add_message("user", current_message)
    
    # Get messages for API
    messages = conv_buffer.get_messages_for_api()
    
    # Call appropriate API
    try:
        if provider == "claude":
            response, usage = claude_client.call_api(messages, system_prompt, model_name)
        else:
            response, usage = openai_client.call_api(messages, system_prompt, model_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"API Error: {str(e)}")
    
    # Add response to history
    conv_buffer.add_message("assistant", response)
    
    # Prepare token usage (normalize between providers)
    if provider == "claude":
        token_usage = TokenUsage(
            input_tokens=usage.get('input_tokens', 0),
            output_tokens=usage.get('output_tokens', 0),
            cache_creation_input_tokens=usage.get('cache_creation_input_tokens', 0),
            cache_read_input_tokens=usage.get('cache_read_input_tokens', 0),
            cached_tokens=0
        )
    else:  # openai
        token_usage = TokenUsage(
            input_tokens=usage.get('prompt_tokens', 0),
            output_tokens=usage.get('completion_tokens', 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            cached_tokens=usage.get('cached_tokens', 0)
        )
    
    # Parse JSON response
    try:
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Remove old_content
            if provider == "claude":
                parsed = remove_old_content(parsed)
            else:
                parsed = openai_remove(parsed)
            
            # Store generated code
            if parsed.get('type') == 'code_generation':
                generated_code[session_id] = json.dumps(parsed, indent=2)
            
            # Sort modifications
            if parsed.get('type') == 'code_changes' and 'changes' in parsed:
                if provider == "claude":
                    parsed['changes'] = sort_modifications(parsed['changes'])
                else:
                    parsed['changes'] = openai_sort(parsed['changes'])
            
            response_type = parsed.get('type', 'code_generation')
            is_code = response_type in ['code_generation', 'code_changes']
            request_type = "modification" if response_type == 'code_changes' else "generation"
            
            return ChatResponse(
                type=response_type,
                parsed=parsed,
                session_id=session_id,
                is_code_change=is_code,
                request_type=request_type,
                workspace_tree=context.workspace_tree.dict() if context and context.workspace_tree else None,
                usage=token_usage,
                model_name=model_name,
                provider=provider
            )
        else:
            raise ValueError("No JSON found")
    
    except Exception as e:
        # Conversational or error response
        if not is_likely_code_request(query):
            return ChatResponse(
                type="conversation",
                parsed={
                    "type": "conversation",
                    "summary": response.strip()
                },
                session_id=session_id,
                is_code_change=False,
                request_type="conversation",
                workspace_tree=context.workspace_tree.dict() if context and context.workspace_tree else None,
                usage=token_usage,
                model_name=model_name,
                provider=provider
            )
        else:
            return ChatResponse(
                type="error",
                parsed={
                    "type": "error",
                    "error": f"Expected code but received unexpected response: {str(e)}",
                    "summary": response.strip()
                },
                session_id=session_id,
                is_code_change=False,
                request_type="error",
                workspace_tree=context.workspace_tree.dict() if context and context.workspace_tree else None,
                usage=token_usage,
                model_name=model_name,
                provider=provider
            )

# ============================================================================
# REST API ENDPOINTS
# ============================================================================

@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_endpoint(
    query: str = Form(...),
    session_id: str = Form(default="default"),
    model_name: Optional[str] = Form(default=None),
    workspace_tree: Optional[str] = Form(default=None),
    files: List[UploadFile] = File(default=[])
):
    """
    Process chat request with file uploads.
    
    Automatically routes to Claude or OpenAI based on model_name.
    
    Supported models:
    - Claude: claude-3-5-sonnet, claude-3-7-sonnet, claude-sonnet-4, claude-sonnet-4-5
    - OpenAI: gpt-4o, gpt-4o-mini, o1, o1-mini
    """
    try:
        # Read uploaded files
        file_contexts = []
        for file in files:
            try:
                content = await file.read()
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    print(f"Warning: Skipping binary file {file.filename}")
                    continue
                
                file_contexts.append(FileContext(
                    path=file.filename,
                    content=text_content
                ))
            except Exception as e:
                print(f"Error reading file {file.filename}: {str(e)}")
                continue
        
        # Parse workspace tree
        ws_tree = None
        if workspace_tree:
            try:
                ws_tree = WorkspaceTree(**json.loads(workspace_tree))
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid workspace_tree JSON")
        
        # Build context
        context = None
        if file_contexts or ws_tree:
            context = ChatContext(
                open_files=file_contexts if file_contexts else None,
                workspace_tree=ws_tree
            )
        
        # Create request
        request = ChatRequest(
            query=query,
            context=context,
            session_id=session_id,
            model_name=model_name
        )
        
        return await process_chat_request(request)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reset", tags=["Session"])
async def reset_session(request: ResetRequest):
    """Reset conversation history for a session (both providers)"""
    session_id = request.session_id
    
    # Clear from both providers
    for provider in ["claude", "openai"]:
        if session_id in conversations[provider]:
            del conversations[provider][session_id]
    
    if session_id in generated_code:
        del generated_code[session_id]
    
    return {"message": f"Session {session_id} reset successfully"}

@app.get("/history/{session_id}", tags=["Session"])
async def get_history(session_id: str, provider: str = "claude"):
    """Get conversation history for a session"""
    if session_id not in conversations[provider]:
        return {"messages": [], "has_code": False, "provider": provider}
    
    conv_buffer = conversations[provider][session_id]
    
    return {
        "messages": conv_buffer.messages,
        "session_id": session_id,
        "has_code": generated_code.get(session_id) is not None,
        "examples_cached": conv_buffer.examples_injected,
        "provider": provider
    }

@app.get("/code/{session_id}", tags=["Session"])
async def get_code(session_id: str):
    """Get generated code for a session"""
    if session_id not in generated_code or generated_code[session_id] is None:
        return {"code": None, "message": "No code generated yet"}
    
    return {
        "code": generated_code[session_id],
        "session_id": session_id
    }

@app.get("/models", tags=["System"])
async def get_available_models():
    """Get all available models from both providers"""
    return {
        "claude": list(claude_client.model_mapping.keys()),
        "openai": list(openai_client.model_mapping.keys()),
        "default_claude": claude_client.default_model,
        "default_openai": openai_client.default_model
    }

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": {
            "claude": len(conversations["claude"]),
            "openai": len(conversations["openai"])
        },
        "active_websockets": len(active_connections),
        "providers": ["claude", "openai"]
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    active_connections[session_id] = websocket
    
    try:
        while True:
            data = await websocket.receive_text()
            request_data = json.loads(data)
            
            request = ChatRequest(
                query=request_data.get('query', ''),
                context=ChatContext(**request_data.get('context', {})) if request_data.get('context') else None,
                session_id=session_id,
                model_name=request_data.get('model_name')
            )
            
            response = await process_chat_request(request)
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
        "message": "React Code Assistant API v3.0 - Multi-Provider",
        "version": "3.0.0",
        "providers": {
            "claude": {
                "models": list(claude_client.model_mapping.keys()),
                "optimization": "XML-structured prompts with explicit caching"
            },
            "openai": {
                "models": list(openai_client.model_mapping.keys()),
                "optimization": "Markdown-structured prompts with automatic caching"
            }
        },
        "features": [
            "Multi-provider support (Claude + OpenAI)",
            "Automatic provider detection from model name",
            "Provider-specific prompt optimization",
            "Prompt caching (explicit for Claude, automatic for OpenAI)",
            "Conversation history with context",
            "React-focused code generation and modification",
            "Token usage tracking with cache metrics",
            "No old_content in modifications"
        ],
        "docs": "/docs",
        "websocket": "/ws/{session_id}"
    }

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("React Code Assistant API v3.0 - Multi-Provider")
    print("=" * 70)
    
    # Check credentials
    has_aws = os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')
    has_openai = os.getenv('OPENAI_API_KEY')
    
    print(f"✅ Claude (AWS Bedrock): {'Configured' if has_aws else '❌ Not configured'}")
    print(f"✅ OpenAI: {'Configured' if has_openai else '❌ Not configured'}")
    
    if has_aws:
        print(f"\nClaude Models: {list(claude_client.model_mapping.keys())}")
    
    if has_openai:
        print(f"OpenAI Models: {list(openai_client.model_mapping.keys())}")
    
    print("\n🎯 Provider routing: Automatic based on model name")
    print("✅ Claude: XML-optimized prompts")
    print("✅ OpenAI: Markdown-optimized prompts")
    print("\n🚀 Starting server on http://0.0.0.0:5000")
    print("📚 API docs: http://0.0.0.0:5000/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=5000)
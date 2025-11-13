import os
import json
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from openai import OpenAI
import asyncio
from collections import defaultdict
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# FastAPI app
app = FastAPI(
    title="React Code Assistant API - OpenAI",
    description="AI-powered React code generation and modification assistant using OpenAI GPT-4.1",
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
# OPENAI CONFIGURATION
# ============================================================================

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Model ID mapping - OpenAI model names
MODEL_MAPPING = {
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gpt-4o": "gpt-4o-2024-11-20",
    "gpt-4.1-mini": "gpt-4.1-mini-2025-04-14",
    "o3": "o3-2025-01-31",
    "o4-mini": "o4-mini-2025-01-31"
}

DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'gpt-4.1')

def get_model_id(model_name: Optional[str] = None) -> str:
    """
    Convert friendly model name to OpenAI model ID
    
    Args:
        model_name: Friendly name like "gpt-4.1"
    
    Returns:
        Full OpenAI model ID
    """
    name = model_name or DEFAULT_MODEL
    return MODEL_MAPPING.get(name, MODEL_MAPPING[DEFAULT_MODEL])

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

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
    model_name: Optional[str] = None  # e.g., "gpt-4.1"

class TokenUsage(BaseModel):
    """Token usage information from OpenAI API"""
    input_tokens: int
    output_tokens: int
    cached_tokens: Optional[int] = 0

class ChatResponse(BaseModel):
    """
    Unified response structure for all request types
    
    Response types:
    - "code_generation": New code created
    - "code_modification": Existing code modified
    - "conversation": Chat response (no code)
    - "error": Error occurred
    """
    type: str  # "code_generation", "code_modification", "conversation", "error"
    parsed: Dict[str, Any]
    session_id: str
    is_code_change: bool
    request_type: str
    workspace_tree: Optional[Dict[str, Any]] = None
    usage: Optional[TokenUsage] = None
    model_name: Optional[str] = None

class ResetRequest(BaseModel):
    session_id: str = "default"

# ============================================================================
# PROMPTS - REACT-FOCUSED WITH OPENAI OPTIMIZATION
# ============================================================================

# System Prompt - More explicit for GPT-4.1's literal instruction following
MAIN_SYSTEM_PROMPT = """You are an expert React developer assistant specializing in modern React development with hooks, TypeScript, and best practices.

<react_expertise>
YOU MUST USE these React patterns and technologies:
- Modern React with functional components and hooks (useState, useEffect, useContext, useReducer, useMemo, useCallback)
- TypeScript for type safety and better developer experience
- Component composition and prop drilling avoidance
- Custom hooks for reusable logic
- State management: Context API, Redux, Zustand
- Routing: React Router v6
- Styling: CSS Modules, Styled Components, Tailwind CSS
- API integration: fetch, axios, React Query, SWR
- Form handling: React Hook Form, Formik
- Performance optimization: memo, lazy loading, code splitting
- Testing: Jest, React Testing Library
- Build tools: Vite, Create React App, Next.js
</react_expertise>

<core_responsibilities>
YOU MUST:
1. Generate well-structured, production-ready React components
2. Modify existing React code accurately using line-based changes
3. Follow React best practices and modern patterns
4. Provide TypeScript types when appropriate
5. Respond naturally to casual conversation
6. Maintain context throughout conversations
7. Be EXTREMELY PRECISE with line numbers and modifications
8. ALWAYS output valid JSON for code requests (no markdown, no code blocks)
9. Include brief analysis before JSON output
</core_responsibilities>

<interaction_style>
FOLLOW THESE RULES EXACTLY:
- For code requests: Provide structured JSON responses as instructed in examples
- For casual chat: Respond conversationally without JSON
- Ask clarifying questions when requirements are ambiguous
- Be precise and detail-oriented in code generation
- NEVER add markdown code blocks around JSON output
- ALWAYS escape special characters in JSON strings properly
</interaction_style>

<agentic_behavior>
YOU ARE AN AGENT - This means:
1. PERSISTENCE: Keep working until the user's query is completely resolved before ending your turn
2. TOOL AWARENESS: If you need information about files or code structure, ask the user (do NOT guess or make assumptions)
3. PLANNING: For complex tasks, write a brief plan first, then execute
4. REFLECTION: After generating code or modifications, verify they address all requirements
5. STEP-BY-STEP: For multi-step tasks, break them down and explain your approach
</agentic_behavior>

You will receive task-specific instructions and examples in the conversation. Follow them EXACTLY as written."""


# React Code Generation Examples and Instructions
REACT_GENERATION_EXAMPLES = """<mode>REACT CODE GENERATION</mode>

<instructions>
You are now in React code generation mode. FOLLOW THIS PROCESS EXACTLY:

1. ANALYZE: Understand the React requirements and plan your component structure
2. STRUCTURE: Determine the file organization (components, hooks, utils, styles)
3. GENERATE: Create complete React code in JSON format

CRITICAL OUTPUT FORMAT - FOLLOW EXACTLY:
- Start with brief analysis (2-3 sentences explaining your approach)
- Then output ONLY valid JSON (NO markdown, NO code blocks, NO extra text)
- Format: [Brief reasoning] + JSON structure
- DO NOT wrap JSON in ```json or ``` - output raw JSON only

VALIDATION CHECKLIST:
□ Brief analysis comes first (2-3 sentences)
□ JSON starts immediately after analysis
□ No markdown code blocks
□ All quotes properly escaped
□ All newlines as \\n
□ Valid JSON structure
</instructions>

<json_structure>
{
  "type": "code_generation",
  "changes": [
    {
      "file": "src/components/ComponentName.jsx",
      "content": "complete file content as a string with escaped newlines"
    }
  ],
  "summary": "Brief description of what was generated"
}
</json_structure>

<react_best_practices>
FOLLOW THESE RULES EXACTLY:
1. Use functional components with hooks (NEVER use class components)
2. Destructure props for cleaner code
3. Use proper TypeScript types when applicable (.tsx extension)
4. Follow naming conventions: PascalCase for components, camelCase for functions
5. Keep components focused and single-responsibility
6. Extract reusable logic into custom hooks
7. Use proper key props in lists
8. Handle loading and error states
9. Add PropTypes or TypeScript interfaces
10. Include necessary imports (React, hooks, libraries)
</react_best_practices>

<file_structure_patterns>
USE THESE FILE PATHS:
- Components: src/components/ComponentName.jsx or .tsx
- Hooks: src/hooks/useHookName.js
- Utils: src/utils/utilName.js
- Styles: src/components/ComponentName.module.css or ComponentName.styled.js
- Types: src/types/types.ts
- Constants: src/constants/constants.js
- API: src/api/apiName.js
- Context: src/context/ContextName.jsx
</file_structure_patterns>

<examples>
<example name="functional_component_with_state">
<scenario>User: "Create a Counter component with increment and decrement buttons"</scenario>
<response>
I'll create a simple Counter component using useState hook with increment and decrement functionality.

{
  "type": "code_generation",
  "changes": [
    {
      "file": "src/components/Counter.jsx",
      "content": "import React, { useState } from 'react';\\nimport './Counter.css';\\n\\nconst Counter = () => {\\n  const [count, setCount] = useState(0);\\n\\n  const increment = () => setCount(prev => prev + 1);\\n  const decrement = () => setCount(prev => prev - 1);\\n  const reset = () => setCount(0);\\n\\n  return (\\n    <div className=\\"counter-container\\">\\n      <h2>Counter</h2>\\n      <div className=\\"count-display\\">{count}</div>\\n      <div className=\\"button-group\\">\\n        <button onClick={decrement} className=\\"btn btn-decrement\\">-</button>\\n        <button onClick={reset} className=\\"btn btn-reset\\">Reset</button>\\n        <button onClick={increment} className=\\"btn btn-increment\\">+</button>\\n      </div>\\n    </div>\\n  );\\n};\\n\\nexport default Counter;"
    },
    {
      "file": "src/components/Counter.css",
      "content": ".counter-container {\\n  display: flex;\\n  flex-direction: column;\\n  align-items: center;\\n  padding: 2rem;\\n  border-radius: 10px;\\n  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\\n  color: white;\\n  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);\\n}\\n\\n.counter-container h2 {\\n  margin: 0 0 1rem 0;\\n  font-size: 1.5rem;\\n}\\n\\n.count-display {\\n  font-size: 3rem;\\n  font-weight: bold;\\n  margin: 1rem 0;\\n}\\n\\n.button-group {\\n  display: flex;\\n  gap: 1rem;\\n}\\n\\n.btn {\\n  padding: 0.75rem 1.5rem;\\n  font-size: 1.2rem;\\n  border: none;\\n  border-radius: 5px;\\n  cursor: pointer;\\n  transition: transform 0.2s, box-shadow 0.2s;\\n  font-weight: 600;\\n}\\n\\n.btn:hover {\\n  transform: translateY(-2px);\\n  box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);\\n}\\n\\n.btn-decrement {\\n  background-color: #e74c3c;\\n  color: white;\\n}\\n\\n.btn-reset {\\n  background-color: #95a5a6;\\n  color: white;\\n}\\n\\n.btn-increment {\\n  background-color: #27ae60;\\n  color: white;\\n}"
    }
  ],
  "summary": "Created Counter component with useState hook, increment/decrement/reset functionality, and styled UI"
}
</response>
</example>

<example name="form_with_validation">
<scenario>User: "Create a login form with email and password validation"</scenario>
<response>
I'll create a login form using useState for form state and validation logic, with proper error handling.

{
  "type": "code_generation",
  "changes": [
    {
      "file": "src/components/LoginForm.jsx",
      "content": "import React, { useState } from 'react';\\nimport './LoginForm.css';\\n\\nconst LoginForm = ({ onSubmit }) => {\\n  const [formData, setFormData] = useState({\\n    email: '',\\n    password: ''\\n  });\\n  const [errors, setErrors] = useState({});\\n  const [isSubmitting, setIsSubmitting] = useState(false);\\n\\n  const validateEmail = (email) => {\\n    const regex = /^[^\\\\s@]+@[^\\\\s@]+\\\\.[^\\\\s@]+$/;\\n    return regex.test(email);\\n  };\\n\\n  const validateForm = () => {\\n    const newErrors = {};\\n\\n    if (!formData.email) {\\n      newErrors.email = 'Email is required';\\n    } else if (!validateEmail(formData.email)) {\\n      newErrors.email = 'Invalid email format';\\n    }\\n\\n    if (!formData.password) {\\n      newErrors.password = 'Password is required';\\n    } else if (formData.password.length < 6) {\\n      newErrors.password = 'Password must be at least 6 characters';\\n    }\\n\\n    setErrors(newErrors);\\n    return Object.keys(newErrors).length === 0;\\n  };\\n\\n  const handleChange = (e) => {\\n    const { name, value } = e.target;\\n    setFormData(prev => ({ ...prev, [name]: value }));\\n    \\n    if (errors[name]) {\\n      setErrors(prev => ({ ...prev, [name]: '' }));\\n    }\\n  };\\n\\n  const handleSubmit = async (e) => {\\n    e.preventDefault();\\n    \\n    if (!validateForm()) return;\\n\\n    setIsSubmitting(true);\\n    try {\\n      await onSubmit(formData);\\n    } catch (error) {\\n      setErrors({ submit: error.message || 'Login failed' });\\n    } finally {\\n      setIsSubmitting(false);\\n    }\\n  };\\n\\n  return (\\n    <div className=\\"login-container\\">\\n      <form className=\\"login-form\\" onSubmit={handleSubmit}>\\n        <h2>Login</h2>\\n        \\n        <div className=\\"form-group\\">\\n          <label htmlFor=\\"email\\">Email</label>\\n          <input\\n            type=\\"email\\"\\n            id=\\"email\\"\\n            name=\\"email\\"\\n            value={formData.email}\\n            onChange={handleChange}\\n            className={errors.email ? 'error' : ''}\\n            disabled={isSubmitting}\\n          />\\n          {errors.email && <span className=\\"error-message\\">{errors.email}</span>}\\n        </div>\\n\\n        <div className=\\"form-group\\">\\n          <label htmlFor=\\"password\\">Password</label>\\n          <input\\n            type=\\"password\\"\\n            id=\\"password\\"\\n            name=\\"password\\"\\n            value={formData.password}\\n            onChange={handleChange}\\n            className={errors.password ? 'error' : ''}\\n            disabled={isSubmitting}\\n          />\\n          {errors.password && <span className=\\"error-message\\">{errors.password}</span>}\\n        </div>\\n\\n        {errors.submit && (\\n          <div className=\\"error-message submit-error\\">{errors.submit}</div>\\n        )}\\n\\n        <button type=\\"submit\\" className=\\"submit-btn\\" disabled={isSubmitting}>\\n          {isSubmitting ? 'Logging in...' : 'Login'}\\n        </button>\\n      </form>\\n    </div>\\n  );\\n};\\n\\nexport default LoginForm;"
    },
    {
      "file": "src/components/LoginForm.css",
      "content": ".login-container {\\n  display: flex;\\n  justify-content: center;\\n  align-items: center;\\n  min-height: 100vh;\\n  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\\n}\\n\\n.login-form {\\n  background: white;\\n  padding: 2rem;\\n  border-radius: 10px;\\n  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);\\n  width: 100%;\\n  max-width: 400px;\\n}\\n\\n.login-form h2 {\\n  margin: 0 0 1.5rem 0;\\n  text-align: center;\\n  color: #333;\\n}\\n\\n.form-group {\\n  margin-bottom: 1.5rem;\\n}\\n\\n.form-group label {\\n  display: block;\\n  margin-bottom: 0.5rem;\\n  color: #555;\\n  font-weight: 500;\\n}\\n\\n.form-group input {\\n  width: 100%;\\n  padding: 0.75rem;\\n  border: 2px solid #e0e0e0;\\n  border-radius: 5px;\\n  font-size: 1rem;\\n  transition: border-color 0.3s;\\n}\\n\\n.form-group input:focus {\\n  outline: none;\\n  border-color: #667eea;\\n}\\n\\n.form-group input.error {\\n  border-color: #e74c3c;\\n}\\n\\n.error-message {\\n  display: block;\\n  color: #e74c3c;\\n  font-size: 0.875rem;\\n  margin-top: 0.25rem;\\n}\\n\\n.submit-error {\\n  text-align: center;\\n  padding: 0.75rem;\\n  background-color: #fee;\\n  border-radius: 5px;\\n  margin-bottom: 1rem;\\n}\\n\\n.submit-btn {\\n  width: 100%;\\n  padding: 0.75rem;\\n  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);\\n  color: white;\\n  border: none;\\n  border-radius: 5px;\\n  font-size: 1rem;\\n  font-weight: 600;\\n  cursor: pointer;\\n  transition: transform 0.2s;\\n}\\n\\n.submit-btn:hover:not(:disabled) {\\n  transform: translateY(-2px);\\n}\\n\\n.submit-btn:disabled {\\n  opacity: 0.6;\\n  cursor: not-allowed;\\n}"
    }
  ],
  "summary": "Created login form with email/password validation, error handling, loading states, and responsive styling"
}
</response>
</example>
</examples>

<critical_reminders>
REMEMBER - FOLLOW EXACTLY:
- Output brief analysis (2-3 sentences) then ONLY the JSON structure
- No markdown code blocks (```)
- Use proper React patterns and hooks
- Include all necessary imports
- Ensure components are functional (not class-based)
- Add proper error handling and loading states
- Follow React naming conventions
- Properly escape all special characters in content strings
- Ensure JSON is valid and parseable
</critical_reminders>"""


# React Code Modification Examples and Instructions
REACT_MODIFICATION_EXAMPLES = """<mode>REACT CODE MODIFICATION</mode>

<instructions>
You are now in React code modification mode. FOLLOW THIS PROCESS EXACTLY:

1. ANALYZE: Understand what React code needs to be changed and why
2. LOCATE: Identify EXACT line numbers and content to modify
3. MODIFY: Provide precise line-based changes in JSON format

CRITICAL OUTPUT FORMAT - FOLLOW EXACTLY:
- Start with brief analysis (2-3 sentences explaining your changes)
- Then output ONLY valid JSON (NO markdown, NO code blocks, NO extra text)
- Format: [Brief reasoning] + JSON structure
- NOTE: Do NOT include "old_content" field - only provide line numbers and new content
- BE EXTREMELY PRECISE with line numbers - they are 1-indexed (first line is 1)

VALIDATION CHECKLIST:
□ Brief analysis comes first (2-3 sentences)
□ JSON starts immediately after analysis
□ No markdown code blocks
□ Line numbers are EXACT and 1-indexed
□ All quotes properly escaped
□ All newlines as \\n
□ Valid JSON structure
□ NO "old_content" field
</instructions>

<json_structure>
{
  "type": "code_changes",
  "changes": [
    {
      "file": "path/to/Component.jsx",
      "modifications": [
        {
          "operation": "replace" | "insert" | "delete" | "insert_before",
          "start_line": <number>,
          "end_line": <number>,
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
Purpose: Change existing lines in the React component
Required fields:
- start_line: First line number to replace (1-indexed, MUST BE EXACT)
- end_line: Last line number to replace (inclusive, 1-indexed, MUST BE EXACT)
- new_content: New content to insert (NO old_content field needed)

Use when: Modifying existing code, fixing bugs, updating values, changing props

CRITICAL: Count line numbers EXACTLY - include ALL lines (even blank lines)
</operation>

<operation name="insert">
Purpose: Add new lines AFTER a specified line
Required fields:
- start_line: Line number after which to insert (1-indexed, MUST BE EXACT)
- new_content: Content to insert

Use when: Adding new hooks, props, state, functions
</operation>

<operation name="insert_before">
Purpose: Add new lines BEFORE a specified line
Required fields:
- start_line: Line number before which to insert (1-indexed, MUST BE EXACT)
- new_content: Content to insert

Use when: Adding imports, adding code before existing logic
</operation>

<operation name="delete">
Purpose: Remove lines from the file
Required fields:
- start_line: First line number to delete (1-indexed, MUST BE EXACT)
- end_line: Last line number to delete (inclusive, 1-indexed, MUST BE EXACT)

Use when: Removing unnecessary code, unused imports, deprecated props
</operation>
</operations>

<react_modification_patterns>
FOLLOW THESE PATTERNS:
1. Adding state: Insert useState hook after existing hooks
2. Adding props: Modify component function signature
3. Adding event handlers: Insert new function before return statement
4. Modifying JSX: Replace specific lines in return statement
5. Adding imports: Insert at top of file
6. Updating hooks: Replace hook definition lines
7. Adding effects: Insert useEffect after state declarations
8. Refactoring: Replace multiple lines with cleaner implementation
</react_modification_patterns>

<rules>
FOLLOW THESE RULES EXACTLY:
1. Line numbers are 1-indexed (first line of file is line 1)
2. NO "old_content" field - frontend handles line identification
3. Preserve proper indentation in new_content
4. Order modifications from top to bottom within each file
5. Use \\n to represent line breaks in new_content strings
6. Escape quotes and backslashes properly
7. Ensure JSON is valid and parseable
8. COUNT LINE NUMBERS EXACTLY - include blank lines, comments, everything
</rules>

<examples>
<example>
<scenario>
User: "Add a loading state to the UserList component"

Current file (UserList.jsx):
Line 1: import React, { useState, useEffect } from 'react';
Line 2: 
Line 3: const UserList = () => {
Line 4:   const [users, setUsers] = useState([]);
Line 5:   
Line 6:   useEffect(() => {
Line 7:     fetch('/api/users')
Line 8:       .then(res => res.json())
Line 9:       .then(data => setUsers(data));
Line 10:   }, []);
Line 11:   
Line 12:   return (
Line 13:     <div>
Line 14:       {users.map(user => <div key={user.id}>{user.name}</div>)}
Line 15:     </div>
Line 16:   );
Line 17: };
</scenario>
<response>
I'll add a loading state using useState and display a loading message while data is being fetched.

{
  "type": "code_changes",
  "changes": [
    {
      "file": "UserList.jsx",
      "modifications": [
        {
          "operation": "insert",
          "start_line": 4,
          "new_content": "  const [loading, setLoading] = useState(true);"
        },
        {
          "operation": "replace",
          "start_line": 6,
          "end_line": 10,
          "new_content": "  useEffect(() => {\\n    setLoading(true);\\n    fetch('/api/users')\\n      .then(res => res.json())\\n      .then(data => {\\n        setUsers(data);\\n        setLoading(false);\\n      });\\n  }, []);"
        },
        {
          "operation": "replace",
          "start_line": 12,
          "end_line": 16,
          "new_content": "  if (loading) {\\n    return <div>Loading users...</div>;\\n  }\\n\\n  return (\\n    <div>\\n      {users.map(user => <div key={user.id}>{user.name}</div>)}\\n    </div>\\n  );"
        }
      ]
    }
  ],
  "summary": "Added loading state with useState and conditional rendering for loading indicator"
}
</response>
</example>

<example>
<scenario>
User: "Change the button color prop from 'blue' to 'green' in Button.jsx"

Current file (Button.jsx):
Line 1: import React from 'react';
Line 2: 
Line 3: const Button = ({ onClick, children }) => {
Line 4:   return (
Line 5:     <button 
Line 6:       onClick={onClick}
Line 7:       style={{ backgroundColor: 'blue', color: 'white' }}
Line 8:     >
Line 9:       {children}
Line 10:     </button>
Line 11:   );
Line 12: };
</scenario>
<response>
I'll update the backgroundColor in the style prop from blue to green.

{
  "type": "code_changes",
  "changes": [
    {
      "file": "Button.jsx",
      "modifications": [
        {
          "operation": "replace",
          "start_line": 7,
          "end_line": 7,
          "new_content": "      style={{ backgroundColor: 'green', color: 'white' }}"
        }
      ]
    }
  ],
  "summary": "Changed button background color from blue to green"
}
</response>
</example>
</examples>

<critical_reminders>
REMEMBER - FOLLOW EXACTLY:
- Output brief analysis (2-3 sentences) then ONLY the JSON structure
- NO "old_content" field - not needed for frontend
- No markdown code blocks (```)
- Use \\n for newlines in strings
- Escape quotes and backslashes properly
- Ensure JSON is valid and parseable
- List modifications in top-to-bottom order per file
- Preserve proper React component structure and indentation
- BE EXTREMELY PRECISE with line numbers - count EXACTLY
</critical_reminders>"""


# Unified Examples Conversation - Optimized for OpenAI automatic caching
UNIFIED_EXAMPLES_CONVERSATION = [
    {
        "role": "user",
        "content": f"{REACT_GENERATION_EXAMPLES}\n\n---\n\n{REACT_MODIFICATION_EXAMPLES}"
    },
    {
        "role": "assistant",
        "content": "I understand both React code generation and modification formats EXACTLY. For generation requests, I will create complete React components with proper hooks, state management, and styling in JSON format. For modification requests, I will provide precise line-based changes using the appropriate operations (replace, insert, insert_before, delete) WITHOUT including old_content field. I will ALWAYS start with brief analysis, then output ONLY valid JSON without markdown or extra text. I will be EXTREMELY PRECISE with line numbers and count them EXACTLY including all blank lines."
    }
]


# ============================================================================
# CONVERSATION BUFFER - OPTIMIZED FOR OPENAI AUTOMATIC CACHING
# ============================================================================

class ConversationBuffer:
    """
    Manages conversation history optimized for OpenAI's automatic prompt caching
    
    Caching strategy (automatic by OpenAI):
    1. Static content (examples) at the beginning - gets cached automatically
    2. Conversation history in the middle - cached when >1024 tokens
    3. Recent dynamic content at the end - not cached
    
    OpenAI caches the longest prefix >1024 tokens automatically
    """
    
    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages
        self.examples_injected = False
    
    def inject_examples(self):
        """
        Inject unified React examples at the start of conversation (only once)
        OpenAI will automatically cache this when >1024 tokens
        """
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
        """
        Keep recent messages but PRESERVE examples at start
        Maintains conversation window while keeping cacheable examples
        """
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
        Get messages formatted for OpenAI API
        
        OpenAI's automatic caching will handle optimization:
        - Caches longest prefix >1024 tokens
        - Examples at start get cached automatically
        - Recent messages stay dynamic
        """
        return self.messages
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.examples_injected = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_context_string(context: Optional[ChatContext]) -> str:
    """
    Build structured context string from provided files and workspace
    Uses XML tags for clear semantic boundaries
    """
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
    """
    Determine if request is for modification vs generation
    
    This is a HINT for the model, not a hard rule.
    Model can self-correct if classification is wrong.
    """
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
    
    # It's likely a modification if keywords match AND context/history exists
    return (has_modification_keyword or has_reference) and (has_context or has_previous_code)


def is_likely_code_request(query: str) -> bool:
    """
    Check if query is asking for code generation/modification
    Used to detect if conversational response is appropriate
    """
    code_keywords = [
        'create', 'generate', 'build', 'make', 'add', 'modify',
        'change', 'update', 'fix', 'remove', 'delete', 'refactor',
        'component', 'hook', 'function', 'app', 'page', 'form'
    ]
    
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in code_keywords)


def call_openai_chat(
    messages: List[Dict], 
    system_prompt: str, 
    model_name: Optional[str] = None
) -> tuple[str, Dict[str, int]]:
    """
    Call OpenAI Chat Completions API with automatic prompt caching
    
    Args:
        messages: Full conversation history (examples + conversation)
        system_prompt: System-level role definition
        model_name: Friendly model name (e.g., "gpt-4.1")
    
    Returns:
        tuple: (response_text, usage_dict)
        
    Caching strategy (automatic by OpenAI):
    - Static content >1024 tokens at start gets cached
    - Cache lifetime: 5-10 minutes
    - 75% discount on cached tokens for GPT-4.1
    """
    try:
        # Get actual model ID from friendly name
        model_id = get_model_id(model_name)
        
        # Structure messages: system first, then conversation history
        full_messages = [
            {"role": "system", "content": system_prompt}
        ] + messages
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model=model_id,
            messages=full_messages,
            max_tokens=4096,
            temperature=0.3
        )
        
        # Extract response text
        response_text = response.choices[0].message.content
        
        # Extract usage information
        usage = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "cached_tokens": 0
        }
        
        # Get cached tokens if available (GPT-4.1 and newer)
        if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
            if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                usage["cached_tokens"] = response.usage.prompt_tokens_details.cached_tokens
        
        return response_text, usage
    
    except Exception as e:
        raise Exception(f"OpenAI API Error: {str(e)}")


def sort_and_apply_modifications(changes: List[Dict]) -> List[Dict]:
    """
    Sort modifications by file, then by descending line number
    This ensures modifications are applied from bottom to top
    """
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


def remove_old_content_from_modifications(parsed: Dict) -> Dict:
    """
    Remove "old_content" field from all modifications
    Frontend doesn't need it per user's request
    """
    if parsed.get('type') == 'code_changes' and 'changes' in parsed:
        for change in parsed['changes']:
            if 'modifications' in change:
                for mod in change['modifications']:
                    # Remove old_content if it exists
                    mod.pop('old_content', None)
    
    return parsed


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
    """
    Process chat request with full conversation history and automatic caching
    
    Response types:
    - "code_generation": New React components/files created
    - "code_modification": Existing code modified
    - "conversation": Chat response (no code)
    - "error": Error occurred
    """
    session_id = request.session_id
    query = request.query
    context = request.context
    model_name = request.model_name
    
    # Initialize session
    if session_id not in conversations:
        conversations[session_id] = ConversationBuffer(max_messages=20)
        generated_code[session_id] = None
    
    conv_buffer = conversations[session_id]
    has_previous_code = generated_code[session_id] is not None
    has_context = context is not None and (context.open_files or context.workspace_tree)
    
    # Inject React examples ONCE at start of conversation
    # OpenAI will automatically cache these (>1024 tokens)
    if not conv_buffer.examples_injected:
        conv_buffer.inject_examples()
    
    # Build context string with XML structure
    context_string = build_context_string(context)
    
    # Determine if modification or generation (hint for model)
    is_modification = is_modification_request(query, has_context, has_previous_code)
    
    # Use system prompt (optimized for GPT-4.1)
    system_prompt = MAIN_SYSTEM_PROMPT
    
    # Build current user message
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
    
    # Get full conversation history (includes examples for automatic caching)
    messages = conv_buffer.get_messages_for_api()
    
    # Call OpenAI with full history and get usage info
    response, usage = call_openai_chat(messages, system_prompt, model_name)
    
    # Add assistant response to history
    conv_buffer.add_message("assistant", response)
    
    # Prepare usage information
    token_usage = TokenUsage(
        input_tokens=usage.get('input_tokens', 0),
        output_tokens=usage.get('output_tokens', 0),
        cached_tokens=usage.get('cached_tokens', 0)
    )
    
    # Parse JSON response or handle as conversation
    try:
        # Look for JSON in response (skip reasoning text)
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            # Found JSON - it's a code response
            json_str = response[json_start:json_end]
            parsed = json.loads(json_str)
            
            # Remove old_content from modifications (user doesn't need it)
            parsed = remove_old_content_from_modifications(parsed)
            
            # Store generated code if generation
            if parsed.get('type') == 'code_generation':
                generated_code[session_id] = json.dumps(parsed, indent=2)
            
            # Sort modifications if modification
            if parsed.get('type') == 'code_changes' and 'changes' in parsed:
                parsed['changes'] = sort_and_apply_modifications(parsed['changes'])
            
            # Determine response type
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
                model_name=model_name or DEFAULT_MODEL
            )
        else:
            raise ValueError("No JSON found in response")
    
    except Exception as e:
        # No valid JSON found - check if it's conversational
        if not is_likely_code_request(query):
            # It's a conversational response (hello, thanks, etc.)
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
                model_name=model_name or DEFAULT_MODEL
            )
        else:
            # User wanted code but model didn't provide JSON - error state
            return ChatResponse(
                type="error",
                parsed={
                    "type": "error",
                    "error": f"Expected code generation but received unexpected response: {str(e)}",
                    "summary": response.strip()
                },
                session_id=session_id,
                is_code_change=False,
                request_type="error",
                workspace_tree=context.workspace_tree.dict() if context and context.workspace_tree else None,
                usage=token_usage,
                model_name=model_name or DEFAULT_MODEL
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
    Process chat request with file uploads via multipart/form-data.
    
    Form fields:
    - query: User's request/query (required)
    - session_id: Session identifier (default: "default")
    - model_name: Model name like "gpt-4.1" (optional)
    - workspace_tree: JSON string of workspace structure (optional)
    - files: Multiple file uploads (optional)
    
    Response: ChatResponse with code generation/modification
    """
    try:
        # Read uploaded files efficiently
        file_contexts = []
        for file in files:
            try:
                # Read file content
                content = await file.read()
                
                # Decode to string (assumes text files)
                try:
                    text_content = content.decode('utf-8')
                except UnicodeDecodeError:
                    # Skip binary files or handle differently
                    print(f"Warning: Skipping binary file {file.filename}")
                    continue
                
                file_contexts.append(FileContext(
                    path=file.filename,
                    content=text_content
                ))
            except Exception as e:
                print(f"Error reading file {file.filename}: {str(e)}")
                continue
        
        # Parse workspace tree if provided
        ws_tree = None
        if workspace_tree:
            try:
                ws_tree = WorkspaceTree(**json.loads(workspace_tree))
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400, 
                    detail="Invalid workspace_tree JSON format"
                )
        
        # Build context
        context = None
        if file_contexts or ws_tree:
            context = ChatContext(
                open_files=file_contexts if file_contexts else None,
                workspace_tree=ws_tree
            )
        
        # Create request object
        request = ChatRequest(
            query=query,
            context=context,
            session_id=session_id,
            model_name=model_name
        )
        
        # Process request
        return await process_chat_request(request)
    
    except HTTPException:
        raise
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


@app.get("/models", tags=["System"])
async def get_available_models():
    """Get list of available OpenAI models"""
    return {
        "models": list(MODEL_MAPPING.keys()),
        "default": DEFAULT_MODEL
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(conversations),
        "active_websockets": len(active_connections),
        "available_models": list(MODEL_MAPPING.keys())
    }


# ============================================================================
# WEBSOCKET ENDPOINT
# ============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time chat communication.
    
    Send JSON: 
    {
      "query": "your request", 
      "context": {...},
      "model_name": "gpt-4.1"  // optional
    }
    
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
                session_id=session_id,
                model_name=request_data.get('model_name')
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
        "message": "React Code Assistant API v3.0 - OpenAI Edition",
        "version": "3.0.0",
        "specialized": "React development with modern hooks and TypeScript",
        "provider": "OpenAI",
        "features": [
            "React-focused code generation and modification",
            "Conversation history with context awareness",
            "Automatic prompt caching (75% discount on cached tokens)",
            "Unified React examples (generation + modification)",
            "XML-structured context handling",
            "Token usage tracking with cache metrics",
            "Multi-model support (GPT-4.1, GPT-4o, o3, o4-mini)",
            "No old_content in modification responses",
            "Optimized for GPT-4.1's literal instruction following"
        ],
        "models": list(MODEL_MAPPING.keys()),
        "default_model": DEFAULT_MODEL,
        "docs": "/docs",
        "websocket": "/ws/{session_id}"
    }


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("React Code Assistant API v3.0 - OpenAI Edition")
    print("=" * 70)
    
    # Check OpenAI API key
    if not OPENAI_API_KEY:
        print("⚠️  WARNING: OpenAI API key not set!")
        print("Required environment variables:")
        print("  - OPENAI_API_KEY")
        print("=" * 70)
    else:
        print("✅ OpenAI API key configured")
    
    # Show model configuration
    print(f"✅ Available models: {len(MODEL_MAPPING)}")
    for model_name, model_id in MODEL_MAPPING.items():
        is_default = " (DEFAULT)" if model_name == DEFAULT_MODEL else ""
        print(f"   - {model_name}{is_default}: {model_id}")
    
    print("\n🎯 Specialized for: React development")
    print("✅ Automatic prompt caching enabled (OpenAI)")
    print("✅ 75% discount on cached tokens (GPT-4.1)")
    print("✅ Cache threshold: >1024 tokens")
    print("✅ Cache lifetime: 5-10 minutes")
    print("✅ Conversation history enabled")
    print("✅ Token usage tracking with cache metrics")
    print("✅ React-specific examples loaded")
    print("✅ Optimized for GPT-4.1 literal instruction following")
    print("❌ old_content removed from modifications")
    print("\n🚀 Starting server on http://0.0.0.0:5000")
    print("📚 API docs: http://0.0.0.0:5000/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=5000)
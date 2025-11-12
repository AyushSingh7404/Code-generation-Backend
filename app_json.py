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
    title="React Code Assistant API",
    description="AI-powered React code generation and modification assistant using AWS Bedrock",
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

# ============================================================================
# AWS & MODEL CONFIGURATION
# ============================================================================

AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

# Model ID mapping from environment variables
# Frontend sends friendly names like "claude-sonnet-4-5"
# Backend maps to actual ARNs
MODEL_MAPPING = {
    "claude-3-5-sonnet": os.getenv('CLAUDE_3_5_SONNET_ID'),
    "claude-3-7-sonnet": os.getenv('CLAUDE_3_7_SONNET_ID'),
    "claude-sonnet-4": os.getenv('CLAUDE_SONNET_4_ID'),
    "claude-sonnet-4-5": os.getenv('CLAUDE_SONNET_4_5_ID'),
}

DEFAULT_MODEL = os.getenv('DEFAULT_MODEL', 'claude-sonnet-4-5')

def get_model_id(model_name: Optional[str] = None) -> str:
    """
    Convert friendly model name to AWS Bedrock ARN
    
    Args:
        model_name: Friendly name like "claude-sonnet-4-5"
    
    Returns:
        Full ARN for the model
    """
    name = model_name or DEFAULT_MODEL
    return MODEL_MAPPING.get(name, MODEL_MAPPING[DEFAULT_MODEL])

# Initialize Bedrock client
bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY
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
    model_name: Optional[str] = None  # e.g., "claude-sonnet-4-5"

class TokenUsage(BaseModel):
    """Token usage information from Claude API"""
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: Optional[int] = 0
    cache_read_input_tokens: Optional[int] = 0

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
# PROMPTS - REACT-FOCUSED
# ============================================================================

# SHORT System Prompt (Role Definition Only) - Will be cached
MAIN_SYSTEM_PROMPT = """You are an expert React developer assistant specializing in modern React development with hooks, TypeScript, and best practices.

<react_expertise>
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
- Generate well-structured, production-ready React components
- Modify existing React code accurately using line-based changes
- Follow React best practices and modern patterns
- Provide TypeScript types when appropriate
- Respond naturally to casual conversation
- Maintain context throughout conversations
</core_responsibilities>

<interaction_style>
- For code requests: Provide structured JSON responses as instructed
- For casual chat: Respond conversationally without JSON
- Ask clarifying questions when requirements are ambiguous
- Be precise and detail-oriented in code generation
</interaction_style>

You will receive task-specific instructions and examples in the conversation. Follow them carefully."""


# React Code Generation Examples and Instructions
REACT_GENERATION_EXAMPLES = """<mode>REACT CODE GENERATION</mode>

<instructions>
You are now in React code generation mode. Follow this process:

1. ANALYZE: Understand the React requirements and plan your component structure
2. STRUCTURE: Determine the file organization (components, hooks, utils, styles)
3. GENERATE: Create complete React code in JSON format

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
      "file": "src/components/ComponentName.jsx",
      "content": "complete file content as a string with escaped newlines"
    }
  ],
  "summary": "Brief description of what was generated"
}
</json_structure>

<react_best_practices>
1. Use functional components with hooks (not class components)
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

<example name="api_data_fetching">
<scenario>User: "Create a UserList component that fetches users from an API"</scenario>
<response>
I'll create a UserList component using useEffect for data fetching, with loading and error states.

{
  "type": "code_generation",
  "changes": [
    {
      "file": "src/components/UserList.jsx",
      "content": "import React, { useState, useEffect } from 'react';\\nimport './UserList.css';\\n\\nconst UserList = ({ apiUrl = 'https://jsonplaceholder.typicode.com/users' }) => {\\n  const [users, setUsers] = useState([]);\\n  const [loading, setLoading] = useState(true);\\n  const [error, setError] = useState(null);\\n\\n  useEffect(() => {\\n    const fetchUsers = async () => {\\n      try {\\n        setLoading(true);\\n        setError(null);\\n        \\n        const response = await fetch(apiUrl);\\n        \\n        if (!response.ok) {\\n          throw new Error(`HTTP error! status: ${response.status}`);\\n        }\\n        \\n        const data = await response.json();\\n        setUsers(data);\\n      } catch (err) {\\n        setError(err.message);\\n        console.error('Error fetching users:', err);\\n      } finally {\\n        setLoading(false);\\n      }\\n    };\\n\\n    fetchUsers();\\n  }, [apiUrl]);\\n\\n  if (loading) {\\n    return (\\n      <div className=\\"user-list-container\\">\\n        <div className=\\"loading\\">Loading users...</div>\\n      </div>\\n    );\\n  }\\n\\n  if (error) {\\n    return (\\n      <div className=\\"user-list-container\\">\\n        <div className=\\"error\\">Error: {error}</div>\\n      </div>\\n    );\\n  }\\n\\n  return (\\n    <div className=\\"user-list-container\\">\\n      <h2>User List</h2>\\n      <div className=\\"user-grid\\">\\n        {users.map(user => (\\n          <div key={user.id} className=\\"user-card\\">\\n            <h3>{user.name}</h3>\\n            <p className=\\"username\\">@{user.username}</p>\\n            <p className=\\"email\\">{user.email}</p>\\n            <p className=\\"company\\">{user.company.name}</p>\\n          </div>\\n        ))}\\n      </div>\\n    </div>\\n  );\\n};\\n\\nexport default UserList;"
    },
    {
      "file": "src/components/UserList.css",
      "content": ".user-list-container {\\n  padding: 2rem;\\n  max-width: 1200px;\\n  margin: 0 auto;\\n}\\n\\n.user-list-container h2 {\\n  text-align: center;\\n  color: #333;\\n  margin-bottom: 2rem;\\n}\\n\\n.loading, .error {\\n  text-align: center;\\n  padding: 2rem;\\n  font-size: 1.2rem;\\n}\\n\\n.error {\\n  color: #e74c3c;\\n}\\n\\n.user-grid {\\n  display: grid;\\n  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));\\n  gap: 1.5rem;\\n}\\n\\n.user-card {\\n  background: white;\\n  padding: 1.5rem;\\n  border-radius: 10px;\\n  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);\\n  transition: transform 0.2s, box-shadow 0.2s;\\n}\\n\\n.user-card:hover {\\n  transform: translateY(-5px);\\n  box-shadow: 0 10px 20px rgba(0, 0, 0, 0.15);\\n}\\n\\n.user-card h3 {\\n  margin: 0 0 0.5rem 0;\\n  color: #667eea;\\n}\\n\\n.username {\\n  color: #7f8c8d;\\n  font-style: italic;\\n  margin: 0.25rem 0;\\n}\\n\\n.email {\\n  color: #555;\\n  font-size: 0.9rem;\\n  margin: 0.5rem 0;\\n}\\n\\n.company {\\n  color: #95a5a6;\\n  font-size: 0.85rem;\\n  margin: 0.5rem 0 0 0;\\n}"
    }
  ],
  "summary": "Created UserList component with useEffect for API data fetching, loading/error states, and responsive grid layout"
}
</response>
</example>

<example name="custom_hook">
<scenario>User: "Create a custom hook for form handling"</scenario>
<response>
I'll create a reusable useForm custom hook that handles form state, validation, and submission.

{
  "type": "code_generation",
  "changes": [
    {
      "file": "src/hooks/useForm.js",
      "content": "import { useState } from 'react';\\n\\nconst useForm = (initialValues, validate, onSubmit) => {\\n  const [values, setValues] = useState(initialValues);\\n  const [errors, setErrors] = useState({});\\n  const [isSubmitting, setIsSubmitting] = useState(false);\\n\\n  const handleChange = (e) => {\\n    const { name, value } = e.target;\\n    setValues(prev => ({ ...prev, [name]: value }));\\n    \\n    // Clear error for this field\\n    if (errors[name]) {\\n      setErrors(prev => ({ ...prev, [name]: '' }));\\n    }\\n  };\\n\\n  const handleSubmit = async (e) => {\\n    e.preventDefault();\\n    \\n    // Validate\\n    const validationErrors = validate ? validate(values) : {};\\n    setErrors(validationErrors);\\n    \\n    // If no errors, submit\\n    if (Object.keys(validationErrors).length === 0) {\\n      setIsSubmitting(true);\\n      try {\\n        await onSubmit(values);\\n      } catch (error) {\\n        setErrors({ submit: error.message });\\n      } finally {\\n        setIsSubmitting(false);\\n      }\\n    }\\n  };\\n\\n  const reset = () => {\\n    setValues(initialValues);\\n    setErrors({});\\n    setIsSubmitting(false);\\n  };\\n\\n  return {\\n    values,\\n    errors,\\n    isSubmitting,\\n    handleChange,\\n    handleSubmit,\\n    reset\\n  };\\n};\\n\\nexport default useForm;"
    },
    {
      "file": "src/components/ExampleFormWithHook.jsx",
      "content": "import React from 'react';\\nimport useForm from '../hooks/useForm';\\n\\nconst ExampleFormWithHook = () => {\\n  const initialValues = {\\n    name: '',\\n    email: '',\\n    message: ''\\n  };\\n\\n  const validate = (values) => {\\n    const errors = {};\\n    \\n    if (!values.name) {\\n      errors.name = 'Name is required';\\n    }\\n    \\n    if (!values.email) {\\n      errors.email = 'Email is required';\\n    } else if (!/^[^\\\\s@]+@[^\\\\s@]+\\\\.[^\\\\s@]+$/.test(values.email)) {\\n      errors.email = 'Invalid email format';\\n    }\\n    \\n    if (!values.message) {\\n      errors.message = 'Message is required';\\n    }\\n    \\n    return errors;\\n  };\\n\\n  const handleFormSubmit = async (values) => {\\n    // Simulate API call\\n    await new Promise(resolve => setTimeout(resolve, 1000));\\n    console.log('Form submitted:', values);\\n    alert('Form submitted successfully!');\\n  };\\n\\n  const { values, errors, isSubmitting, handleChange, handleSubmit, reset } = useForm(\\n    initialValues,\\n    validate,\\n    handleFormSubmit\\n  );\\n\\n  return (\\n    <form onSubmit={handleSubmit} style={{ maxWidth: '400px', margin: '0 auto', padding: '2rem' }}>\\n      <h2>Contact Form (using useForm hook)</h2>\\n      \\n      <div style={{ marginBottom: '1rem' }}>\\n        <label>Name:</label>\\n        <input\\n          type=\\"text\\"\\n          name=\\"name\\"\\n          value={values.name}\\n          onChange={handleChange}\\n          style={{ width: '100%', padding: '0.5rem' }}\\n        />\\n        {errors.name && <span style={{ color: 'red' }}>{errors.name}</span>}\\n      </div>\\n      \\n      <div style={{ marginBottom: '1rem' }}>\\n        <label>Email:</label>\\n        <input\\n          type=\\"email\\"\\n          name=\\"email\\"\\n          value={values.email}\\n          onChange={handleChange}\\n          style={{ width: '100%', padding: '0.5rem' }}\\n        />\\n        {errors.email && <span style={{ color: 'red' }}>{errors.email}</span>}\\n      </div>\\n      \\n      <div style={{ marginBottom: '1rem' }}>\\n        <label>Message:</label>\\n        <textarea\\n          name=\\"message\\"\\n          value={values.message}\\n          onChange={handleChange}\\n          style={{ width: '100%', padding: '0.5rem', minHeight: '100px' }}\\n        />\\n        {errors.message && <span style={{ color: 'red' }}>{errors.message}</span>}\\n      </div>\\n      \\n      <button type=\\"submit\\" disabled={isSubmitting} style={{ marginRight: '1rem' }}>\\n        {isSubmitting ? 'Submitting...' : 'Submit'}\\n      </button>\\n      <button type=\\"button\\" onClick={reset}>Reset</button>\\n    </form>\\n  );\\n};\\n\\nexport default ExampleFormWithHook;"
    }
  ],
  "summary": "Created reusable useForm custom hook with form state management, validation, and example usage component"
}
</response>
</example>
</examples>

<critical_reminders>
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
You are now in React code modification mode. Follow this process:

1. ANALYZE: Understand what React code needs to be changed and why
2. LOCATE: Identify exact line numbers and content to modify
3. MODIFY: Provide precise line-based changes in JSON format

CRITICAL OUTPUT FORMAT:
- Start with brief analysis (2-3 sentences explaining your changes)
- Then output ONLY valid JSON (no markdown, no code blocks, no extra text)
- Format: [Brief reasoning] + JSON structure
- NOTE: Do NOT include "old_content" field - only provide line numbers and new content
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
- start_line: First line number to replace (1-indexed)
- end_line: Last line number to replace (inclusive, 1-indexed)
- new_content: New content to insert (NO old_content field needed)

Use when: Modifying existing code, fixing bugs, updating values, changing props
</operation>

<operation name="insert">
Purpose: Add new lines AFTER a specified line
Required fields:
- start_line: Line number after which to insert (1-indexed)
- new_content: Content to insert

Use when: Adding new hooks, props, state, functions
</operation>

<operation name="insert_before">
Purpose: Add new lines BEFORE a specified line
Required fields:
- start_line: Line number before which to insert (1-indexed)
- new_content: Content to insert

Use when: Adding imports, adding code before existing logic
</operation>

<operation name="delete">
Purpose: Remove lines from the file
Required fields:
- start_line: First line number to delete (1-indexed)
- end_line: Last line number to delete (inclusive, 1-indexed)

Use when: Removing unnecessary code, unused imports, deprecated props
</operation>
</operations>

<react_modification_patterns>
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
1. Line numbers are 1-indexed (first line of file is line 1)
2. NO "old_content" field - frontend handles line identification
3. Preserve proper indentation in new_content
4. Order modifications from top to bottom within each file
5. Use \\n to represent line breaks in new_content strings
6. Escape quotes and backslashes properly
7. Ensure JSON is valid and parseable
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

<example>
<scenario>
User: "Add error handling to the API call in UserList.jsx"

Current file (UserList.jsx):
Line 1: import React, { useState, useEffect } from 'react';
Line 2: 
Line 3: const UserList = () => {
Line 4:   const [users, setUsers] = useState([]);
Line 5:   const [loading, setLoading] = useState(true);
Line 6:   
Line 7:   useEffect(() => {
Line 8:     fetch('/api/users')
Line 9:       .then(res => res.json())
Line 10:       .then(data => setUsers(data))
Line 11:       .finally(() => setLoading(false));
Line 12:   }, []);
Line 13:   
Line 14:   return (
Line 15:     <div>{/* ... */}</div>
Line 16:   );
Line 17: };
</scenario>
<response>
I'll add error state and error handling with try-catch in the useEffect hook.

{
  "type": "code_changes",
  "changes": [
    {
      "file": "UserList.jsx",
      "modifications": [
        {
          "operation": "insert",
          "start_line": 5,
          "new_content": "  const [error, setError] = useState(null);"
        },
        {
          "operation": "replace",
          "start_line": 7,
          "end_line": 12,
          "new_content": "  useEffect(() => {\\n    const fetchUsers = async () => {\\n      try {\\n        setLoading(true);\\n        setError(null);\\n        const response = await fetch('/api/users');\\n        if (!response.ok) throw new Error('Failed to fetch');\\n        const data = await response.json();\\n        setUsers(data);\\n      } catch (err) {\\n        setError(err.message);\\n      } finally {\\n        setLoading(false);\\n      }\\n    };\\n    fetchUsers();\\n  }, []);"
        }
      ]
    }
  ],
  "summary": "Added error state and comprehensive error handling with try-catch for API call"
}
</response>
</example>

<example>
<scenario>
User: "Add a new prop 'disabled' to the Button component and use it"

Current file (Button.jsx):
Line 1: import React from 'react';
Line 2: 
Line 3: const Button = ({ onClick, children }) => {
Line 4:   return (
Line 5:     <button onClick={onClick}>
Line 6:       {children}
Line 7:     </button>
Line 8:   );
Line 9: };
</scenario>
<response>
I'll add the disabled prop to the component signature and apply it to the button element.

{
  "type": "code_changes",
  "changes": [
    {
      "file": "Button.jsx",
      "modifications": [
        {
          "operation": "replace",
          "start_line": 3,
          "end_line": 3,
          "new_content": "const Button = ({ onClick, children, disabled = false }) => {"
        },
        {
          "operation": "replace",
          "start_line": 5,
          "end_line": 5,
          "new_content": "    <button onClick={onClick} disabled={disabled}>"
        }
      ]
    }
  ],
  "summary": "Added disabled prop with default value false and applied it to button element"
}
</response>
</example>
</examples>

<critical_reminders>
- Output brief analysis (2-3 sentences) then ONLY the JSON structure
- NO "old_content" field - not needed for frontend
- No markdown code blocks (```)
- Use \\n for newlines in strings
- Escape quotes and backslashes properly
- Ensure JSON is valid and parseable
- List modifications in top-to-bottom order per file
- Preserve proper React component structure and indentation
</critical_reminders>"""


# Unified Examples Conversation - This gets cached once per session
UNIFIED_EXAMPLES_CONVERSATION = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{REACT_GENERATION_EXAMPLES}\n\n---\n\n{REACT_MODIFICATION_EXAMPLES}",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    },
    {
        "role": "assistant",
        "content": "I understand both React code generation and modification formats. For generation requests, I will create complete React components with proper hooks, state management, and styling in JSON format. For modification requests, I will provide precise line-based changes using the appropriate operations (replace, insert, insert_before, delete) WITHOUT including old_content field. I will always start with brief analysis, then output only valid JSON without markdown or extra text."
    }
]


# ============================================================================
# CONVERSATION BUFFER - IMPROVED WITH CACHING SUPPORT
# ============================================================================

class ConversationBuffer:
    """
    Manages conversation history with caching support for examples
    
    Caching strategy:
    1. System prompt: Always cached
    2. Examples (first 2 messages): Cached once per session
    3. Conversation history: Older messages cached when >3 messages
    """
    
    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages
        self.examples_injected = False
    
    def inject_examples(self):
        """
        Inject unified React examples at the start of conversation (only once)
        This is cached and reused throughout the session
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
        Maintains conversation window while keeping cached examples
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
        Get messages formatted for Claude API with cache control
        
        Caching layers:
        1. Examples (first 2 msgs): Already have cache_control
        2. Older conversation: Cache when >3 messages (changed from >5)
        3. Recent 3 messages: Not cached (kept fresh)
        """
        # Changed threshold from 5 to 3 for more aggressive caching
        if len(self.messages) <= 3:
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
    
    This is a HINT for Claude, not a hard rule.
    Claude can self-correct if classification is wrong.
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


def call_claude_bedrock(
    messages: List[Dict], 
    system_prompt: str, 
    model_name: Optional[str] = None
) -> tuple[str, Dict[str, int]]:
    """
    Call Claude via AWS Bedrock with conversation history and prompt caching
    
    Args:
        messages: Full conversation history with cache control
        system_prompt: System-level role definition (will be cached)
        model_name: Friendly model name (e.g., "claude-sonnet-4-5")
    
    Returns:
        tuple: (response_text, usage_dict)
        
    Caching strategy:
    - System prompt: Always cached
    - Examples in messages: Cached via cache_control in message content
    - Older conversation: Cached via cache_control in message content
    """
    try:
        # Get actual model ID from friendly name
        model_id = get_model_id(model_name)
        
        # Cache system prompt (500 tokens cached every request)
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
        
        # Extract response text and usage information
        response_text = response_body['content'][0]['text']
        usage = response_body.get('usage', {})
        
        return response_text, usage
    
    except Exception as e:
        raise Exception(f"Bedrock API Error: {str(e)}")


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
    Process chat request with full conversation history and caching
    
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
    # These are cached and reused throughout the session
    if not conv_buffer.examples_injected:
        conv_buffer.inject_examples()
    
    # Build context string with XML structure
    context_string = build_context_string(context)
    
    # Determine if modification or generation (hint for Claude)
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
    
    # Call Claude with full history and get usage info
    response, usage = call_claude_bedrock(messages, system_prompt, model_name)
    
    # Add assistant response to history
    conv_buffer.add_message("assistant", response)
    
    # Prepare usage information
    token_usage = TokenUsage(
        input_tokens=usage.get('input_tokens', 0),
        output_tokens=usage.get('output_tokens', 0),
        cache_creation_input_tokens=usage.get('cache_creation_input_tokens', 0),
        cache_read_input_tokens=usage.get('cache_read_input_tokens', 0)
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
                    "message": response.strip()
                },
                session_id=session_id,
                is_code_change=False,
                request_type="conversation",
                workspace_tree=context.workspace_tree.dict() if context and context.workspace_tree else None,
                usage=token_usage,
                model_name=model_name or DEFAULT_MODEL
            )
        else:
            # User wanted code but Claude didn't provide JSON - error state
            return ChatResponse(
                type="error",
                parsed={
                    "type": "error",
                    "error": f"Expected code generation but received unexpected response: {str(e)}",
                    "message": response.strip()
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
async def chat_endpoint(request: ChatRequest):
    """
    Process a chat request for React code generation or modification.
    
    Request body:
    - **query**: The user's request/query
    - **context**: Optional context including open files and workspace structure
    - **session_id**: Session identifier for maintaining conversation history
    - **model_name**: Optional model name (e.g., "claude-sonnet-4-5")
    
    Response types:
    - "code_generation": New React code created
    - "code_modification": Existing code modified
    - "conversation": Chat response
    - "error": Error occurred
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


@app.get("/models", tags=["System"])
async def get_available_models():
    """Get list of available Claude models"""
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
      "model_name": "claude-sonnet-4-5"  // optional
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
        "message": "React Code Assistant API v2.0",
        "version": "2.0.0",
        "specialized": "React development with modern hooks and TypeScript",
        "features": [
            "React-focused code generation and modification",
            "Conversation history with context awareness",
            "Prompt caching for cost optimization (70-80% savings)",
            "Unified React examples (generation + modification)",
            "XML-structured context handling",
            "Token usage tracking",
            "Multi-model support via environment variables",
            "No old_content in modification responses"
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
    print("React Code Assistant API v2.0")
    print("=" * 70)
    
    # Check AWS credentials
    if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
        print("⚠️  WARNING: AWS credentials not set!")
        print("Required environment variables:")
        print("  - AWS_ACCESS_KEY_ID")
        print("  - AWS_SECRET_ACCESS_KEY")
        print("  - AWS_REGION (optional, defaults to us-east-1)")
        print("=" * 70)
    else:
        print("✅ AWS credentials configured")
    
    # Show model configuration
    print(f"✅ Available models: {len(MODEL_MAPPING)}")
    for model_name, model_id in MODEL_MAPPING.items():
        is_default = " (DEFAULT)" if model_name == DEFAULT_MODEL else ""
        print(f"   - {model_name}{is_default}")
        print(f"     {model_id[:50]}...")
    
    print("\n🎯 Specialized for: React development")
    print("✅ Prompt caching enabled (3-layer strategy)")
    print("✅ Conversation history enabled")
    print("✅ Token usage tracking enabled")
    print("✅ React-specific examples loaded")
    print("✅ History caching threshold: >3 messages")
    print("❌ old_content removed from modifications")
    print("\n🚀 Starting server on http://0.0.0.0:8000")
    print("📚 API docs: http://0.0.0.0:8000/docs")
    print("=" * 70)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
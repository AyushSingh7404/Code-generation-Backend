"""
OpenAI Backend Implementation
Optimized for GPT models with markdown-structured prompts and automatic caching
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
from openai import OpenAI
from collections import defaultdict

# ============================================================================
# OPENAI-SPECIFIC PROMPTS (MARKDOWN-OPTIMIZED)
# ============================================================================

OPENAI_SYSTEM_PROMPT = """You are an expert React developer assistant specializing in modern React development with hooks, TypeScript, and best practices.

## React Expertise

You are proficient in:
- **Modern React**: Functional components with hooks (useState, useEffect, useContext, useReducer, useMemo, useCallback)
- **TypeScript**: Type safety and better developer experience
- **Component Patterns**: Composition, avoiding prop drilling, custom hooks
- **State Management**: Context API, Redux, Zustand
- **Routing**: React Router v6
- **Styling**: CSS Modules, Styled Components, Tailwind CSS
- **API Integration**: fetch, axios, React Query, SWR
- **Forms**: React Hook Form, Formik
- **Performance**: memo, lazy loading, code splitting
- **Testing**: Jest, React Testing Library
- **Build Tools**: Vite, Create React App, Next.js

## Core Responsibilities

You MUST:
1. Generate well-structured, production-ready React components
2. Modify existing React code accurately using line-based changes
3. Follow React best practices and modern patterns
4. Provide TypeScript types when appropriate
5. Respond naturally to casual conversation
6. Maintain context throughout conversations
7. Be EXTREMELY PRECISE with line numbers in modifications
8. ALWAYS output valid JSON for code requests (no markdown code blocks around JSON)

## Interaction Style

**For code requests:**
- Provide structured JSON responses as shown in examples
- Start with brief analysis (2-3 sentences)
- Then output ONLY raw JSON (no ```json wrapper)

**For casual chat:**
- Respond conversationally without JSON

**Important:**
- Ask clarifying questions when requirements are ambiguous
- Be precise and detail-oriented
- NEVER wrap JSON output in markdown code blocks

You will receive task-specific instructions and examples in the conversation. Follow them EXACTLY."""

OPENAI_GENERATION_PROMPT = """# React Code Generation Mode

## Instructions

You are now in React code generation mode. Follow this process EXACTLY:

### Step 1: ANALYZE
Understand the React requirements and plan your component structure.

### Step 2: STRUCTURE
Determine the file organization (components, hooks, utils, styles).

### Step 3: GENERATE
Create complete React code in JSON format.

## Output Format (CRITICAL)

**Format:** [Brief analysis] + JSON structure

1. Start with brief analysis (2-3 sentences explaining your approach)
2. Then output ONLY valid JSON
3. **DO NOT** wrap JSON in ```json or ``` blocks
4. Output raw JSON only

### Example Flow:
```
I'll create a Counter component using useState hook with increment/decrement functionality.

{
  "type": "code_generation",
  "changes": [...]
}
```

## Expected JSON Structure

```json
{
  "type": "code_generation",
  "changes": [
    {
      "file": "src/components/ComponentName.jsx",
      "content": "complete file content as string with escaped newlines"
    }
  ],
  "summary": "Brief description of what was generated"
}
```

## React Best Practices

Follow these rules:
1. ✅ Use functional components with hooks (NO class components)
2. ✅ Destructure props for cleaner code
3. ✅ Use TypeScript types when applicable (.tsx extension)
4. ✅ Follow naming: PascalCase for components, camelCase for functions
5. ✅ Keep components focused (single responsibility)
6. ✅ Extract reusable logic into custom hooks
7. ✅ Use proper key props in lists
8. ✅ Handle loading and error states
9. ✅ Add PropTypes or TypeScript interfaces
10. ✅ Include all necessary imports

## File Structure Patterns

Use these paths:
- **Components**: `src/components/ComponentName.jsx` or `.tsx`
- **Hooks**: `src/hooks/useHookName.js`
- **Utils**: `src/utils/utilName.js`
- **Styles**: `src/components/ComponentName.module.css`
- **Types**: `src/types/types.ts`
- **API**: `src/api/apiName.js`

## Critical Reminders

- ⚠️ Output brief analysis FIRST (2-3 sentences)
- ⚠️ Then output ONLY the JSON (no markdown blocks)
- ⚠️ Properly escape special characters (\\n for newlines, \\" for quotes)
- ⚠️ Ensure JSON is valid and parseable
- ⚠️ Include all necessary imports
- ⚠️ Use proper React patterns"""

OPENAI_MODIFICATION_PROMPT = """# React Code Modification Mode

## Instructions

You are now in React code modification mode. Follow this process EXACTLY:

### Step 1: ANALYZE
Understand what React code needs to be changed and why.

### Step 2: LOCATE
Identify EXACT line numbers (1-indexed) and content to modify.

### Step 3: MODIFY
Provide precise line-based changes in JSON format.

## Output Format (CRITICAL)

**Format:** [Brief analysis] + JSON structure

1. Start with brief analysis (2-3 sentences explaining changes)
2. Then output ONLY valid JSON
3. **DO NOT** wrap JSON in ```json or ``` blocks
4. Output raw JSON only
5. **DO NOT** include "old_content" field

### Example Flow:
```
I'll add a loading state using useState and display a loading message.

{
  "type": "code_changes",
  "changes": [...]
}
```

## Expected JSON Structure

```json
{
  "type": "code_changes",
  "changes": [
    {
      "file": "path/to/Component.jsx",
      "modifications": [
        {
          "operation": "replace",
          "start_line": 5,
          "end_line": 7,
          "new_content": "new code here"
        }
      ]
    }
  ],
  "summary": "Brief description of changes"
}
```

## Operations

### 1. **replace**
Change existing lines in the component.

**Fields:**
- `start_line`: First line to replace (1-indexed, must be EXACT)
- `end_line`: Last line to replace (inclusive, 1-indexed, must be EXACT)
- `new_content`: New content (NO old_content field)

**Use when:** Modifying code, fixing bugs, updating values

### 2. **insert**
Add new lines AFTER a specified line.

**Fields:**
- `start_line`: Line after which to insert (1-indexed)
- `new_content`: Content to insert

**Use when:** Adding new hooks, state, functions

### 3. **insert_before**
Add new lines BEFORE a specified line.

**Fields:**
- `start_line`: Line before which to insert (1-indexed)
- `new_content`: Content to insert

**Use when:** Adding imports, code before existing logic

### 4. **delete**
Remove lines from the file.

**Fields:**
- `start_line`: First line to delete (1-indexed)
- `end_line`: Last line to delete (inclusive, 1-indexed)

**Use when:** Removing unnecessary code

## Modification Patterns

Common React modification patterns:
1. **Adding state**: Insert useState after existing hooks
2. **Adding props**: Modify component function signature
3. **Event handlers**: Insert new function before return
4. **JSX changes**: Replace specific lines in return
5. **Imports**: Insert at top of file
6. **Hooks**: Replace hook definition lines
7. **Effects**: Insert useEffect after state

## Critical Rules

- ⚠️ Line numbers are 1-indexed (first line = 1)
- ⚠️ **NO** "old_content" field
- ⚠️ Count line numbers EXACTLY (include blank lines)
- ⚠️ Preserve proper indentation
- ⚠️ Use \\n for line breaks
- ⚠️ Escape quotes properly
- ⚠️ Order modifications top to bottom
- ⚠️ Ensure valid JSON

## Critical Reminders

- ⚠️ Output brief analysis FIRST (2-3 sentences)
- ⚠️ Then output ONLY the JSON (no markdown blocks)
- ⚠️ **NO** "old_content" field
- ⚠️ Be EXTREMELY PRECISE with line numbers
- ⚠️ Count ALL lines including blanks"""

# Unified examples for OpenAI (automatic caching >1024 tokens)
OPENAI_EXAMPLES = [
    {
        "role": "user",
        "content": f"{OPENAI_GENERATION_PROMPT}\n\n---\n\n{OPENAI_MODIFICATION_PROMPT}"
    },
    {
        "role": "assistant",
        "content": "I understand both React code generation and modification formats EXACTLY. For generation, I create complete React components in JSON format. For modifications, I provide precise line-based changes WITHOUT old_content field. I ALWAYS start with brief analysis, then output ONLY valid JSON without markdown code blocks. I am EXTREMELY PRECISE with line numbers and count them EXACTLY."
    }
]

# ============================================================================
# OPENAI CONVERSATION BUFFER
# ============================================================================

class OpenAIConversationBuffer:
    """Manages conversation history with OpenAI automatic caching"""
    
    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages
        self.examples_injected = False
    
    def inject_examples(self):
        """Inject examples (OpenAI caches automatically when >1024 tokens)"""
        if not self.examples_injected and not self.messages:
            self.messages = OPENAI_EXAMPLES.copy()
            self.examples_injected = True
    
    def add_message(self, role: str, content: str):
        """Add message to history"""
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Keep recent messages but preserve examples at start"""
        if len(self.messages) > self.max_messages:
            if self.examples_injected:
                examples = self.messages[:2]
                recent = self.messages[-(self.max_messages - 2):]
                self.messages = examples + recent
            else:
                self.messages = self.messages[-self.max_messages:]
            
            if len(self.messages) > 2 and self.messages[2]["role"] != "user":
                self.messages = self.messages[:2] + self.messages[3:]
    
    def get_messages_for_api(self) -> List[Dict]:
        """Get messages for OpenAI (automatic caching handles optimization)"""
        return self.messages
    
    def clear(self):
        """Clear conversation history"""
        self.messages = []
        self.examples_injected = False

# ============================================================================
# OPENAI API CLIENT
# ============================================================================

class OpenAIClient:
    """OpenAI API client with automatic caching"""
    
    def __init__(self):
        # Model mapping for OpenAI
        self.model_mapping = {
            "gpt-4o": "gpt-4o-2024-11-20",
            "gpt-4o-mini": "gpt-4o-mini-2024-07-18",
            "o1": "o1-2024-12-17",
            "o1-mini": "o1-mini-2024-09-12"
        }
        self.default_model = os.getenv('DEFAULT_OPENAI_MODEL', 'gpt-4o')
        
        # Initialize OpenAI client
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def get_model_id(self, model_name: Optional[str] = None) -> str:
        """Convert friendly name to OpenAI model ID"""
        name = model_name or self.default_model
        return self.model_mapping.get(name, self.model_mapping[self.default_model])
    
    def call_api(
        self, 
        messages: List[Dict], 
        system_prompt: str, 
        model_name: Optional[str] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Call OpenAI API with automatic caching
        
        Returns:
            tuple: (response_text, usage_dict)
        """
        try:
            model_id = self.get_model_id(model_name)
            
            # Structure messages: system first, then conversation
            full_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages
            
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=model_id,
                messages=full_messages,
                max_tokens=4096,
                temperature=0.3
            )
            
            # Extract response text
            response_text = response.choices[0].message.content
            
            # Extract usage information
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "cached_tokens": 0
            }
            
            # Get cached tokens if available
            if hasattr(response.usage, 'prompt_tokens_details') and response.usage.prompt_tokens_details:
                if hasattr(response.usage.prompt_tokens_details, 'cached_tokens'):
                    usage["cached_tokens"] = response.usage.prompt_tokens_details.cached_tokens
            
            return response_text, usage
        
        except Exception as e:
            raise Exception(f"OpenAI API Error: {str(e)}")
    
    def is_openai_model(self, model_name: str) -> bool:
        """Check if model name is an OpenAI model"""
        return model_name in self.model_mapping

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def sort_modifications(changes: List[Dict]) -> List[Dict]:
    """Sort modifications by descending line number"""
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

def remove_old_content(parsed: Dict) -> Dict:
    """Remove old_content field from modifications"""
    if parsed.get('type') == 'code_changes' and 'changes' in parsed:
        for change in parsed['changes']:
            if 'modifications' in change:
                for mod in change['modifications']:
                    mod.pop('old_content', None)
    
    return parsed
"""
Claude Backend Implementation
Optimized for Claude with XML-structured prompts and explicit caching
"""

import os
import json
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
import boto3
from collections import defaultdict

# ============================================================================
# CLAUDE-SPECIFIC PROMPTS (XML-OPTIMIZED)
# ============================================================================

CLAUDE_SYSTEM_PROMPT = """You are an expert React developer assistant specializing in modern React development with hooks, TypeScript, and best practices.

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

CLAUDE_GENERATION_PROMPT = """<mode>REACT CODE GENERATION</mode>

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

<critical_reminders>
- Output brief analysis (2-3 sentences) then ONLY the JSON structure
- No markdown code blocks (```)
- Use proper React patterns and hooks
- Include all necessary imports
- Properly escape all special characters in content strings
- Ensure JSON is valid and parseable
</critical_reminders>"""

CLAUDE_MODIFICATION_PROMPT = """<mode>REACT CODE MODIFICATION</mode>

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
- start_line: First line number to replace (1-indexed)
- end_line: Last line number to replace (inclusive, 1-indexed)
- new_content: New content to insert (NO old_content field needed)
</operation>

<operation name="insert">
- start_line: Line number after which to insert (1-indexed)
- new_content: Content to insert
</operation>

<operation name="insert_before">
- start_line: Line number before which to insert (1-indexed)
- new_content: Content to insert
</operation>

<operation name="delete">
- start_line: First line number to delete (1-indexed)
- end_line: Last line number to delete (inclusive, 1-indexed)
</operation>
</operations>

<critical_reminders>
- Output brief analysis (2-3 sentences) then ONLY the JSON structure
- NO "old_content" field - not needed for frontend
- No markdown code blocks (```)
- Use \\n for newlines in strings
- Ensure JSON is valid and parseable
- List modifications in top-to-bottom order per file
</critical_reminders>"""

# Unified examples conversation for Claude (with cache control)
CLAUDE_EXAMPLES = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": f"{CLAUDE_GENERATION_PROMPT}\n\n---\n\n{CLAUDE_MODIFICATION_PROMPT}",
                "cache_control": {"type": "ephemeral"}
            }
        ]
    },
    {
        "role": "assistant",
        "content": "I understand both React code generation and modification formats. For generation requests, I will create complete React components in JSON format. For modification requests, I will provide precise line-based changes WITHOUT old_content field. I will always start with brief analysis, then output only valid JSON without markdown."
    }
]

# ============================================================================
# CLAUDE CONVERSATION BUFFER
# ============================================================================

class ClaudeConversationBuffer:
    """Manages conversation history with Claude-specific caching"""
    
    def __init__(self, max_messages=20):
        self.messages = []
        self.max_messages = max_messages
        self.examples_injected = False
    
    def inject_examples(self):
        """Inject examples with cache control for Claude"""
        if not self.examples_injected and not self.messages:
            self.messages = CLAUDE_EXAMPLES.copy()
            self.examples_injected = True
    
    def add_message(self, role: str, content: str):
        """Add message to history"""
        self.messages.append({"role": role, "content": content})
        self._trim_if_needed()
    
    def _trim_if_needed(self):
        """Keep recent messages but preserve examples"""
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
        """Get messages with cache control for Claude"""
        if len(self.messages) <= 3:
            return self.messages
        
        if self.examples_injected:
            examples = self.messages[:2]
            conversation = self.messages[2:]
            
            if len(conversation) <= 3:
                return self.messages
            
            cacheable_conversation = conversation[:-3]
            recent_conversation = conversation[-3:]
            
            formatted = examples.copy()
            
            for i, msg in enumerate(cacheable_conversation):
                formatted_msg = {"role": msg["role"], "content": msg["content"]}
                
                if i == len(cacheable_conversation) - 1:
                    formatted_msg["content"] = [
                        {
                            "type": "text",
                            "text": msg["content"],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ]
                
                formatted.append(formatted_msg)
            
            formatted.extend(recent_conversation)
            return formatted
        else:
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
# CLAUDE API CLIENT
# ============================================================================

class ClaudeClient:
    """Claude API client via AWS Bedrock"""
    
    def __init__(self):
        # Model mapping for Claude
        self.model_mapping = {
            "claude-3-5-sonnet": os.getenv('CLAUDE_3_5_SONNET_ID'),
            "claude-3-7-sonnet": os.getenv('CLAUDE_3_7_SONNET_ID'),
            "claude-sonnet-4": os.getenv('CLAUDE_SONNET_4_ID'),
            "claude-sonnet-4-5": os.getenv('CLAUDE_SONNET_4_5_ID'),
        }
        self.default_model = os.getenv('DEFAULT_CLAUDE_MODEL', 'claude-sonnet-4-5')
        
        # Initialize Bedrock client
        self.bedrock_runtime = boto3.client(
            service_name='bedrock-runtime',
            region_name=os.getenv('AWS_REGION', 'us-east-1'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
    
    def get_model_id(self, model_name: Optional[str] = None) -> str:
        """Convert friendly name to ARN"""
        name = model_name or self.default_model
        return self.model_mapping.get(name, self.model_mapping[self.default_model])
    
    def call_api(
        self, 
        messages: List[Dict], 
        system_prompt: str, 
        model_name: Optional[str] = None
    ) -> Tuple[str, Dict[str, int]]:
        """
        Call Claude via AWS Bedrock with caching
        
        Returns:
            tuple: (response_text, usage_dict)
        """
        try:
            model_id = self.get_model_id(model_name)
            
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
                "messages": messages,
                "temperature": 0.3
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            
            response_text = response_body['content'][0]['text']
            usage = response_body.get('usage', {})
            
            return response_text, usage
        
        except Exception as e:
            raise Exception(f"Claude API Error: {str(e)}")
    
    def is_claude_model(self, model_name: str) -> bool:
        """Check if model name is a Claude model"""
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
#!/usr/bin/env python3
"""
Terminal chatbot using Groq API with MCP filesystem and git servers.
Requires: pip install groq mcp anthropic-mcp-sdk python-dotenv
"""

import asyncio
import json
import os
import sys
from typing import List, Dict, Any, Optional

import groq
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPServerManager:
    """Manages MCP server connections."""
    
    def __init__(self):
        self.servers = {}
    
    async def connect_server(self, name: str, server_params: StdioServerParameters):
        """Connect to an MCP server."""
        try:
            stdio_context = stdio_client(server_params)
            read, write = await stdio_context.__aenter__()
            
            session_context = ClientSession(read, write)
            session = await session_context.__aenter__()
            await session.initialize()
            
            self.servers[name] = {
                'session': session,
                'stdio_context': stdio_context,
                'session_context': session_context
            }
            print(f"✓ Connected to {name} MCP server")
            return True
        except Exception as e:
            print(f"⚠ Could not connect to {name} MCP server: {e}")
            return False
    
    async def setup_all_servers(self):
        """Setup all MCP servers."""
        # Filesystem MCP server
        fs_server = StdioServerParameters(
            command="npx",
            args=["@modelcontextprotocol/server-filesystem", os.getcwd() + '/test'],
            env=None
        )
        
        # Git MCP server  
        git_server = StdioServerParameters(
            command="uv",
            args=["tool", "run", "mcp-server-git"],
            env=None
        )
        
        # ArXiv MCP server (local)
        arxiv_server = StdioServerParameters(
            command="uv",
            args=["run", "--directory", "/Users/manuelmartinezflores/Documents/GitHub/arxiv-mcp-server",
                  "/Users/manuelmartinezflores/Documents/GitHub/arxiv-mcp-server/arxiv_mcp_server.py"],
            env=None
        )
        
        await self.connect_server('filesystem', fs_server)
        await self.connect_server('git', git_server)
        await self.connect_server('arxiv', arxiv_server)
    
    def get_session(self, name: str) -> Optional[ClientSession]:
        """Get a session for a server."""
        return self.servers.get(name, {}).get('session')
    
    def get_all_sessions(self) -> Dict[str, ClientSession]:
        """Get all active sessions."""
        return {name: data['session'] for name, data in self.servers.items() if 'session' in data}
    
    async def cleanup(self):
        """Clean up all server connections."""
        for name, data in self.servers.items():
            try:
                if 'session_context' in data:
                    await data['session_context'].__aexit__(None, None, None)
                if 'stdio_context' in data:
                    await data['stdio_context'].__aexit__(None, None, None)
            except Exception as e:
                print(f"Error cleaning up {name}: {e}")
        
        self.servers.clear()


class GroqMCPChatbot:
    def __init__(self, groq_api_key: str):
        self.groq_client = groq.Groq(api_key=groq_api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.mcp_manager = MCPServerManager()
        self.mcp_log = []
    
    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get all available tools from MCP servers."""
        all_tools = []
        
        for server_name, session in self.mcp_manager.get_all_sessions().items():
            try:
                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    # Convert MCP tool format to Groq function calling format
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": f"{server_name}_{tool.name}",
                            "description": tool.description or f"Tool {tool.name} from {server_name}",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}}
                        }
                    }
                    all_tools.append(tool_def)
            except Exception as e:
                print(f"Error getting tools from {server_name}: {e}")
                
        return all_tools
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call via MCP."""
        # Parse server and tool name
        if '_' in tool_name:
            server_name, actual_tool_name = tool_name.split('_', 1)
        else:
            return f"Invalid tool name format: {tool_name}"
            
        session = self.mcp_manager.get_session(server_name)
        if not session:
            return f"Server {server_name} not available"
        
        try:
            result = await session.call_tool(actual_tool_name, arguments)
            if result.content:
                # Format the result content
                formatted_results = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        formatted_results.append(content.text)
                    else:
                        formatted_results.append(str(content))
                return '\n'.join(formatted_results)
            return "Tool executed successfully (no output)"
        except Exception as e:
            return f"Error executing {tool_name}: {e}"
    
    async def chat_with_tools(self, user_message: str) -> str:
        """Send message to Groq with available tools."""
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Get available tools
        tools = await self.get_available_tools()
        
        
        # Prepare messages for Groq
        messages = [
            {"role": "system", "content": "You are a helpful assistant with access to filesystem, git, and arXiv tools. Use tools when appropriate to help the user. For arXiv searches, you can search by query terms, authors, categories, or get specific paper details. You can use multiple tools in sequence if needed to fully answer the user's question. However do not use any if you do not need to. "}
        ] + self.conversation_history
        
        max_iterations = 5  # Prevent infinite loops
        iteration = 0
        
        try:
            while iteration < max_iterations:
                iteration += 1
                
                # Make request to Groq
                response = self.groq_client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=messages,
                    tools=tools if tools else None,
                    tool_choice="auto" if tools else None,
                    temperature=0.7,
                    max_tokens=4096
                )
                
                assistant_message = response.choices[0].message
                
                # Check if the model wants to use tools
                if assistant_message.tool_calls:
                    # Add assistant message to conversation
                    assistant_msg = {
                        "role": "assistant",
                        "content": assistant_message.content or ""
                    }
                    
                    # Add tool_calls to the message if they exist
                    if assistant_message.tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": tc.type,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments
                                }
                            } for tc in assistant_message.tool_calls
                        ]
                    
                    messages.append(assistant_msg)
                    
                    # Execute tool calls
                    for tool_call in assistant_message.tool_calls:
                        tool_name = tool_call.function.name
                        tool_args = json.loads(tool_call.function.arguments)
                        
                        print(f"🔧 Executing {tool_name}...")
                        result = await self.execute_tool(tool_name, tool_args)
                        
                        # Add tool result to messages
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": tool_name,
                            "content": result
                        })
                        self.mcp_log.append({
                            'tool_name': tool_name,
                            'args': tool_args,
                            'answer': result
                        })
                    
                    # Continue the loop to let the model potentially make more tool calls
                    continue
                
                else:
                    # No more tools to call, we have our final response
                    content = assistant_message.content or "I apologize, but I couldn't generate a response."
                    
                    # Update conversation history with the complete conversation
                    self.conversation_history = messages[1:]  # Skip system message
                    self.conversation_history.append({"role": "assistant", "content": content})
                    
                    return content
            
            # If we hit max iterations, return what we have
            return "I've completed multiple tool calls but reached the iteration limit. The results from the tools should be helpful."
                
        except Exception as e:
            error_msg = f"Error communicating with Groq: {e}"
            print(f"❌ {error_msg}")
            return error_msg
        
    async def format_mcp_log(self):
        log = [f"Tool: {call['tool_name']}\nArgs: {call['args']}\nAnswer:\n{call['answer']}" for call in self.mcp_log]
        return ('\n' + '='*80 + '\n').join(log)
    
    async def run_chat_loop(self):
        """Main chat loop."""
        print("🤖 Terminal Chatbot with MCP Tools")
        print("Type 'quit', 'exit', or press Ctrl+C to exit")
        print("Type 'clear' to clear conversation history")
        print("=" * 50)
        
        # Setup MCP servers
        await self.mcp_manager.setup_all_servers()
        
        # Show available tools
        tools = await self.get_available_tools()
        if tools:
            print(f"📋 Available tools: {len(tools)}")
            for tool in tools:  # Show first 5
                print(f"  • {tool['function']['name']}: {tool['function']['description']}")
            # if len(tools) > 5:
            #     print(f"  ... and {len(tools) - 5} more")
        else:
            print("⚠ No MCP tools available")
        print("=" * 50)
        
        try:
            while True:
                try:
                    user_input = input("\n💬 You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit']:
                        print("👋 Goodbye!")
                        break
                    elif user_input.lower() == 'clear':
                        self.conversation_history.clear()
                        print("🧹 Conversation history cleared!")
                        continue
                    elif user_input.lower() == 'log':
                        log = await self.format_mcp_log()
                        print(log)
                        continue
                    elif user_input.lower() == 'tools':
                        log = await self.get_available_tools()
                        print(log)
                        continue
                    elif not user_input:
                        continue
                    
                    print("🤔 Assistant: ", end="", flush=True)
                    response = await self.chat_with_tools(user_input)
                    print(response)
                    
                except KeyboardInterrupt:
                    print("\n👋 Goodbye!")
                    break
                except EOFError:
                    print("\n👋 Goodbye!")
                    break
                    
        finally:
            # Clean up MCP sessions
            await self.mcp_manager.cleanup()


async def main():
    """Main entry point."""
    # Load environment variables from .env file
    load_dotenv()
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("❌ Error: GROQ_API_KEY not found")
        print("Please add it to your .env file: GROQ_API_KEY=your_api_key_here")
        sys.exit(1)
    
    chatbot = GroqMCPChatbot(groq_api_key)
    await chatbot.run_chat_loop()


if __name__ == "__main__":
    asyncio.run(main())
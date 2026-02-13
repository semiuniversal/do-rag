import asyncio
import sys

from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    print("Connecting to MCP Server at http://localhost:8000/sse...")
    
    try:
        async with sse_client("http://localhost:8000/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                # List tools to confirm connection and finding 'ask_documents'
                tools = await session.list_tools()
                ask_tool = next((t for t in tools.tools if t.name == "ask_documents"), None)
                
                if not ask_tool:
                    print("Error: 'ask_documents' tool not found on server.")
                    return
                
                print("\n=== Local RAG Chat (Qwen 2.5 + Nomic) ===")
                print("Type 'quit' or 'exit' to stop.\n")
                
                while True:
                    try:
                        query = input("> ")
                        if query.lower() in ["quit", "exit"]:
                            break
                        if not query.strip():
                            continue
                            
                        print("\nThinking...", end="", flush=True)
                        
                        # Call the tool
                        result = await session.call_tool("ask_documents", arguments={"query": query})
                        
                        print("\r", end="") # Clear "Thinking..."
                        
                        # Inspect result structure
                        # ToolResult is usually a list of TextContent / ImageContent
                        if result.content:
                            for content in result.content:
                                if content.type == "text":
                                    print(content.text)
                                else:
                                    print(f"[{content.type} content]")
                        else:
                            print("No response content.")
                            
                        print("-" * 40)
                        
                    except KeyboardInterrupt:
                        break
                    except Exception as e:
                        print(f"\nError: {e}")
                        
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure the server is running: ./run_mcp_server.sh start")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

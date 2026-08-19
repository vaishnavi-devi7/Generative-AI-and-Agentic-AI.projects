import os
import streamlit as st
from typing import TypedDict, Annotated
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import create_react_agent
from operator import add


api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("GROQ_API_KEY is not set. Please configure it.")
    st.stop()


llm = ChatGroq(
    temperature=0,
    model_name="openai/gpt-oss-120b",
    api_key=api_key
)


class CodeState(TypedDict):
    query: str
    context: str
    messages: Annotated[list, add]
    response: str


def analyze_code_context(query: str) -> str:
    """
    Analyze the code and provide context.
    """
    return f"Analyzing the following code snippet:\n{query}"


@tool
def get_code_analysis_guidance(query: str) -> str:
    """Get guidance on how to analyze code properly."""
    return "Focus on: syntax, logic errors, performance, readability and best practices."


tools = [get_code_analysis_guidance]


def create_propmt_node(state: CodeState) -> CodeState:
    """Create a structured prompt with code snippet"""
    query = state["query"]
    context = analyze_code_context(query)

    prompt = f"""You are a code analysis assistant.
Context:
{context}

Task:
-Analyze the code snippet
-Identify potential issues or bugs
-Suggest improvements or optimizations
-Explain reasoning step by step

User Code:
{query}
"""

    return {
        **state,
        "context": context,
        "messages": [HumanMessage(content=prompt)]
    }


def generate_response_node(state: CodeState) -> CodeState:
    """Generate response using LLM"""
    agent = create_react_agent(model=llm, tools=tools)

    response = agent.invoke({
        "messages": state["messages"]
    })

    if response and "messages" in response:
        bot_message = response["messages"][-1]
        response_text = bot_message.content
    else:
        response_text = str(response)

    return {
        **state,
        "response": response_text,
        "messages": state["messages"] +
        [AIMessage(content=response_text)]
    }


def build_code_graph():
    graph = StateGraph(CodeState)
    graph.add_node("create prompt", create_propmt_node)
    graph.add_node("generate_response", generate_response_node)
    graph.add_edge(START, "create prompt")
    graph.add_edge("create prompt", "generate_response")
    graph.add_edge("generate_response", END)

    return graph.compile()


code_workflow = build_code_graph()

st.title("Code Analyzer Chatbot (LangGraph+LLM)")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_code = st.text_area("Paste your code snippet here:")

if user_code:
    try:
        with st.spinner("Analyzing code..."):
            initial_state = {
                "query": user_code,
                "context": " ",
                "messages": [],
                "response": ""
            }

            result = code_workflow.invoke(initial_state)

            bot_response_text = result.get(
                "response",
                "No response generated"
            )

            st.session_state.conversation.append({
                "user": user_code,
                "bot": bot_response_text
            })

        st.success("Analysis Complete")

        st.markdown(
            f"**You:**\n```python\n{user_code}\n```"
        )

        st.markdown(
            f"**Assistant Analysis:**\n{bot_response_text}"
        )

        with st.expander("Context Used"):
            st.text(
                result.get(
                    "context",
                    "No context found"
                )
            )

    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        st.error(traceback.format_exc())


if st.session_state.conversation:
    st.divider()
    st.subheader("Conversation History")

    for turn in st.session_state.conversation:
        st.markdown(
            f"**You:**\n```python\n{turn['user']}\n```"
        )
        st.markdown(
            f"**Assistant Analysis:**\n{turn['bot']}"
        )
        st.divider()
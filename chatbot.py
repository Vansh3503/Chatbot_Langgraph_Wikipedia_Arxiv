import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

import streamlit as st
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import ChatGoogleGenerativeAI

class ChatbotPipeline:
    def __init__(self):
        self._load_environment_variables()
        self.tools = self._initialize_tools()
        self.llm = self._initialize_llm()
        self.llm_with_tools = self.llm.bind_tools(tools=self.tools)
        self.graph = self._build_graph()

    @staticmethod
    def _load_environment_variables():
        load_dotenv()
        if "GOOGLE_API_KEY" not in os.environ:
            raise EnvironmentError("GOOGLE_API_KEY not found in environment variables.")

    @staticmethod
    def _initialize_tools():
        arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
        arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

        wiki_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=300)
        wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

        return [wiki_tool, arxiv_tool]

    @staticmethod
    def _initialize_llm():
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def _build_graph(self):
        class State(TypedDict):
            messages: Annotated[list, add_messages]

        def chatbot(state: State):
            response = self.llm_with_tools.invoke(state["messages"])
            return {"messages": state["messages"] + [response]}

        graph_builder = StateGraph(State)
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_edge(START, "chatbot")

        tool_node = ToolNode(tools=self.tools)
        graph_builder.add_node("tools", tool_node)

        graph_builder.add_conditional_edges("chatbot", tools_condition)
        graph_builder.add_edge("tools", "chatbot")
        graph_builder.add_edge("chatbot", END)

        return graph_builder.compile()

    def chat(self, user_messages: list):
        events = self.graph.stream({"messages": user_messages}, stream_mode="values")
        responses = [event["messages"][-1].content for event in events]
        return responses

# ---------------- Streamlit App -------------------

def main():
    st.set_page_config(page_title="LLM Chatbot", page_icon="🤖", layout="centered")
    st.title("🤖 Gemini-Powered Chatbot with Wikipedia & Arxiv Tools")

    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatbotPipeline()
        st.session_state.messages = []

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

    user_input = st.chat_input("Ask me anything...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Generating response..."):
            responses = st.session_state.chatbot.chat(st.session_state.messages)
            bot_response = responses[-1]
            st.session_state.messages.append({"role": "assistant", "content": bot_response})

        # Rerender the chat without experimental_rerun (automatically handled by Streamlit)
        chat_container.empty()
        with chat_container:
            for message in st.session_state.messages:
                if message["role"] == "user":
                    with st.chat_message("user"):
                        st.markdown(message["content"])
                else:
                    with st.chat_message("assistant"):
                        st.markdown(message["content"])

if __name__ == "__main__":
    main()

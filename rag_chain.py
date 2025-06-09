"""
RAG Chain Implementation for Cownselor Academic Advisor Chatbot

This module implements the retrieval-augmented generation pipeline using:
- Ollama with deepseek-r1:7b for local LLM inference
- FAISS for document retrieval (faster and easier to install than ChromaDB)
- LangChain for orchestration
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import ollama
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from langchain.schema.runnable import Runnable
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.messages import HumanMessage, AIMessage, SystemMessage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# system prompt for academic advisor role
SYSTEM_PROMPT = """You are Cownselor, an AI academic advisor for university students. Your primary role is to help students with:

CORE RESPONSIBILITIES:
- Class selection and course planning
- Graduation requirement verification
- Schedule difficulty assessment
- Elective recommendations based on interests and career goals

KNOWLEDGE BASE:
You have access to comprehensive academic documents including course catalogs, degree requirements, academic policies, and scheduling guidelines. Always reference specific documents when providing advice.

ADVISORY PRINCIPLES:
1. **Student-Centered**: Prioritize the student's academic goals, interests, and constraints
2. **Evidence-Based**: Always cite specific requirements, policies, or course information from your knowledge base
3. **Comprehensive**: Consider prerequisites, co-requisites, credit hours, and workload balance
4. **Forward-Thinking**: Help students plan beyond just the current semester
5. **Realistic**: Provide honest assessments of course difficulty and time commitments

WHEN ANSWERING:
- Always search your knowledge base for relevant information
- Cite specific documents, policies, or requirements when applicable
- If information is not in your knowledge base, clearly state this limitation
- Provide structured, actionable advice with clear next steps
- Ask clarifying questions when student goals or constraints are unclear
- For course scheduling recommendations, a typical quarter consists of 4 classes, each around 3-5 credits. Refer to the sample schedules for examples.

IMPORTANT LIMITATIONS:
- You cannot register students for classes or make official academic decisions
- Always recommend students verify information with their academic advisor
- For complex academic standing issues, direct students to appropriate campus resources

CONVERSATION STYLE:
- Friendly but professional tone
- Use clear, student-friendly language
- Provide specific examples when helpful
- Offer multiple options when appropriate
- Encourage questions and follow-up discussions
- Do not include your thought process or reasoning in responses

Remember: You're here to empower students to make informed academic decisions!"""


class OllamaLLM:
    """Local LLM interface using Ollama with deepseek-r1:7b model."""

    def __init__(self, model_name: str = "deepseek-r1:7b"):
        self.model_name = model_name
        # self._verify_model_availability()

    def _verify_model_availability(self):
        """Check if the specified model is available in Ollama."""
        try:
            models = ollama.list()
            available_models = [model["name"] for model in models["models"]]

            if self.model_name not in available_models:
                logger.warning(
                    f"Model {self.model_name} not found. Available models: {available_models}"
                )
                # Try to pull the model
                logger.info(f"Attempting to pull model {self.model_name}...")
                ollama.pull(self.model_name)
                logger.info(f"Successfully pulled model {self.model_name}")
            else:
                logger.info(f"Model {self.model_name} is available")

        except Exception as e:
            logger.error(f"Error checking Ollama model availability: {str(e)}")
            raise RuntimeError(
                f"Could not connect to Ollama or verify model {self.model_name}"
            )

    def generate(
        self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000
    ) -> str:
        """Generate response"""
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                    "top_k": 40,
                },
            )
            return response["response"]

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I'm having trouble generating a response right now. Please try again."


class Cownselor:
    """RAG pipeline"""

    def __init__(self, vectorstore_path: str = "vectorstore"):
        self.vectorstore_path = vectorstore_path
        self.chat_history: List[Dict[str, str]] = []

        self._initialize_embeddings()
        self._initialize_vectorstore()
        self._initialize_llm()

        logger.info("Cownselor RAG chain initialized successfully")

    def _initialize_embeddings(self):
        """embedding model"""
        logger.info("Loading embedding model for retrieval...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _initialize_vectorstore(self):
        """FAISS vector store"""
        vectorstore_dir = Path(self.vectorstore_path)

        if not vectorstore_dir.exists():
            logger.warning(f"Vector store not found at {self.vectorstore_path}")
            logger.warning(
                "Please run 'python ingest.py' first to create the knowledge base"
            )
            self.vectorstore = None
            return

        try:
            self.vectorstore = FAISS.load_local(
                self.vectorstore_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info("Vector store loaded successfully")

        except Exception as e:
            logger.error(f"Error loading vector store: {str(e)}")
            self.vectorstore = None

    def _initialize_llm(self):
        """init llm"""
        logger.info("Initializing Ollama LLM...")
        self.llm = OllamaLLM()

    def retrieve_relevant_docs(self, query: str, k: int = 5) -> List[Document]:
        """get relevant documents from vector store"""
        if not self.vectorstore:
            logger.warning("Vector store not available for retrieval")
            return []

        try:
            enhanced_query = self._enhance_query(query)
            docs = self.vectorstore.similarity_search(enhanced_query, k=k)

            logger.info(f"Retrieved {len(docs)} documents for query: {query[:50]}...")
            return docs

        except Exception as e:
            logger.error(f"Error during document retrieval: {str(e)}")
            return []

    def _enhance_query(self, query: str) -> str:
        # use keywords (improve performance by changing this probably)
        academic_keywords = {
            "class": "course class",
            "classes": "courses classes",
            "major": "major degree program",
            "graduation": "graduation requirements degree",
            "elective": "elective course recommendation",
            "prerequisite": "prerequisite requirement",
            "schedule": "schedule planning course load",
        }

        enhanced = query.lower()
        for key, expansion in academic_keywords.items():
            if key in enhanced:
                enhanced = enhanced.replace(key, expansion)

        return enhanced

    def format_context(self, docs: List[Document]) -> str:
        """context string gen"""
        if not docs:
            return "No relevant documents found in the knowledge base."

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "Unknown source")
            category = doc.metadata.get("category", "general")
            content = doc.page_content.strip()

            context_parts.append(f"📄 Source {i}: {source} ({category})\n{content}\n")

        return "\n" + "=" * 50 + "\n".join(context_parts) + "=" * 50

    def create_prompt(
        self, user_query: str, context: str, chat_history: List[Dict[str, str]]
    ) -> str:
        """complete prompt for the LLM"""

        # Format chat history
        history_text = ""
        if chat_history:
            history_text = "\n📜 CONVERSATION HISTORY:\n"
            for i, msg in enumerate(chat_history[-6:]):  # Last 6 messages for context
                role = "Student" if msg["role"] == "user" else "Cownselor"
                history_text += f"{role}: {msg['content']}\n"
            history_text += "\n"

        prompt = f"""{SYSTEM_PROMPT}

{history_text} 
RELEVANT KNOWLEDGE BASE CONTENT:
{context}

STUDENT QUESTION:
{user_query}

COWNSELOR RESPONSE:
Please provide a helpful, comprehensive response based on the knowledge base content above. If the information needed isn't in the knowledge base, be honest about this limitation and suggest appropriate next steps."""

        return prompt

    def generate_response(self, user_query: str) -> Tuple[str, List[Document]]:
        """generate response"""
        try:
            # get docs, format context, create prompt, generate response
            docs = self.retrieve_relevant_docs(user_query)
            context = self.format_context(docs)
            prompt = self.create_prompt(user_query, context, self.chat_history)
            response = self.llm.generate(prompt, temperature=0.7)

            # history update
            self.chat_history.append({"role": "user", "content": user_query})
            self.chat_history.append({"role": "assistant", "content": response})

            if len(self.chat_history) > 20:
                self.chat_history = self.chat_history[-20:]

            return response, docs

        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            error_response = "I apologize, but I encountered an error while processing your question. Please try rephrasing your question or check if the system is properly configured."
            return error_response, []

    def clear_history(self):
        """clear history"""
        self.chat_history = []
        logger.info("Chat history cleared")

    def get_system_status(self) -> Dict[str, Any]:
        status = {
            "vectorstore_available": self.vectorstore is not None,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": self.llm.model_name,
            "chat_history_length": len(self.chat_history),
            "vectorstore_path": self.vectorstore_path,
        }
        if self.vectorstore:
            try:
                status["documents_in_vectorstore"] = self.vectorstore.index.ntotal
            except:
                status["documents_in_vectorstore"] = "Unknown"

        return status


def main():
    # placeholder (testing)
    print("Testing Cownselor RAG Chain")
    print("=" * 40)

    try:
        counselor = Cownselor()

        test_query = "What courses do I need to take for a Computer Science major?"
        print(f"\nTest Query: {test_query}")

        response, docs = counselor.generate_response(test_query)

        print(f"\nResponse: {response}")
        print(f"\nRetrieved {len(docs)} documents")

        status = counselor.get_system_status()
        print(f"\nSystem Status:")
        for key, value in status.items():
            print(f"  • {key}: {value}")

    except Exception as e:
        print(f"Error testing RAG chain: {str(e)}")


if __name__ == "__main__":
    main()

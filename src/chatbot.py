import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.telemetry.metrics import tracker


class ChatBot:
    """
    A simple, stateful ChatBot baseline that uses LLMProvider directly.
    Does NOT use tools - only sends prompts to the LLM.

    This serves as a baseline to compare against the ReAct Agent.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        system_prompt: Optional[str] = None,
        max_history: int = 10,
    ):
        """
        Initialize the ChatBot.

        Args:
            llm_provider: An instance of LLMProvider (OpenAI, Gemini, Local, etc.)
            system_prompt: Optional system prompt to guide the LLM behavior
            max_history: Maximum conversation history to keep (for context window management)
        """
        self.llm_provider = llm_provider
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.session_start_time = datetime.utcnow()
        self.total_requests = 0
        self.total_tokens_used = 0

        logger.log_event(
            "CHATBOT_INITIALIZED",
            {
                "provider": llm_provider.__class__.__name__,
                "model": llm_provider.model_name,
                "system_prompt_length": len(self.system_prompt),
            },
        )

    def _default_system_prompt(self) -> str:
        """
        Returns a default system prompt for the ChatBot.
        """
        return (
            "You are a helpful, friendly, and knowledgeable customer support assistant. "
            "Answer questions clearly and concisely. If you don't know something, say so. "
            "Be professional and courteous."
        )

    def chat(self, user_message: str) -> Dict[str, Any]:
        """
        Send a message to the ChatBot and receive a response.

        Args:
            user_message: The user's input message

        Returns:
            Dict containing:
            - response: The LLM's response text
            - latency_ms: Time taken to generate response
            - tokens_used: Token usage (if available)
            - timestamp: When the response was generated
        """
        start_time = time.time()

        try:
            # Build conversation context
            context_messages = self._build_context()
            full_prompt = context_messages + f"\nUser: {user_message}\nAssistant:"

            # Call the LLM provider
            logger.log_event(
                "CHATBOT_REQUEST_START",
                {
                    "user_message_length": len(user_message),
                    "history_length": len(self.conversation_history),
                },
            )

            result = self.llm_provider.generate(
                prompt=full_prompt, system_prompt=self.system_prompt
            )

            latency_ms = (time.time() - start_time) * 1000
            response_text = result.get("content", "").strip()
            usage = result.get("usage", {})

            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append(
                {"role": "assistant", "content": response_text}
            )

            # Trim history if it exceeds max_history
            if len(self.conversation_history) > self.max_history * 2:
                self.conversation_history = self.conversation_history[
                    -(self.max_history * 2) :
                ]

            # Track metrics
            self.total_requests += 1
            total_tokens = usage.get("total_tokens", 0)
            self.total_tokens_used += total_tokens

            tracker.track_request(
                provider=self.llm_provider.__class__.__name__,
                model=self.llm_provider.model_name,
                usage=usage,
                latency_ms=int(latency_ms),
            )

            response_data = {
                "response": response_text,
                "latency_ms": latency_ms,
                "tokens_used": total_tokens,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "success",
            }

            logger.log_event("CHATBOT_REQUEST_SUCCESS", response_data)

            return response_data

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            error_msg = str(e)

            logger.error(f"ChatBot error: {error_msg}")
            logger.log_event(
                "CHATBOT_REQUEST_ERROR",
                {
                    "error": error_msg,
                    "latency_ms": latency_ms,
                    "user_message_length": len(user_message),
                },
            )

            return {
                "response": f"Sorry, I encountered an error: {error_msg}",
                "latency_ms": latency_ms,
                "tokens_used": 0,
                "timestamp": datetime.utcnow().isoformat(),
                "status": "error",
                "error": error_msg,
            }

    def _build_context(self) -> str:
        """
        Build conversation context from history.
        Formats the conversation history as a string for the prompt.
        """
        if not self.conversation_history:
            return ""

        context_lines = []
        for msg in self.conversation_history:
            role = msg["role"].capitalize()
            content = msg["content"]
            context_lines.append(f"{role}: {content}")

        return "\n".join(context_lines)

    def clear_history(self) -> None:
        """
        Clear the conversation history.
        Useful for starting a fresh conversation.
        """
        self.conversation_history = []
        logger.log_event("CHATBOT_HISTORY_CLEARED", {})

    def get_history(self) -> List[Dict[str, str]]:
        """
        Retrieve the current conversation history.
        """
        return self.conversation_history.copy()

    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics for this session.

        Returns:
            Dict containing:
            - total_requests: Number of LLM calls made
            - total_tokens_used: Total tokens consumed
            - session_duration_seconds: Time since ChatBot was initialized
            - avg_tokens_per_request: Average tokens per request
            - provider: The LLM provider being used
            - model: The model name
        """
        session_duration = (datetime.utcnow() - self.session_start_time).total_seconds()
        avg_tokens = (
            self.total_tokens_used / self.total_requests
            if self.total_requests > 0
            else 0
        )

        metrics = {
            "total_requests": self.total_requests,
            "total_tokens_used": self.total_tokens_used,
            "session_duration_seconds": round(session_duration, 2),
            "avg_tokens_per_request": round(avg_tokens, 2),
            "provider": self.llm_provider.__class__.__name__,
            "model": self.llm_provider.model_name,
            "history_length": len(self.conversation_history),
        }

        logger.log_event("CHATBOT_METRICS_SUMMARY", metrics)

        return metrics

    def set_system_prompt(self, new_prompt: str) -> None:
        """
        Update the system prompt for the ChatBot.
        """
        self.system_prompt = new_prompt
        logger.log_event(
            "CHATBOT_SYSTEM_PROMPT_UPDATED",
            {"new_prompt_length": len(new_prompt)},
        )

    def get_session_info(self) -> Dict[str, Any]:
        """
        Get detailed session information.
        """
        return {
            "provider": self.llm_provider.__class__.__name__,
            "model": self.llm_provider.model_name,
            "session_start": self.session_start_time.isoformat(),
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens_used,
            "history_messages": len(self.conversation_history),
            "max_history": self.max_history,
        }

from snowflake.snowpark import Session

import json
from typing import Any, Dict, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

SUPPORTED_ROLES: List[str] = [
    "system",
    "user",
    "assistant",
]


class ChatSnowflakeCortexError(Exception):
    """Error with Snowpark client."""


def _convert_message_to_dict(message: BaseMessage) -> dict:
    message_dict: Dict[str, Any] = {
        "content": message.content.replace("'", '"'),
    }

    if isinstance(message, ChatMessage) and message.role in SUPPORTED_ROLES:
        message_dict["role"] = message.role
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _truncate_at_stop_tokens(
        text: str,
        stop: Optional[List[str]],
) -> str:
    """Truncates text at the earliest stop token found."""
    if stop is None:
        return text

    for stop_token in stop:
        stop_token_idx = text.find(stop_token)
        if stop_token_idx != -1:
            text = text[:stop_token_idx]
    return text


class ChatSnowflakeCortex(BaseChatModel):
    session_builder_conf: dict = {}
    model: str = "llama3.1-8b"
    cortex_function: str = "complete"
    temperature: float = 0.9

    @property
    def _llm_type(self) -> str:
        return f"snowflake-cortex-{self.model}"

    def _generate(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> ChatResult:
        message_dicts = [_convert_message_to_dict(m) for m in messages]
        message_str = str(message_dicts)
        options = {"temperature": self.temperature}
        options_str = str(options)
        sql_stmt = f"""
            select snowflake.cortex.{self.cortex_function}(
                '{self.model}'
                ,{message_str},{options_str}) as llm_response;"""

        session = Session.builder.configs(self.session_builder_conf).create()
        try:
            l_rows = session.sql(sql_stmt).collect()
        except Exception as e:
            raise ChatSnowflakeCortexError(
                f"Error while making request to Snowflake Cortex via Snowpark: {e}"
            )
        finally:
            session.close()

        response = json.loads(l_rows[0]["LLM_RESPONSE"])
        ai_message_content = response["choices"][0]["messages"]

        content = _truncate_at_stop_tokens(ai_message_content, stop)
        message = AIMessage(
            content=content,
            response_metadata=response["usage"],
        )
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

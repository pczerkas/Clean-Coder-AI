import json
import logging
import os
import queue
import threading
import time
from queue import Queue
from typing import Any, AsyncIterator, Dict, Generator, Iterator, List, Optional, Union
from urllib.parse import urlencode, urlparse, urlunparse

import psutil
import websocket
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel, SimpleChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator

logger = logging.getLogger(__name__)


class CodeiumChatModel(BaseChatModel):
    """TODO: Codeium Chat Model"""

    # workspace_id: str
    # client: Any = Field(default=None, exclude=True)  #: :meta private:
    # async_client: Any = Field(default=None, exclude=True)  #: :meta private:

    workspace_id: str = Field(default=None)
    client: Any = Field(default=None, exclude=True)  #: :meta private:
    streaming: bool = False
    request_timeout: int = 30
    # manager_dir: str = Field(default=None, exclude=True)  #: :meta private:
    # port_number: int = Field(default=None, exclude=True)  #: :meta private:
    # websocket_client: Any = Field(default=None, exclude=True)  #: :meta private:

    # API_SERVER_URL_ARGUMENT = "https://server.codeium.com"

    # responses: List[BaseMessage]
    responses: List[str]
    """List of responses to **cycle** through in order."""
    sleep: Optional[float] = None
    """Sleep time in seconds between responses."""
    i: int = 0
    """Internally incremented after every model invocation."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        self.client = _CodeiumChatClient(
            self.workspace_id,
        )

        # self.manager_dir = self._get_manager_dir()
        # self.port_number = self._find_port_number()
        # self.websocket_client = websocket

    @property
    def _llm_type(self) -> str:
        return "codeium-chat-model"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Top Level call"""
        # return ChatResult(generations=[ChatGeneration(message=messages[-1])])

        prompt = messages[0].content

        if self.streaming:
            completion = ""
            for chunk in self._stream(prompt, stop, run_manager, **kwargs):
                completion += chunk.text
            return completion
        completion = ""
        self.client.arun(
            [{"role": "user", "content": prompt}],
            self.streaming,
        )
        for content in self.client.subscribe(timeout=self.request_timeout):
            if "data" not in content:
                continue
            completion = content["data"]["content"]

        # return completion
        return ChatResult(generations=[ChatGeneration(message=completion)])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[CallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        for i_c, c in enumerate(response):
            if self.sleep is not None:
                time.sleep(self.sleep)
            if (
                self.error_on_chunk_number is not None
                and i_c == self.error_on_chunk_number
            ):
                raise Exception("Fake error")

            yield ChatGenerationChunk(message=AIMessageChunk(content=c))


class _CodeiumChatClient:
    API_SERVER_URL_ARGUMENT = "https://server.codeium.com"
    # workspace_id: str = Field(default=None)
    # manager_dir: str = Field(default=None, exclude=True)  #: :meta private:
    # port_number: int = Field(default=None, exclude=True)  #: :meta private:
    # websocket_client: Any = Field(default=None, exclude=True)  #: :meta private:

    def __init__(
        self,
        workspace_id: str,
        api_url: Optional[str] = None,
    ):
        self.workspace_id = workspace_id
        self.manager_dir = self._get_manager_dir()
        # self.port_number = self._find_port_number()
        self.port_number = 34767

        self.api_url = (
            "ws://127.0.0.1:%s/connect/chat" % self.port_number
            if not api_url
            else api_url
        )
        # self.api_url = 'ws://127.0.0.1:%s/connect/ide' % self.port_number if not api_url else api_url

        self.websocket_client = websocket
        self.queue: Queue[Dict] = Queue()

    def _get_manager_dir(self) -> str:
        for process in psutil.process_iter():
            try:
                if not process.cmdline():
                    continue
                cmdline = " ".join(process.cmdline())
                if (
                    "--api_server_url %s" % self.API_SERVER_URL_ARGUMENT in cmdline
                    and "--workspace_id %s" % self.workspace_id in cmdline
                ):
                    return process.cmdline()[
                        process.cmdline().index("--manager_dir") + 1
                    ]
            except psutil.NoSuchProcess:
                continue
            except psutil.ZombieProcess:
                continue

        raise Exception("Codeium manager directory not found")

    def _find_port_number(self) -> int:
        for filename in os.listdir(self.manager_dir):
            port_number = int(filename)
            file_path = os.path.join(self.manager_dir, filename)
            # file_mtime = os.path.getmtime(file_path)
            # if os.path.isfile(file_path) and port_number and file_mtime >= start_time:
            if os.path.isfile(file_path) and port_number:
                return port_number

        raise Exception("Codeium chat port number not found")

    @staticmethod
    def _create_url(api_url: str) -> str:
        """
        Generate a request url.
        """
        parsed_url = urlparse(api_url)
        params_dict = {}
        encoded_params = urlencode(params_dict)
        url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                encoded_params,
                parsed_url.fragment,
            )
        )

        return url

    def run(
        self,
        messages: List[Dict],
        streaming: bool = False,
    ) -> None:
        self.websocket_client.enableTrace(False)
        ws = self.websocket_client.WebSocketApp(
            _CodeiumChatClient._create_url(
                self.api_url,
            ),
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close,
            on_open=self.on_open,
        )
        ws.messages = messages  # type: ignore[attr-defined]
        ws.streaming = streaming  # type: ignore[attr-defined]
        ws.run_forever()

    def arun(
        self,
        messages: List[Dict],
        streaming: bool = False,
    ) -> threading.Thread:
        ws_thread = threading.Thread(
            target=self.run,
            args=(
                messages,
                streaming,
            ),
        )
        ws_thread.start()
        return ws_thread

    def on_error(self, ws: Any, error: Optional[Any]) -> None:
        self.queue.put({"error": error})
        ws.close()

    def on_close(self, ws: Any, close_status_code: int, close_reason: str) -> None:
        logger.debug(
            {
                "log": {
                    "close_status_code": close_status_code,
                    "close_reason": close_reason,
                }
            }
        )
        self.queue.put({"done": True})

    def on_open(self, ws: Any) -> None:
        self.blocking_message = {"content": "", "role": "assistant"}
        data = json.dumps(
            self.gen_params(
                messages=ws.messages,
            )
        )
        ws.send(data)

    def on_message(self, ws: Any, message: str) -> None:
        data = json.loads(message)
        code = data["header"]["code"]
        if code != 0:
            self.queue.put(
                {"error": f"Code: {code}, Error: {data['header']['message']}"}
            )
            ws.close()
        else:
            choices = data["payload"]["choices"]
            status = choices["status"]
            content = choices["text"][0]["content"]
            if ws.streaming:
                self.queue.put({"data": choices["text"][0]})
            else:
                self.blocking_message["content"] += content
            if status == 2:
                if not ws.streaming:
                    self.queue.put({"data": self.blocking_message})
                usage_data = (
                    data.get("payload", {}).get("usage", {}).get("text", {})
                    if data
                    else {}
                )
                self.queue.put({"usage": usage_data})
                ws.close()

    def gen_params(
        self,
        messages: list,
    ) -> dict:
        data: Dict = {
            "payload": {"message": {"text": messages}},
        }

        logger.debug(f"Request Parameters: {data}")

        return data

    def subscribe(self, timeout: Optional[int] = 30) -> Generator[Dict, None, None]:
        while True:
            try:
                content = self.queue.get(timeout=timeout)
            except queue.Empty as _:
                raise TimeoutError(
                    f"SparkLLMClient wait LLM api response timeout {timeout} seconds"
                )
            if "error" in content:
                raise ConnectionError(content["error"])
            if "usage" in content:
                yield content
                continue
            if "done" in content:
                break
            if "data" not in content:
                break
            yield content


# class FakeMessagesListChatModel(BaseChatModel):
#     """Fake ChatModel for testing purposes."""

#     responses: List[BaseMessage]
#     """List of responses to **cycle** through in order."""
#     sleep: Optional[float] = None
#     """Sleep time in seconds between responses."""
#     i: int = 0
#     """Internally incremented after every model invocation."""

#     def _generate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         response = self.responses[self.i]
#         if self.i < len(self.responses) - 1:
#             self.i += 1
#         else:
#             self.i = 0
#         generation = ChatGeneration(message=response)
#         return ChatResult(generations=[generation])

#     @property
#     def _llm_type(self) -> str:
#         return "fake-messages-list-chat-model"


# class FakeListChatModel(SimpleChatModel):
#     """Fake ChatModel for testing purposes."""

#     responses: List[str]
#     """List of responses to **cycle** through in order."""
#     sleep: Optional[float] = None
#     i: int = 0
#     """List of responses to **cycle** through in order."""
#     error_on_chunk_number: Optional[int] = None
#     """Internally incremented after every model invocation."""

#     @property
#     def _llm_type(self) -> str:
#         return "fake-list-chat-model"

#     def _call(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         """First try to lookup in queries, else return 'foo' or 'bar'."""
#         response = self.responses[self.i]
#         if self.i < len(self.responses) - 1:
#             self.i += 1
#         else:
#             self.i = 0
#         return response

#     def _stream(
#         self,
#         messages: List[BaseMessage],
#         stop: Union[List[str], None] = None,
#         run_manager: Union[CallbackManagerForLLMRun, None] = None,
#         **kwargs: Any,
#     ) -> Iterator[ChatGenerationChunk]:
#         response = self.responses[self.i]
#         if self.i < len(self.responses) - 1:
#             self.i += 1
#         else:
#             self.i = 0
#         for i_c, c in enumerate(response):
#             if self.sleep is not None:
#                 time.sleep(self.sleep)
#             if (
#                 self.error_on_chunk_number is not None
#                 and i_c == self.error_on_chunk_number
#             ):
#                 raise Exception("Fake error")

#             yield ChatGenerationChunk(message=AIMessageChunk(content=c))

#     async def _astream(
#         self,
#         messages: List[BaseMessage],
#         stop: Union[List[str], None] = None,
#         run_manager: Union[AsyncCallbackManagerForLLMRun, None] = None,
#         **kwargs: Any,
#     ) -> AsyncIterator[ChatGenerationChunk]:
#         response = self.responses[self.i]
#         if self.i < len(self.responses) - 1:
#             self.i += 1
#         else:
#             self.i = 0
#         for i_c, c in enumerate(response):
#             if self.sleep is not None:
#                 await asyncio.sleep(self.sleep)
#             if (
#                 self.error_on_chunk_number is not None
#                 and i_c == self.error_on_chunk_number
#             ):
#                 raise Exception("Fake error")
#             yield ChatGenerationChunk(message=AIMessageChunk(content=c))

#     @property
#     def _identifying_params(self) -> Dict[str, Any]:
#         return {"responses": self.responses}


# class FakeChatModel(SimpleChatModel):
#     """Fake Chat Model wrapper for testing purposes."""

#     def _call(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> str:
#         return "fake response"

#     async def _agenerate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         output_str = "fake response"
#         message = AIMessage(content=output_str)
#         generation = ChatGeneration(message=message)
#         return ChatResult(generations=[generation])

#     @property
#     def _llm_type(self) -> str:
#         return "fake-chat-model"

#     @property
#     def _identifying_params(self) -> Dict[str, Any]:
#         return {"key": "fake"}


# class GenericFakeChatModel(BaseChatModel):
#     """Generic fake chat model that can be used to test the chat model interface.

#     * Chat model should be usable in both sync and async tests
#     * Invokes on_llm_new_token to allow for testing of callback related code for new
#       tokens.
#     * Includes logic to break messages into message chunk to facilitate testing of
#       streaming.
#     """

#     messages: Iterator[Union[AIMessage, str]]
#     """Get an iterator over messages.

#     This can be expanded to accept other types like Callables / dicts / strings
#     to make the interface more generic if needed.

#     Note: if you want to pass a list, you can use `iter` to convert it to an iterator.

#     Please note that streaming is not implemented yet. We should try to implement it
#     in the future by delegating to invoke and then breaking the resulting output
#     into message chunks.
#     """

#     def _generate(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> ChatResult:
#         """Top Level call"""
#         message = next(self.messages)
#         if isinstance(message, str):
#             message_ = AIMessage(content=message)
#         else:
#             message_ = message
#         generation = ChatGeneration(message=message_)
#         return ChatResult(generations=[generation])

#     def _stream(
#         self,
#         messages: List[BaseMessage],
#         stop: Optional[List[str]] = None,
#         run_manager: Optional[CallbackManagerForLLMRun] = None,
#         **kwargs: Any,
#     ) -> Iterator[ChatGenerationChunk]:
#         """Stream the output of the model."""
#         chat_result = self._generate(
#             messages, stop=stop, run_manager=run_manager, **kwargs
#         )
#         if not isinstance(chat_result, ChatResult):
#             raise ValueError(
#                 f"Expected generate to return a ChatResult, "
#                 f"but got {type(chat_result)} instead."
#             )

#         message = chat_result.generations[0].message

#         if not isinstance(message, AIMessage):
#             raise ValueError(
#                 f"Expected invoke to return an AIMessage, "
#                 f"but got {type(message)} instead."
#             )

#         content = message.content

#         if content:
#             # Use a regular expression to split on whitespace with a capture group
#             # so that we can preserve the whitespace in the output.
#             assert isinstance(content, str)
#             content_chunks = cast(List[str], re.split(r"(\s)", content))

#             for token in content_chunks:
#                 chunk = ChatGenerationChunk(
#                     message=AIMessageChunk(content=token, id=message.id)
#                 )
#                 if run_manager:
#                     run_manager.on_llm_new_token(token, chunk=chunk)
#                 yield chunk

#         if message.additional_kwargs:
#             for key, value in message.additional_kwargs.items():
#                 # We should further break down the additional kwargs into chunks
#                 # Special case for function call
#                 if key == "function_call":
#                     for fkey, fvalue in value.items():
#                         if isinstance(fvalue, str):
#                             # Break function call by `,`
#                             fvalue_chunks = cast(List[str], re.split(r"(,)", fvalue))
#                             for fvalue_chunk in fvalue_chunks:
#                                 chunk = ChatGenerationChunk(
#                                     message=AIMessageChunk(
#                                         id=message.id,
#                                         content="",
#                                         additional_kwargs={
#                                             "function_call": {fkey: fvalue_chunk}
#                                         },
#                                     )
#                                 )
#                                 if run_manager:
#                                     run_manager.on_llm_new_token(
#                                         "",
#                                         chunk=chunk,  # No token for function call
#                                     )
#                                 yield chunk
#                         else:
#                             chunk = ChatGenerationChunk(
#                                 message=AIMessageChunk(
#                                     id=message.id,
#                                     content="",
#                                     additional_kwargs={"function_call": {fkey: fvalue}},
#                                 )
#                             )
#                             if run_manager:
#                                 run_manager.on_llm_new_token(
#                                     "",
#                                     chunk=chunk,  # No token for function call
#                                 )
#                             yield chunk
#                 else:
#                     chunk = ChatGenerationChunk(
#                         message=AIMessageChunk(
#                             id=message.id, content="", additional_kwargs={key: value}
#                         )
#                     )
#                     if run_manager:
#                         run_manager.on_llm_new_token(
#                             "",
#                             chunk=chunk,  # No token for function call
#                         )
#                     yield chunk

#     @property
#     def _llm_type(self) -> str:
#         return "generic-fake-chat-model"


# class ParrotFakeChatModel(BaseChatModel):

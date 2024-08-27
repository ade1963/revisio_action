"""
title: Revisio Action
author: Dmitry Andreev
author_url: https://github.com/ade1963/revisio_action
funding_url: https://github.com/open-webui
version: 0.0.2
required_open_webui_version: 0.3.9
"""
# The LLM agent evaluates the previous response and only offers a new one if it is significantly superior; otherwise, it replies with "NO IMPROVEMENTS."
# The idea is taken from https://www.reddit.com/r/LocalLLaMA/s/btTU6CNl3c, by: https://www.reddit.com/u/GeneriAcc/s/hrDwJkLYLb

from pydantic import BaseModel, Field
from typing import Optional, List, Callable, Awaitable, Union, Generator, Iterator
import requests
from requests.exceptions import HTTPError
import aiohttp

class Action:
    class Valves(BaseModel):
        openai_api_url: str = Field(
            default="http://localhost:11434/v1",
            description="Ollama compartibel Open AI API",
        )
        models: List[str] = Field(
            default=["llama3.1:8b-instruct-q8_0"],
            description="List of comma-separated models for self-reflecting.",
        )
        critic_prompt: str = Field(
            default="Evaluate the previous response. If and only if you can provide an objectively superior answer that is significantly more accurate, comprehensive, or helpful, present only the improved response without any introductory statements. The threshold for improvement is extremely high - minor enhancements or rephrasing do not qualify. In the vast majority of cases, simply state 'NO IMPROVEMENTS'. Only in rare instances where the original response is clearly inadequate or incorrect should you offer an alternative answer.",
            description="Critic's prompt",
        )
        stop_word: str = Field(
            default="NO IMPROVEMENTS",
            description="Stop expression to stop iterations",
        )

    def __init__(self):
        self.valves = self.Valves()
        pass

    async def action(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
    ) -> Optional[dict]:

        if not __event_emitter__:
            return None

        # We expect messages[0] - User request, messages[1] - assistans response
        if len(body["messages"]) < 2:
            await __event_emitter__( {"type": "status", "data": {"description": "Fail: chat too short", "done": True}})
            return None

        # await __event_emitter__( {"type": "status", "data": {"description": "Revisio started...", "done": False}})
        user_prompt = body["messages"][0]["content"]
        prev_response = body["messages"][1]["content"]
        try:
            for i, model in enumerate (self.valves.models):
                await __event_emitter__( 
                    {"type": "status", "data": {"description": f"Self-reflecting, iter: {i+1}/{len(self.valves.models)}, model: {model}", "done": False}}
                    )

                response = await self.query_openai_api(
                    model, user_prompt, prev_response, self.valves.critic_prompt
                )
                
                if self.valves.stop_word in response and len(response) <= (len(self.valves.stop_word)+4):
                    await __event_emitter__( {"type": "status", "data": {"description": f"Received stop word: {response}", "done": True}})
                    return body
                
                body["messages"][1]["content"] = response
                prev_response = response
        except Exception as e:
            await __event_emitter__({"type": "info", "data": {"description": str(e), "done": True}})
            return None
        
        await __event_emitter__({"type": "status", "data": {"description": "Revision complete", "done": True}})
        return body
    
    async def query_openai_api(
        self,
        model: str,
        prompt: str,
        prev_response: str,
        critic_prompt: str,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> str:
        url = f"{self.valves.openai_api_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prev_response},
                {"role": "user", "content": critic_prompt},
            ],
        }
        try:
            async with aiohttp.ClientSession() as session:
                response = await session.post(url, headers=headers, json=payload)
                response.raise_for_status()
                json = await response.json()
            return json["choices"][0]["message"]["content"]
        except HTTPError as e:
            raise Exception(f"Http error: {e.response.text}")

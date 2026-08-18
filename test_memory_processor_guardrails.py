"""记忆处理器结构护栏回归测试。"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.features.recall.processors.memory_processor import MemoryProcessor
from core.shared.contracts.conversation import Message


def _processor(response_text: str) -> MemoryProcessor:
    """构造返回固定结构化响应的记忆处理器。"""

    context = MagicMock()
    context.get_using_provider.return_value = None
    context.persona_manager = None
    context.get_registered_llm_tools.return_value = []
    provider = MagicMock()
    provider.text_chat = AsyncMock(
        return_value=MagicMock(completion_text=response_text)
    )
    return MemoryProcessor(context=context, llm_provider=provider)


def _messages() -> list[Message]:
    """构造具有完整可引用正文的私聊消息。"""

    return [
        Message(
            id=1,
            session_id="session-1",
            role="user",
            content="阿明喜欢喝咖啡。",
            sender_id="user-1",
            sender_name="阿明",
        )
    ]


@pytest.mark.asyncio
async def test_processor_accepts_sixteen_prompt_source_references() -> None:
    """门禁允许的 16 条来源引用应通过结构护栏。"""

    source = "阿明喜欢喝咖啡。"
    response = json.dumps(
        {
            "memories": [
                {
                    "summary": "阿明喜欢喝咖啡。",
                    "topics": ["咖啡偏好"],
                    "key_facts": ["阿明喜欢喝咖啡"],
                    "sentiment": "positive",
                    "importance": 0.8,
                    "source_refs": [
                        {"message_index": 0, "start": 0, "end": len(source)}
                        for _ in range(16)
                    ],
                }
            ]
        }
    )

    results = await _processor(response).process_conversation(_messages())

    assert len(results) == 1
    assert results[0]["metadata"]["guardrails_validated"] is True


@pytest.mark.asyncio
async def test_processor_fallback_preserves_nested_memory_contract() -> None:
    """结构护栏回退后仍应保留 Prompt 的 memories[] 候选字段。"""

    response = json.dumps(
        {
            "memories": [
                {
                    "summary": "我记住了阿明喜欢喝咖啡。",
                    "topics": ["咖啡偏好"],
                    "key_facts": ["阿明喜欢喝咖啡"],
                    "sentiment": "positive",
                    "importance": 0.8,
                    "atom_type": "invalid",
                }
            ]
        }
    )

    results = await _processor(response).process_conversation(_messages())

    assert len(results) == 1
    assert results[0]["metadata"]["summary_quality"] == "normal"
    assert results[0]["metadata"]["guardrail_fallback"] is True
    assert results[0]["metadata"]["key_facts"] == ["阿明喜欢喝咖啡"]

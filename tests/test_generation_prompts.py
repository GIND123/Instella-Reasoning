"""Chat-template formatting for generation (the fix for degenerate Instruct output).

These are hermetic: they use a stub tokenizer so no torch/model download is needed.
They lock in the two behaviours that matter for research validity — an instruction
model's prompt is wrapped in its chat template (and special tokens are not doubled),
and a base model with no template falls back to the raw prompt.
"""
from instella_reasoning.evaluation import _format_prompts


class _ChatTokenizer:
    """Minimal stand-in for an Instruct tokenizer exposing a chat template."""

    chat_template = "{{ messages }}"  # presence is what _format_prompts checks

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is True
        content = messages[0]["content"]
        return f"<|user|>\n{content}\n<|assistant|>\n"


class _BaseTokenizer:
    """A base-model tokenizer with no chat template."""

    chat_template = None


def test_chat_template_is_applied_for_instruct_models() -> None:
    texts, add_special, used = _format_prompts(_ChatTokenizer(), ["2+2?"], use_chat_template=True)
    assert used is True
    # Template already carries the special tokens, so the caller must not add BOS again.
    assert add_special is False
    assert texts == ["<|user|>\n2+2?\n<|assistant|>\n"]


def test_base_model_falls_back_to_raw_prompt() -> None:
    texts, add_special, used = _format_prompts(_BaseTokenizer(), ["2+2?"], use_chat_template=True)
    assert used is False
    assert add_special is True  # base model wants a normal BOS
    assert texts == ["2+2?"]


def test_chat_template_can_be_disabled() -> None:
    texts, add_special, used = _format_prompts(_ChatTokenizer(), ["2+2?"], use_chat_template=False)
    assert used is False
    assert add_special is True
    assert texts == ["2+2?"]

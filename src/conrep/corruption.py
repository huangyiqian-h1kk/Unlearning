"""Pure helpers for ConRep's historical token-swap text convention."""

from __future__ import annotations

from dataclasses import dataclass


TOKEN_SWAP_SEPARATOR = "!@#$%^&*()"


@dataclass(frozen=True)
class TokenSwapText:
    """The model-visible text and the span used as a corruption target."""

    model_text: str
    replacement_span: str
    annotated: bool


def parse_token_swap(text: str) -> TokenSwapText:
    """Parse the separator convention used by the preserved trainer.

    Historical code removes the separator before tokenization and treats the
    text immediately after the first separator as the replacement span.
    """

    pieces = text.split(TOKEN_SWAP_SEPARATOR)
    if len(pieces) == 1:
        return TokenSwapText(text, "", False)
    return TokenSwapText("".join(pieces), pieces[1], True)


def annotate_token_swap(prefix: str, replacement_span: str, suffix: str = "") -> str:
    """Create a historical token-swap annotation."""

    if TOKEN_SWAP_SEPARATOR in prefix + replacement_span + suffix:
        raise ValueError("token-swap fields may not contain the separator")
    annotated = prefix + TOKEN_SWAP_SEPARATOR + replacement_span
    if suffix:
        annotated += TOKEN_SWAP_SEPARATOR + suffix
    return annotated

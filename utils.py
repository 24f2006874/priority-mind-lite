"""
PriorityMind-Lite: Shared Utility Functions
============================================
Common functions used across multiple modules to avoid code duplication.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import Observation

# ============================================================================
# Text Processing Utilities
# ============================================================================


def contains_any(text: str, keywords: set[str]) -> bool:
    """Check if text contains any of the given keywords.

    Args:
        text: The text to search in
        keywords: Set of keywords to search for

    Returns:
        True if any keyword is found in the text
    """
    normalized_text = normalize_text(text)
    for keyword in keywords:
        normalized_keyword = normalize_text(keyword)
        if not normalized_keyword:
            continue

        # Use boundary-aware matching for single-token keywords to avoid false
        # positives like matching "app" inside "happened".
        if " " not in normalized_keyword:
            escaped = re.escape(normalized_keyword)
            if re.fullmatch(r"\w+", normalized_keyword):
                pattern = rf"\b{escaped}\b"
            else:
                # For punctuated tokens like "now!" use non-word boundaries.
                pattern = rf"(?<!\w){escaped}(?!\w)"
            if re.search(pattern, normalized_text):
                return True
        elif normalized_keyword in normalized_text:
            return True
    return False


def normalize_text(value: str | None) -> str:
    """Normalize text by lowercasing and collapsing whitespace.

    Args:
        value: The text to normalize

    Returns:
        Normalized text (lowercase, single spaces)
    """
    if not value:
        return ""
    return " ".join(value.lower().split())


# ============================================================================
# Ticket Classification Utilities
# ============================================================================

# Complaint indicators - highest priority for severe issues
COMPLAINT_KEYWORDS: set[str] = {
    "unacceptable",
    "deleted my data",
    "data breach",
    "account hacked",
    "terrible service",
    "transferred 5 times",
    "service down",
    "losing business",
    "compensation",
    "want compensation",
    "not responding fast enough",
    "exposed in your data breach",
    "haven't heard from you",
}

# Billing indicators
BILLING_KEYWORDS: set[str] = {
    "bill",
    "refund",
    "charge",
    "charged",
    "invoice",
    "fee",
    "fees",
    "paid",
    "payment",
    "subscription",
    "overcharge",
    "overcharged",
    "monthly fee",
    "charges on",
    "charged twice",
    "payment failed",
    "why did my",
    "increase without notice",
    "explain the charges",
}

# Technical indicators
TECHNICAL_KEYWORDS: set[str] = {
    "crash",
    "crashes",
    "crashing",
    "app",
    "freeze",
    "freezes",
    "frozen",
    "error",
    "bug",
    "bugs",
    "not working",
    "stopped working",
    "broken",
    "login page",
    "upload",
    "notification",
    "notifications",
    "search function",
    "battery",
    "drains",
    "slow",
    "lag",
    "lagging",
    "update",
    "updated",
    "error 500",
    "no results",
    "returns no results",
}


def infer_ticket_category(ticket_text: str) -> str:
    """Infer ticket category from text using keyword matching.

    Priority order:
    1. Complaint: Contains complaint indicators (unacceptable, terrible, hacked, breach, etc.)
       OR contains billing/technical issues with strong negative sentiment indicators
    2. Billing: Contains billing indicators (bill, refund, charge, invoice, fee, etc.)
    3. Technical: Contains technical indicators (crash, app, error, freeze, etc.)
    4. General: Everything else

    Args:
        ticket_text: The customer support ticket text

    Returns:
        One of: "billing", "technical", "general", "complaint"
    """
    text = normalize_text(ticket_text)

    # Complaint indicators - highest priority for severe issues
    if contains_any(text, COMPLAINT_KEYWORDS):
        return "complaint"

    # Billing with data/security issue becomes complaint
    if contains_any(text, {"bill", "refund"}) and contains_any(text, {"data", "breach", "exposed"}):
        return "complaint"

    # Strong negative sentiment with billing/technical issues -> complaint
    complaint_sentiment_keywords = {
        "now!", "unacceptable", "terrible", "awful", "worst",
        "ridiculous", "outrageous", "disgusting", "incompetent",
    }
    has_billing = contains_any(text, BILLING_KEYWORDS)
    has_technical = contains_any(text, TECHNICAL_KEYWORDS)
    has_complaint_sentiment = contains_any(text, complaint_sentiment_keywords)

    if (has_billing or has_technical) and has_complaint_sentiment:
        return "complaint"

    # Billing indicators
    if has_billing:
        return "billing"

    # Technical indicators
    if has_technical:
        return "technical"

    return "general"


def infer_ticket_category_simple(ticket_text: str) -> str:
    """Simplified ticket category inference for quick checks.

    Uses a reduced set of keywords for faster classification.

    Args:
        ticket_text: The customer support ticket text

    Returns:
        One of: "billing", "technical", "general", "complaint"
    """
    text = normalize_text(ticket_text)

    if contains_any(text, {"unacceptable"}) or (contains_any(text, {"refund"}) and contains_any(text, {"data"})):
        return "complaint"
    if contains_any(text, {"bill", "refund"}):
        return "billing"
    if contains_any(text, {"crash", "app"}):
        return "technical"
    return "general"


# ============================================================================
# Formatting Utilities
# ============================================================================


def normalize_partial_signal(value: float | str | None) -> float | None:
    """Normalize a partial signal value to the [0.0, 1.0] range.

    Handles legacy values that may be on a 0-10 scale (values > 1.0
    are divided by 10). Returns None for unparseable inputs.

    Args:
        value: The raw signal value (float, string, or None)

    Returns:
        Normalized float in [0.0, 1.0], or None if unparseable.
    """
    if value is None or (isinstance(value, str) and value == "N/A"):
        return None
    try:
        normalized = float(value)
    except (ValueError, TypeError):
        return None
    if normalized > 1.0:
        normalized /= 10.0
    return max(0.0, min(1.0, normalized))


def format_partial_signal(value: float | str | None, suffix: str = "/1.0") -> str:
    """Format a partial signal value for display.

    Normalizes values to [0.0, 1.0] before formatting. If the value
    cannot be parsed or is None, returns "N/A".

    Args:
        value: The value to format (float, string, or None)
        suffix: Suffix to append (default: "/1.0")

    Returns:
        Formatted string representation, e.g. "0.85/1.0" or "N/A"
    """
    normalized = normalize_partial_signal(value)
    if normalized is None:
        return "N/A"
    return f"{normalized:.2f}{suffix}"


def clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp a value between lower and upper bounds.

    Args:
        value: The value to clamp
        lower: Lower bound (default: 0.0)
        upper: Upper bound (default: 1.0)

    Returns:
        Clamped value
    """
    return max(lower, min(upper, value))
"""Currency and Indian numbering formatting utility.

Implements standard Indian numbering system (Lakhs and Crores),
commas placed according to Indian conventions (e.g. 1,25,000.00),
and INR (₹) symbol prefixing with proper sign placement (-₹1,250.50).
"""

from typing import Union


def format_indian_number(value: Union[float, int], decimal_places: int = 2) -> str:
    """Formats a number using the Indian numbering system (e.g. 12,34,567.89).

    In the Indian numbering system:
    - The first separator is placed after 3 digits from the right.
    - Subsequent separators are placed after every 2 digits.

    Args:
        value: Numeric value to format.
        decimal_places: Number of decimal digits to preserve.

    Returns:
        Formatted string without currency symbol.
    """
    if value is None:
        return "N/A"

    try:
        val = float(value)
    except (ValueError, TypeError):
        return str(value)

    is_negative = val < 0
    val = abs(val)

    # Format to required decimal places
    format_str = f"{{:.{decimal_places}f}}"
    formatted = format_str.format(val)
    parts = formatted.split(".")
    integer_part = parts[0]
    decimal_part = f".{parts[1]}" if len(parts) > 1 and decimal_places > 0 else ""

    if len(integer_part) <= 3:
        formatted_int = integer_part
    else:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        groups = []
        while len(remaining) > 2:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        if remaining:
            groups.insert(0, remaining)
        formatted_int = ",".join(groups) + "," + last_three

    sign = "-" if is_negative else ""
    return f"{sign}{formatted_int}{decimal_part}"


def format_inr(value: Union[float, int], decimal_places: int = 2) -> str:
    """Formats numeric value as Indian Rupees (INR) with the ₹ symbol and Indian commas.

    Properly formats negative numbers as -₹X,XXX.XX rather than ₹-X,XXX.XX.

    Examples:
        format_inr(1250.5) -> "₹1,250.50"
        format_inr(-1250.5) -> "-₹1,250.50"
        format_inr(25000) -> "₹25,000.00"
        format_inr(125000) -> "₹1,25,000.00"
        format_inr(10000000) -> "₹1,00,00,000.00"
    """
    if value is None:
        return "₹N/A"

    try:
        val = float(value)
    except (ValueError, TypeError):
        return f"₹{value}"

    is_negative = val < 0
    formatted_abs = format_indian_number(abs(val), decimal_places)
    sign = "-" if is_negative else ""
    return f"{sign}₹{formatted_abs}"


def format_inr_compact(value: Union[float, int]) -> str:
    """Formats value into Lakhs (L) and Crores (Cr) with INR symbol.

    Examples:
        format_inr_compact(15000000) -> "₹1.50 Cr"
        format_inr_compact(-15000000) -> "-₹1.50 Cr"
        format_inr_compact(250000) -> "₹2.50 L"
        format_inr_compact(1250.5) -> "₹1,250.50"
    """
    if value is None:
        return "₹N/A"

    try:
        val = float(value)
    except (ValueError, TypeError):
        return f"₹{value}"

    is_negative = val < 0
    abs_val = abs(val)
    sign = "-" if is_negative else ""

    if abs_val >= 10_000_000:  # 1 Crore = 10 Million
        return f"{sign}₹{abs_val / 10_000_000:.2f} Cr"
    elif abs_val >= 100_000:   # 1 Lakh = 100 Thousand
        return f"{sign}₹{abs_val / 100_000:.2f} L"
    else:
        return format_inr(value, decimal_places=2)


def format_indian_volume(volume: Union[float, int]) -> str:
    """Formats share trading volume using Indian notations (Lakhs / Crores).

    Examples:
        18,500,000 -> "1.85 Cr shares"
        450,000 -> "4.50 L shares"
        50,000 -> "50,000 shares"
    """
    if volume is None:
        return "N/A"

    try:
        vol = float(volume)
    except (ValueError, TypeError):
        return f"{volume} shares"

    is_negative = vol < 0
    abs_vol = abs(vol)
    sign = "-" if is_negative else ""

    if abs_vol >= 10_000_000:
        return f"{sign}{abs_vol / 10_000_000:.2f} Cr shares"
    elif abs_vol >= 100_000:
        return f"{sign}{abs_vol / 100_000:.2f} L shares"
    else:
        return f"{sign}{format_indian_number(abs_vol, decimal_places=0)} shares"

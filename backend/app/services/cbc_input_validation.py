from collections.abc import Iterable
from decimal import Decimal
from typing import Protocol

from app.core.exceptions import DomainValidationError
from app.core.indicator_input_limits import get_indicator_input_limit


class IndicatorValuePayload(Protocol):
    indicator_code: str
    raw_value: Decimal


def validate_cbc_input_limits(values: Iterable[IndicatorValuePayload]) -> None:
    errors: list[dict[str, object]] = []

    for item in values:
        limit = get_indicator_input_limit(item.indicator_code)
        if limit is None:
            continue

        if item.raw_value < limit.min_allowed or item.raw_value > limit.max_allowed:
            errors.append(
                {
                    "indicator_code": item.indicator_code,
                    "provided_value": _json_number(item.raw_value),
                    "min_allowed": _json_number(limit.min_allowed),
                    "max_allowed": _json_number(limit.max_allowed),
                    "message": (
                        f"{item.indicator_code} must be between "
                        f"{limit.min_allowed} and {limit.max_allowed} {limit.unit}."
                    ),
                }
            )

    if errors:
        raise DomainValidationError(
            {
                "message": "One or more CBC indicator values are outside allowed input limits.",
                "errors": errors,
            }
        )


def _json_number(value: Decimal) -> int | float:
    if value == value.to_integral_value():
        return int(value)
    return float(value)

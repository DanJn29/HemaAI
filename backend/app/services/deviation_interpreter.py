from decimal import Decimal

from app.models.deviation_state import DeviationState
from app.models.enums import DeviationFamily
from app.models.reference_range import ReferenceRange


class DeviationInterpreter:
    def __init__(self, deviation_state_map: dict[str, DeviationState]) -> None:
        self.deviation_state_map = deviation_state_map

    @staticmethod
    def get_family(deviation_state_code: str) -> DeviationFamily:
        if deviation_state_code.endswith("low"):
            return DeviationFamily.LOW
        if deviation_state_code.endswith("high"):
            return DeviationFamily.HIGH
        return DeviationFamily.NORMAL

    def interpret(self, reference_range: ReferenceRange, raw_value: Decimal) -> tuple[DeviationState, float]:
        if reference_range.normal_min <= raw_value <= reference_range.normal_max:
            return self.deviation_state_map["normal"], 0.0

        if raw_value < reference_range.normal_min:
            if raw_value <= reference_range.severe_low_threshold:
                code = "severe_low"
            elif raw_value <= reference_range.moderate_low_threshold:
                code = "moderate_low"
            else:
                code = "mild_low"
            return self.deviation_state_map[code], self._normalized_score(reference_range, raw_value)

        if raw_value >= reference_range.severe_high_threshold:
            code = "severe_high"
        elif raw_value >= reference_range.moderate_high_threshold:
            code = "moderate_high"
        else:
            code = "mild_high"
        return self.deviation_state_map[code], self._normalized_score(reference_range, raw_value)

    @staticmethod
    def _normalized_score(reference_range: ReferenceRange, raw_value: Decimal) -> float:
        width = reference_range.normal_max - reference_range.normal_min
        if width <= 0:
            return 0.0
        if raw_value < reference_range.normal_min:
            delta = raw_value - reference_range.normal_min
        elif raw_value > reference_range.normal_max:
            delta = raw_value - reference_range.normal_max
        else:
            delta = Decimal("0")
        return round(float(delta / width), 4)


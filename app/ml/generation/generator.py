from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import numpy as np
from sqlalchemy.orm import Session

from app.ml.generation.evaluator import RuleEngineContext
from app.ml.generation.profiles import (
    ARCHETYPES,
    CLASS_PROFILES,
    DEFAULT_ARCHETYPE_WEIGHTS,
    DEFAULT_SIGNAL_STRENGTH_WEIGHTS,
    SIGNAL_STRENGTHS,
    ClassProfile,
)
from app.ml.types import SyntheticCase

DECIMAL_PRECISION = Decimal("0.001")
DEFAULT_AGE_BUCKET_WEIGHTS = {
    "18-40": 1 / 3,
    "41-65": 1 / 3,
    "66-120": 1 / 3,
}
DEFAULT_QUALITY_TARGETS = {
    "GOOD": 0.70,
    "AMBIGUOUS": 0.20,
    "BAD": 0.08,
}
SAMPLING_STYLES = (
    "interior",
    "near_normal_boundary",
    "near_far_boundary",
    "edge_normal",
)
AGE_BUCKETS = {
    "18-40": (18, 40),
    "41-65": (41, 65),
    "66-120": (66, 120),
}
INDICATOR_SEVERE_CAPS = {
    "WBC": {"low": Decimal("0"), "high": Decimal("50")},
    "RBC": {"low": Decimal("0"), "high": Decimal("8")},
    "HGB": {"low": Decimal("0"), "high": Decimal("220")},
    "HCT": {"low": Decimal("0"), "high": Decimal("0.700")},
    "MCV": {"low": Decimal("40"), "high": Decimal("130")},
    "MCH": {"low": Decimal("10"), "high": Decimal("45")},
    "MCHC": {"low": Decimal("200"), "high": Decimal("420")},
    "PLT": {"low": Decimal("0"), "high": Decimal("1000")},
    "RDW": {"low": Decimal("5"), "high": Decimal("25")},
    "NEU": {"low": Decimal("0"), "high": Decimal("30")},
    "LYM": {"low": Decimal("0"), "high": Decimal("15")},
    "MONO": {"low": Decimal("0"), "high": Decimal("3")},
    "EOS": {"low": Decimal("0"), "high": Decimal("3")},
    "BASO": {"low": Decimal("0"), "high": Decimal("1")},
}

LOW_STATE_ORDER = ("normal", "mild_low", "moderate_low", "severe_low")
HIGH_STATE_ORDER = ("normal", "mild_high", "moderate_high", "severe_high")


@dataclass(slots=True)
class SyntheticGenerationConfig:
    seed: int = 42
    samples_per_class: int = 1000
    class_target_counts: dict[str, int] | None = None
    signal_strength_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SIGNAL_STRENGTH_WEIGHTS),
    )
    archetype_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ARCHETYPE_WEIGHTS),
    )
    age_bucket_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_AGE_BUCKET_WEIGHTS),
    )
    quality_targets: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_QUALITY_TARGETS),
    )
    default_dataset_ambiguous_ratio: float = 0.25
    bad_ratio_upper: float = 0.10
    bad_ratio_recovery: float = 0.06
    min_conflicted_weight: float = 0.015
    conflicted_reduction_factor: float = 0.75
    conflicted_recovery_factor: float = 1.08
    min_overlap_factor: float = 0.70
    overlap_reduction_factor: float = 0.90
    overlap_recovery_factor: float = 1.03
    low_ambiguity_ratio: float = 0.12
    low_ambiguity_min_cases: int = 80
    low_ambiguity_min_non_malignancy_classes: int = 3
    rolling_bad_window: int = 200
    max_attempt_multiplier: int = 20

    def target_counts(self) -> dict[str, int]:
        if self.class_target_counts is not None:
            return dict(self.class_target_counts)
        return {label: self.samples_per_class for label in CLASS_PROFILES}


class SyntheticCaseGenerator:
    def __init__(
        self,
        session: Session,
        config: SyntheticGenerationConfig | None = None,
        context: RuleEngineContext | None = None,
    ) -> None:
        self.session = session
        self.config = config or SyntheticGenerationConfig()
        self.context = context or RuleEngineContext.from_session(session)
        self.rng = np.random.default_rng(self.config.seed)
        self.indicator_codes = sorted(self.context.indicators_by_code)
        self._validate_profiles()

    def generate_case(
        self,
        intended_label: str,
        case_index: int,
        *,
        signal_strength: str | None = None,
        archetype: str | None = None,
        sex: str | None = None,
        age: int | None = None,
    ) -> SyntheticCase:
        if intended_label not in CLASS_PROFILES:
            raise ValueError(f"Unknown intended label: {intended_label}")

        selected_sex = sex or self._choice(("male", "female"))
        selected_age_bucket = self._choose_age_bucket()
        selected_age = age if age is not None else self._random_age_for_bucket(selected_age_bucket)
        if age is not None:
            selected_age_bucket = self._bucket_for_age(age)

        profile = CLASS_PROFILES[intended_label]
        signal_weights = profile.signal_strength_weights_override or self.config.signal_strength_weights
        selected_signal_strength = signal_strength or self._weighted_choice(signal_weights)
        if selected_signal_strength not in SIGNAL_STRENGTHS:
            raise ValueError(f"Unsupported signal strength: {selected_signal_strength}")

        selected_archetype = archetype or self._weighted_choice(self.config.archetype_weights)
        if selected_archetype not in ARCHETYPES:
            raise ValueError(f"Unsupported archetype: {selected_archetype}")

        intended_states = {indicator_code: "normal" for indicator_code in self.indicator_codes}
        intended_states.update(profile.states_for(selected_signal_strength))
        sampling_styles = self._initial_sampling_styles(intended_states)

        overlap_source, borrowed_indicators = self._apply_archetype(
            profile=profile,
            states=intended_states,
            sampling_styles=sampling_styles,
            signal_strength=selected_signal_strength,
            archetype=selected_archetype,
        )
        self._apply_class_specific_adjustments(
            profile=profile,
            states=intended_states,
            sampling_styles=sampling_styles,
            signal_strength=selected_signal_strength,
            archetype=selected_archetype,
            overlap_source=overlap_source,
            borrowed_indicators=borrowed_indicators,
        )

        raw_values = {
            indicator_code: self.generate_value_for_state(
                reference_range=self.context.get_reference_range(
                    indicator_code=indicator_code,
                    sex=selected_sex,
                    age=selected_age,
                ),
                deviation_state=intended_states[indicator_code],
                indicator_code=indicator_code,
                sampling_style=sampling_styles[indicator_code],
            )
            for indicator_code in self.indicator_codes
        }

        return SyntheticCase(
            case_id=f"{intended_label}-{case_index:07d}",
            intended_label=intended_label,
            sex=selected_sex,
            age=selected_age,
            age_bucket=selected_age_bucket,
            signal_strength=selected_signal_strength,
            archetype=selected_archetype,
            overlap_source=overlap_source,
            borrowed_indicators=tuple(sorted(borrowed_indicators)),
            raw_values=raw_values,
            intended_deviation_states=intended_states,
        )

    def generate_value_for_state(
        self,
        *,
        reference_range: Any,
        deviation_state: str,
        indicator_code: str,
        sampling_style: str = "interior",
    ) -> Decimal:
        epsilon = DECIMAL_PRECISION
        normal_min = Decimal(str(reference_range.normal_min))
        normal_max = Decimal(str(reference_range.normal_max))
        moderate_low = Decimal(str(reference_range.moderate_low_threshold))
        severe_low = Decimal(str(reference_range.severe_low_threshold))
        moderate_high = Decimal(str(reference_range.moderate_high_threshold))
        severe_high = Decimal(str(reference_range.severe_high_threshold))

        lower_band = max(normal_min - moderate_low, moderate_low - severe_low, epsilon)
        upper_band = max(moderate_high - normal_max, severe_high - moderate_high, epsilon)

        severe_caps = INDICATOR_SEVERE_CAPS.get(
            indicator_code,
            {"low": Decimal("0"), "high": severe_high + (upper_band * Decimal("2"))},
        )

        if deviation_state == "normal":
            low, high = normal_min, normal_max
        elif deviation_state == "mild_low":
            low, high = self._open_interval(moderate_low, normal_min, epsilon)
        elif deviation_state == "moderate_low":
            low, high = self._open_interval(severe_low, moderate_low, epsilon, include_high=True)
        elif deviation_state == "severe_low":
            low = max(severe_caps["low"], severe_low - (lower_band * Decimal("2")))
            high = severe_low
        elif deviation_state == "mild_high":
            low, high = self._open_interval(normal_max, moderate_high, epsilon)
        elif deviation_state == "moderate_high":
            low, high = self._open_interval(moderate_high, severe_high, epsilon, include_low=True)
        elif deviation_state == "severe_high":
            low = severe_high
            high = min(severe_caps["high"], severe_high + (upper_band * Decimal("2")))
        else:
            raise ValueError(f"Unsupported deviation state: {deviation_state}")

        sample_low, sample_high = self._styled_interval(
            low=low,
            high=high,
            deviation_state=deviation_state,
            sampling_style=sampling_style,
        )
        return self._sample_decimal(sample_low, sample_high)

    def _validate_profiles(self) -> None:
        active_diseases = set(self.context.diseases_by_code)
        profile_labels = set(CLASS_PROFILES)
        if active_diseases != profile_labels:
            missing_profiles = sorted(active_diseases - profile_labels)
            extra_profiles = sorted(profile_labels - active_diseases)
            raise ValueError(
                f"Class profiles must match active disease codes. Missing={missing_profiles}, extra={extra_profiles}."
            )

    def _apply_archetype(
        self,
        *,
        profile: ClassProfile,
        states: dict[str, str],
        sampling_styles: dict[str, str],
        signal_strength: str,
        archetype: str,
    ) -> tuple[str | None, set[str]]:
        overlap_source: str | None = None
        borrowed_indicators: set[str] = set()
        if archetype == "canonical":
            self._edge_normalise_normals(profile, states, sampling_styles, count=1)
            return None, borrowed_indicators
        if archetype == "weaker":
            self._soften_non_core_states(profile, states, sampling_styles)
            self._edge_normalise_normals(profile, states, sampling_styles, count=2)
            return None, borrowed_indicators
        if archetype == "borderline":
            self._borderline_states(profile, states, sampling_styles)
            self._edge_normalise_normals(profile, states, sampling_styles, count=3)
            return None, borrowed_indicators
        if archetype == "overlap":
            self._borderline_states(profile, states, sampling_styles)
            overlap_source, borrowed_indicators = self._inject_neighbor_overlap(
                profile=profile,
                states=states,
                sampling_styles=sampling_styles,
                signal_strength=signal_strength,
                borrow_limit=1,
            )
            self._edge_normalise_normals(profile, states, sampling_styles, count=3)
            return overlap_source, borrowed_indicators
        if archetype == "conflicted":
            self._borderline_states(profile, states, sampling_styles)
            overlap_source, borrowed_indicators = self._inject_neighbor_overlap(
                profile=profile,
                states=states,
                sampling_styles=sampling_styles,
                signal_strength=signal_strength,
                borrow_limit=1,
            )
            self._soften_for_conflict(profile, states, sampling_styles)
            self._edge_normalise_normals(profile, states, sampling_styles, count=4)
            return overlap_source, borrowed_indicators
        raise ValueError(f"Unsupported archetype: {archetype}")

    def _apply_class_specific_adjustments(
        self,
        *,
        profile: ClassProfile,
        states: dict[str, str],
        sampling_styles: dict[str, str],
        signal_strength: str,
        archetype: str,
        overlap_source: str | None,
        borrowed_indicators: set[str],
    ) -> None:
        if profile.code == "normal":
            self._edge_normalise_normals(profile, states, sampling_styles, count=6)
            if archetype in {"overlap", "conflicted"} and self.rng.random() < 0.10:
                indicator_code = self._choice(("MONO", "RDW", "MONO", "RDW", "WBC"))
                states[indicator_code] = "mild_high"
                sampling_styles[indicator_code] = "near_normal_boundary"
            self._limit_normal_excursions(states, sampling_styles)
            return

        if profile.code == "iron_deficiency_anemia":
            if archetype in {"weaker", "borderline"}:
                soften_count = 1 if archetype == "weaker" else 2
                for code in self._pick_indicators(["RDW", "RBC", "HCT", "MCHC"], soften_count):
                    states[code] = self._soften_state(states[code], steps=1)
                    sampling_styles[code] = (
                        "near_normal_boundary" if states[code] != "normal" else "edge_normal"
                    )
            if archetype in {"borderline", "overlap", "conflicted"}:
                for code in ("HGB", "MCV", "MCH"):
                    sampling_styles[code] = "near_normal_boundary"
            if archetype == "overlap":
                softened_core = self._pick_indicators(["HGB", "MCV", "MCH"], 1)[0]
                states[softened_core] = self._soften_state(
                    states[softened_core],
                    preserve_abnormal=True,
                )
                sampling_styles[softened_core] = "near_normal_boundary"
            if archetype in {"overlap", "conflicted"} and overlap_source == "hematologic_malignancy_suspicion":
                if "PLT" in borrowed_indicators and (archetype == "overlap" or self.rng.random() < 0.60):
                    states["PLT"] = "mild_low"
                    sampling_styles["PLT"] = "near_normal_boundary"
            if archetype == "conflicted" and overlap_source == "hematologic_malignancy_suspicion":
                softened_core = self._pick_indicators(["HGB", "MCV", "MCH"], 1)[0]
                states[softened_core] = self._soften_state(
                    states[softened_core],
                    preserve_abnormal=True,
                )
                sampling_styles[softened_core] = "near_normal_boundary"
            return

        if profile.code == "macrocytic_anemia":
            if archetype == "weaker":
                softened_code = self._pick_indicators(["MCV", "HGB"], 1)[0]
                states[softened_code] = self._soften_state(
                    states[softened_code],
                    steps=1,
                    preserve_abnormal=True,
                )
                sampling_styles[softened_code] = "near_normal_boundary"
            if archetype in {"borderline", "overlap"}:
                sampling_styles["MCV"] = "near_normal_boundary"
                sampling_styles["HGB"] = "near_normal_boundary"
                softened_code = self._pick_indicators(["MCV", "HGB"], 1)[0]
                states[softened_code] = self._soften_state(
                    states[softened_code],
                    steps=1,
                    preserve_abnormal=True,
                )
                sampling_styles[softened_code] = "near_normal_boundary"
            if archetype in {"weaker", "borderline", "overlap"}:
                if self.rng.random() < 0.60:
                    states["RDW"] = self._soften_state(states["RDW"], steps=1)
                    sampling_styles["RDW"] = (
                        "near_normal_boundary" if states["RDW"] != "normal" else "edge_normal"
                    )
            if overlap_source == "hematologic_malignancy_suspicion" and "PLT" in borrowed_indicators:
                states["PLT"] = "mild_low"
                sampling_styles["PLT"] = "near_normal_boundary"
            return

        if profile.code == "bacterial_infection":
            if archetype in {"weaker", "borderline"}:
                softened_code = self._pick_indicators(["WBC", "NEU"], 1)[0]
                states[softened_code] = self._soften_state(
                    states[softened_code],
                    steps=1,
                    preserve_abnormal=True,
                )
                sampling_styles[softened_code] = "near_normal_boundary"
                if archetype == "borderline":
                    other_code = "NEU" if softened_code == "WBC" else "WBC"
                    sampling_styles[other_code] = "near_normal_boundary"
            if archetype == "overlap":
                states["WBC"] = self._soften_state(
                    states["WBC"],
                    steps=1,
                    preserve_abnormal=True,
                )
                states["NEU"] = self._soften_state(states["NEU"], steps=1)
                sampling_styles["WBC"] = "near_normal_boundary"
                sampling_styles["NEU"] = (
                    "near_normal_boundary" if states["NEU"] != "normal" else "edge_normal"
                )
            if archetype in {"overlap", "conflicted"} and (
                states["LYM"] == "normal" or "LYM" in borrowed_indicators
            ):
                states["LYM"] = "mild_high"
                sampling_styles["LYM"] = "near_normal_boundary"
            if archetype == "conflicted":
                softened_code = self._pick_indicators(["WBC", "NEU"], 1)[0]
                states[softened_code] = self._soften_state(
                    states[softened_code],
                    steps=1,
                    preserve_abnormal=True,
                )
                sampling_styles[softened_code] = "near_normal_boundary"
                other_code = "NEU" if softened_code == "WBC" else "WBC"
                if states[other_code] == "normal":
                    states[other_code] = "mild_high"
                sampling_styles[other_code] = (
                    "near_normal_boundary" if states[other_code] != "normal" else "edge_normal"
                )
                if self.rng.random() < 0.35:
                    states["MONO"] = "mild_high"
                    sampling_styles["MONO"] = "near_normal_boundary"
            return

        if profile.code == "viral_infection":
            if archetype == "weaker":
                states["WBC"] = self._soften_state(states["WBC"], steps=1)
                sampling_styles["WBC"] = (
                    "near_normal_boundary" if states["WBC"] != "normal" else "edge_normal"
                )
            if archetype == "borderline":
                states["WBC"] = self._soften_state(
                    states["WBC"],
                    steps=1,
                    preserve_abnormal=True,
                )
                sampling_styles["WBC"] = "near_normal_boundary"
            if archetype in {"borderline", "overlap"}:
                states["LYM"] = self._soften_state(
                    states["LYM"],
                    steps=1,
                    preserve_abnormal=True,
                )
                sampling_styles["LYM"] = "near_normal_boundary"
            if archetype in {"overlap", "conflicted"} and (
                states["NEU"] == "normal" or "NEU" in borrowed_indicators
            ):
                states["NEU"] = "mild_high"
                sampling_styles["NEU"] = "near_normal_boundary"
            if archetype == "overlap":
                states["WBC"] = self._soften_state(
                    states["WBC"],
                    steps=1,
                    preserve_abnormal=True,
                )
                sampling_styles["WBC"] = (
                    "near_normal_boundary" if states["WBC"] != "normal" else "edge_normal"
                )
            if archetype == "conflicted":
                if self.rng.random() < 0.75:
                    states["WBC"] = self._soften_state(
                        states["WBC"],
                        steps=1,
                        preserve_abnormal=True,
                    )
                    sampling_styles["WBC"] = (
                        "near_normal_boundary" if states["WBC"] != "normal" else "edge_normal"
                    )
                else:
                    states["LYM"] = self._soften_state(
                        states["LYM"],
                        preserve_abnormal=True,
                    )
                    sampling_styles["LYM"] = "near_normal_boundary"
            return

        if profile.code == "allergic_or_parasitic_pattern":
            sampling_styles["EOS"] = "near_normal_boundary" if archetype != "canonical" else "interior"
            if archetype in {"borderline", "overlap", "conflicted"}:
                states["EOS"] = "moderate_high"
                if states["WBC"] == "normal" and self.rng.random() < 0.50:
                    states["WBC"] = "mild_high"
                sampling_styles["WBC"] = "near_normal_boundary"
            if archetype in {"overlap", "conflicted"} and "LYM" in borrowed_indicators:
                states["LYM"] = "mild_high"
                sampling_styles["LYM"] = "near_normal_boundary"
                if states["WBC"] == "normal":
                    states["WBC"] = "mild_high"
                    sampling_styles["WBC"] = "near_normal_boundary"
            if (
                archetype == "conflicted"
                and "LYM" in borrowed_indicators
                and self.rng.random() < 0.30
            ):
                states["EOS"] = "mild_high"
                sampling_styles["EOS"] = "near_normal_boundary"
            return

        if profile.code == "thrombocytopenia_pattern":
            if archetype in {"weaker", "borderline"}:
                states["PLT"] = self._soften_state(states["PLT"], steps=1, preserve_abnormal=True)
                sampling_styles["PLT"] = "near_normal_boundary"
            if archetype in {"overlap", "conflicted"}:
                states["PLT"] = self._soften_state(states["PLT"], steps=1, preserve_abnormal=True)
                sampling_styles["PLT"] = "near_normal_boundary"
                secondary_candidates: list[tuple[str, str]] = []
                if states["HGB"] == "normal":
                    secondary_candidates.append(("HGB", "mild_low"))
                if overlap_source == "iron_deficiency_anemia" and "HCT" in borrowed_indicators:
                    secondary_candidates.append(("HCT", "mild_low"))
                elif overlap_source == "iron_deficiency_anemia":
                    secondary_candidates.append(("HCT", "mild_low"))
                if overlap_source == "hematologic_malignancy_suspicion":
                    if "WBC" in borrowed_indicators:
                        secondary_candidates.append(("WBC", "mild_high"))
                    if "HGB" in borrowed_indicators:
                        secondary_candidates.append(("HGB", "mild_low"))
                    if "NEU" in borrowed_indicators:
                        secondary_candidates.append(("NEU", "mild_high"))
                if secondary_candidates:
                    indicator_code, state_code = secondary_candidates[
                        int(self.rng.integers(0, len(secondary_candidates)))
                    ]
                    states[indicator_code] = state_code
                    sampling_styles[indicator_code] = "near_normal_boundary"
            return

        if profile.code == "hematologic_malignancy_suspicion":
            self._apply_malignancy_template(
                states=states,
                sampling_styles=sampling_styles,
                signal_strength=signal_strength,
                archetype=archetype,
                borrowed_indicators=borrowed_indicators,
            )

    def _apply_malignancy_template(
        self,
        *,
        states: dict[str, str],
        sampling_styles: dict[str, str],
        signal_strength: str,
        archetype: str,
        borrowed_indicators: set[str],
    ) -> None:
        template = str(self.rng.choice(("mixed", "cytopenic", "leukocytic")))

        if template == "cytopenic":
            states["WBC"] = "mild_high" if archetype in {"canonical", "weaker"} else (
                "normal" if signal_strength != "strong" else "mild_high"
            )
            states["HGB"] = self._soften_state(states["HGB"], steps=0 if archetype == "canonical" else 1, preserve_abnormal=True)
            states["PLT"] = self._soften_state(states["PLT"], steps=0 if archetype == "canonical" else 1, preserve_abnormal=True)
        elif template == "leukocytic":
            states["WBC"] = "moderate_high" if signal_strength == "strong" else "mild_high"
            states["PLT"] = self._soften_state(states["PLT"], steps=1)
            states["HGB"] = self._soften_state(states["HGB"], steps=1, preserve_abnormal=True)
        else:
            states["WBC"] = self._soften_state(states["WBC"], steps=1 if archetype != "canonical" else 0, preserve_abnormal=True)
            if archetype in {"weaker", "borderline"}:
                softened_code = self._pick_indicators(["HGB", "PLT"], 1)[0]
                states[softened_code] = self._soften_state(
                    states[softened_code],
                    steps=1,
                    preserve_abnormal=True,
                )

        if archetype == "canonical" and signal_strength != "strong":
            reinforced_code = self._pick_indicators(["HGB", "PLT"], 1)[0]
            states[reinforced_code] = "moderate_low"

        for code in ("WBC", "HGB", "PLT", "BASO"):
            sampling_styles[code] = "near_normal_boundary" if archetype != "canonical" else "interior"

        if archetype in {"overlap", "conflicted"} and not borrowed_indicators and self.rng.random() < 0.05:
            states["MCV"] = "mild_low"
            sampling_styles["MCV"] = "near_normal_boundary"
        if archetype == "conflicted" and not borrowed_indicators and self.rng.random() < 0.05:
            states["NEU"] = "mild_high"
            sampling_styles["NEU"] = "near_normal_boundary"

    def _initial_sampling_styles(self, states: dict[str, str]) -> dict[str, str]:
        return {
            indicator_code: ("interior" if state != "normal" else "interior")
            for indicator_code, state in states.items()
        }

    def _soften_non_core_states(
        self,
        profile: ClassProfile,
        states: dict[str, str],
        sampling_styles: dict[str, str],
    ) -> None:
        candidates = [
            code
            for code in profile.secondary_indicators + profile.optional_indicators
            if states.get(code, "normal") != "normal"
        ]
        if not candidates and len(profile.core_indicators) > 1:
            candidates = list(profile.core_indicators[1:])
        if not candidates:
            return
        indicator_code = self._pick_indicators(candidates, 1)[0]
        preserve_abnormal = indicator_code in profile.core_indicators
        states[indicator_code] = self._soften_state(
            states[indicator_code],
            preserve_abnormal=preserve_abnormal,
        )
        sampling_styles[indicator_code] = (
            "near_normal_boundary" if states[indicator_code] != "normal" else "edge_normal"
        )

    def _borderline_states(
        self,
        profile: ClassProfile,
        states: dict[str, str],
        sampling_styles: dict[str, str],
    ) -> None:
        for indicator_code, state in states.items():
            sampling_styles[indicator_code] = (
                "near_normal_boundary" if state != "normal" else sampling_styles[indicator_code]
            )

        candidates = [
            code
            for code in profile.secondary_indicators + profile.optional_indicators
            if states.get(code, "normal") != "normal"
        ]
        if not candidates and len(profile.core_indicators) > 1:
            candidates = list(profile.core_indicators[1:])
        if not candidates and profile.core_indicators:
            candidates = [profile.core_indicators[-1]]
        if not candidates:
            return

        indicator_code = self._pick_indicators(candidates, 1)[0]
        preserve_abnormal = indicator_code in profile.core_indicators
        states[indicator_code] = self._soften_state(
            states[indicator_code],
            preserve_abnormal=preserve_abnormal,
        )
        sampling_styles[indicator_code] = (
            "near_normal_boundary" if states[indicator_code] != "normal" else "edge_normal"
        )

    def _soften_for_conflict(
        self,
        profile: ClassProfile,
        states: dict[str, str],
        sampling_styles: dict[str, str],
    ) -> None:
        candidates = [
            code
            for code in profile.secondary_indicators
            if states.get(code, "normal") != "normal"
        ]
        if not candidates and len(profile.core_indicators) > 1:
            candidates = list(profile.core_indicators[1:])
        if not candidates:
            candidates = [
                code
                for code in profile.core_indicators
                if states.get(code, "normal") != "normal"
            ]
        if not candidates:
            return

        indicator_code = self._pick_indicators(candidates, 1)[0]
        preserve_abnormal = indicator_code in profile.core_indicators and len(profile.core_indicators) <= 1
        states[indicator_code] = self._soften_state(
            states[indicator_code],
            steps=1,
            preserve_abnormal=preserve_abnormal,
        )
        sampling_styles[indicator_code] = (
            "near_normal_boundary" if states[indicator_code] != "normal" else "edge_normal"
        )

    def _inject_neighbor_overlap(
        self,
        *,
        profile: ClassProfile,
        states: dict[str, str],
        sampling_styles: dict[str, str],
        signal_strength: str,
        borrow_limit: int,
    ) -> tuple[str | None, set[str]]:
        if not profile.overlap_neighbors:
            return None, set()

        neighbor_code = self._choice(profile.overlap_neighbors)
        neighbor_profile = CLASS_PROFILES[neighbor_code]
        neighbor_states = neighbor_profile.states_for(signal_strength)
        allowed_indicators = profile.borrowed_indicators_for(neighbor_code)
        if not allowed_indicators:
            return None, set()

        candidates: list[tuple[str, str]] = []
        for indicator_code, neighbor_state in neighbor_states.items():
            if indicator_code not in allowed_indicators:
                continue
            if neighbor_state == "normal":
                continue
            borrowed_state = self._to_mild_state(neighbor_state)
            current_state = states.get(indicator_code, "normal")
            if current_state == borrowed_state:
                continue
            if current_state == "normal" or self._state_family(current_state) != self._state_family(borrowed_state):
                candidates.append((indicator_code, borrowed_state))

        if not candidates:
            return None, set()

        core_candidates = [
            (indicator_code, borrowed_state)
            for indicator_code, borrowed_state in candidates
            if indicator_code in neighbor_profile.core_indicators
        ]
        candidate_pool = core_candidates or candidates
        borrowed_indicators: set[str] = set()
        for indicator_code, borrowed_state in self._pick_pairs(candidate_pool, min(borrow_limit, 1)):
            states[indicator_code] = borrowed_state
            sampling_styles[indicator_code] = "near_normal_boundary"
            borrowed_indicators.add(indicator_code)
        return neighbor_code, borrowed_indicators

    def _limit_normal_excursions(
        self,
        states: dict[str, str],
        sampling_styles: dict[str, str],
    ) -> None:
        abnormal_codes = [code for code, state in states.items() if state != "normal"]
        if len(abnormal_codes) <= 1:
            return
        keep_code = self._pick_indicators(abnormal_codes, 1)[0]
        for indicator_code in abnormal_codes:
            if indicator_code == keep_code:
                continue
            states[indicator_code] = "normal"
            sampling_styles[indicator_code] = "edge_normal"

    def _edge_normalise_normals(
        self,
        profile: ClassProfile,
        states: dict[str, str],
        sampling_styles: dict[str, str],
        *,
        count: int,
    ) -> None:
        candidates = [
            indicator_code
            for indicator_code, state in states.items()
            if state == "normal"
        ]
        if not candidates:
            return
        for indicator_code in self._pick_indicators(candidates, min(count, len(candidates))):
            sampling_styles[indicator_code] = "edge_normal"

    def _styled_interval(
        self,
        *,
        low: Decimal,
        high: Decimal,
        deviation_state: str,
        sampling_style: str,
    ) -> tuple[Decimal, Decimal]:
        if sampling_style not in SAMPLING_STYLES:
            raise ValueError(f"Unsupported sampling style: {sampling_style}")
        if high <= low:
            return low, high

        family = self._state_family(deviation_state)
        if sampling_style == "interior":
            return self._slice_interval(low, high, Decimal("0.25"), Decimal("0.75"))
        if sampling_style == "near_normal_boundary":
            if family == "low":
                return self._slice_interval(low, high, Decimal("0.75"), Decimal("1.00"))
            if family == "high":
                return self._slice_interval(low, high, Decimal("0.00"), Decimal("0.25"))
            return self._edge_normal_interval(low, high)
        if sampling_style == "near_far_boundary":
            if family == "low":
                return self._slice_interval(low, high, Decimal("0.00"), Decimal("0.25"))
            if family == "high":
                return self._slice_interval(low, high, Decimal("0.75"), Decimal("1.00"))
            return self._edge_normal_interval(low, high)
        return self._edge_normal_interval(low, high)

    def _edge_normal_interval(self, low: Decimal, high: Decimal) -> tuple[Decimal, Decimal]:
        if self.rng.random() < 0.5:
            return self._slice_interval(low, high, Decimal("0.00"), Decimal("0.20"))
        return self._slice_interval(low, high, Decimal("0.80"), Decimal("1.00"))

    def _slice_interval(
        self,
        low: Decimal,
        high: Decimal,
        start_fraction: Decimal,
        end_fraction: Decimal,
    ) -> tuple[Decimal, Decimal]:
        width = high - low
        if width <= DECIMAL_PRECISION:
            return low, high
        start = low + (width * start_fraction)
        end = low + (width * end_fraction)
        if start > end:
            start, end = end, start
        return (
            start.quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP),
            end.quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP),
        )

    def _sample_decimal(self, low: Decimal, high: Decimal) -> Decimal:
        if high < low:
            low, high = high, low
        if high == low:
            return low.quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)
        value = low + (Decimal(str(self.rng.random())) * (high - low))
        return value.quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)

    @staticmethod
    def _open_interval(
        low: Decimal,
        high: Decimal,
        epsilon: Decimal,
        *,
        include_low: bool = False,
        include_high: bool = False,
    ) -> tuple[Decimal, Decimal]:
        adjusted_low = low if include_low else low + epsilon
        adjusted_high = high if include_high else high - epsilon
        if adjusted_low > adjusted_high:
            midpoint = (low + high) / Decimal("2")
            midpoint = midpoint.quantize(DECIMAL_PRECISION, rounding=ROUND_HALF_UP)
            return midpoint, midpoint
        return adjusted_low, adjusted_high

    def _soften_state(
        self,
        state: str,
        *,
        steps: int = 1,
        preserve_abnormal: bool = False,
    ) -> str:
        softened = state
        for _ in range(max(steps, 0)):
            if softened in LOW_STATE_ORDER:
                index = LOW_STATE_ORDER.index(softened)
                softened = LOW_STATE_ORDER[max(index - 1, 0)]
            elif softened in HIGH_STATE_ORDER:
                index = HIGH_STATE_ORDER.index(softened)
                softened = HIGH_STATE_ORDER[max(index - 1, 0)]
            else:
                softened = "normal"
        if preserve_abnormal and softened == "normal":
            family = self._state_family(state)
            return "mild_low" if family == "low" else "mild_high"
        return softened

    @staticmethod
    def _to_mild_state(state: str) -> str:
        family = SyntheticCaseGenerator._state_family(state)
        if family == "low":
            return "mild_low"
        if family == "high":
            return "mild_high"
        return "normal"

    @staticmethod
    def _state_family(state: str) -> str:
        if state.endswith("low"):
            return "low"
        if state.endswith("high"):
            return "high"
        return "normal"

    def _pick_indicators(self, candidates: list[str], count: int) -> list[str]:
        if not candidates or count <= 0:
            return []
        picks = self.rng.choice(sorted(candidates), size=count, replace=False)
        return [str(item) for item in np.atleast_1d(picks)]

    def _pick_pairs(
        self,
        candidates: list[tuple[str, str]],
        count: int,
    ) -> list[tuple[str, str]]:
        if not candidates or count <= 0:
            return []
        indices = self.rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
        if np.isscalar(indices):
            indices = [int(indices)]
        return [candidates[int(index)] for index in indices]

    def _choice(self, options: tuple[str, ...] | list[str]) -> str:
        return str(self.rng.choice(list(options)))

    def _weighted_choice(self, weights: dict[str, float]) -> str:
        labels = list(weights)
        probabilities = np.array([weights[label] for label in labels], dtype=float)
        probabilities = probabilities / probabilities.sum()
        return str(self.rng.choice(labels, p=probabilities))

    def _choose_age_bucket(self) -> str:
        return self._weighted_choice(self.config.age_bucket_weights)

    def _random_age_for_bucket(self, age_bucket: str) -> int:
        age_min, age_max = AGE_BUCKETS[age_bucket]
        return int(self.rng.integers(age_min, age_max + 1))

    @staticmethod
    def _bucket_for_age(age: int) -> str:
        for bucket_name, (age_min, age_max) in AGE_BUCKETS.items():
            if age_min <= age <= age_max:
                return bucket_name
        raise ValueError(f"Age {age} is outside the supported adult buckets.")

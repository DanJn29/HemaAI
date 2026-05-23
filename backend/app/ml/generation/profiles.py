from dataclasses import dataclass, field


SIGNAL_STRENGTHS = ("strong", "weak", "variant")
ARCHETYPES = ("canonical", "weaker", "borderline", "overlap", "conflicted")
QUALITY_LABELS = ("GOOD", "AMBIGUOUS", "BAD")
DATASET_VARIANTS = ("strict", "default")

DEFAULT_SIGNAL_STRENGTH_WEIGHTS = {
    "strong": 0.45,
    "weak": 0.35,
    "variant": 0.20,
}

CLASS_SIGNAL_STRENGTH_WEIGHTS = {
    "bacterial_infection": {
        "strong": 0.36,
        "weak": 0.40,
        "variant": 0.24,
    },
    "viral_infection": {
        "strong": 0.48,
        "weak": 0.34,
        "variant": 0.18,
    },
    "hematologic_malignancy_suspicion": {
        "strong": 0.62,
        "weak": 0.28,
        "variant": 0.10,
    },
}

DEFAULT_ARCHETYPE_WEIGHTS = {
    "canonical": 0.45,
    "weaker": 0.20,
    "borderline": 0.20,
    "overlap": 0.10,
    "conflicted": 0.05,
}

CLASS_ARCHETYPE_WEIGHTS = {
    "normal": {
        "canonical": 0.73,
        "weaker": 0.18,
        "borderline": 0.07,
        "overlap": 0.015,
        "conflicted": 0.005,
    },
    "iron_deficiency_anemia": {
        "canonical": 0.46,
        "weaker": 0.25,
        "borderline": 0.18,
        "overlap": 0.08,
        "conflicted": 0.03,
    },
    "macrocytic_anemia": {
        "canonical": 0.47,
        "weaker": 0.24,
        "borderline": 0.18,
        "overlap": 0.08,
        "conflicted": 0.03,
    },
    "bacterial_infection": {
        "canonical": 0.38,
        "weaker": 0.25,
        "borderline": 0.20,
        "overlap": 0.12,
        "conflicted": 0.05,
    },
    "viral_infection": {
        "canonical": 0.38,
        "weaker": 0.25,
        "borderline": 0.20,
        "overlap": 0.12,
        "conflicted": 0.05,
    },
    "allergic_or_parasitic_pattern": {
        "canonical": 0.50,
        "weaker": 0.22,
        "borderline": 0.18,
        "overlap": 0.07,
        "conflicted": 0.03,
    },
    "thrombocytopenia_pattern": {
        "canonical": 0.46,
        "weaker": 0.23,
        "borderline": 0.18,
        "overlap": 0.09,
        "conflicted": 0.04,
    },
    "hematologic_malignancy_suspicion": {
        "canonical": 0.54,
        "weaker": 0.22,
        "borderline": 0.16,
        "overlap": 0.06,
        "conflicted": 0.04,
    },
}

CLASS_QUALITY_TARGETS = {
    "normal": {"ambiguous": 0.02, "bad": 0.01},
    "iron_deficiency_anemia": {"ambiguous": 0.18, "bad": 0.02},
    "macrocytic_anemia": {"ambiguous": 0.20, "bad": 0.02},
    "bacterial_infection": {"ambiguous": 0.25, "bad": 0.04},
    "viral_infection": {"ambiguous": 0.25, "bad": 0.04},
    "allergic_or_parasitic_pattern": {"ambiguous": 0.20, "bad": 0.02},
    "thrombocytopenia_pattern": {"ambiguous": 0.10, "bad": 0.03},
    "hematologic_malignancy_suspicion": {"ambiguous": 0.18, "bad": 0.08},
}

OVERLAP_NEIGHBOR_MAP = {
    "normal": (),
    "iron_deficiency_anemia": ("hematologic_malignancy_suspicion",),
    "macrocytic_anemia": ("hematologic_malignancy_suspicion",),
    "bacterial_infection": ("viral_infection",),
    "viral_infection": ("bacterial_infection",),
    "allergic_or_parasitic_pattern": ("viral_infection",),
    "thrombocytopenia_pattern": (
        "hematologic_malignancy_suspicion",
        "iron_deficiency_anemia",
    ),
    "hematologic_malignancy_suspicion": (
        "iron_deficiency_anemia",
        "thrombocytopenia_pattern",
        "bacterial_infection",
    ),
}

NEIGHBOR_BORROW_WHITELISTS = {
    "normal": {},
    "iron_deficiency_anemia": {
        "hematologic_malignancy_suspicion": ("PLT",),
    },
    "macrocytic_anemia": {
        "hematologic_malignancy_suspicion": ("PLT", "BASO"),
    },
    "bacterial_infection": {
        "viral_infection": ("LYM",),
    },
    "viral_infection": {
        "bacterial_infection": ("NEU",),
    },
    "allergic_or_parasitic_pattern": {
        "viral_infection": ("WBC", "LYM"),
    },
    "thrombocytopenia_pattern": {
        "hematologic_malignancy_suspicion": ("WBC", "NEU", "HGB"),
        "iron_deficiency_anemia": ("HGB", "HCT"),
    },
    "hematologic_malignancy_suspicion": {
        "iron_deficiency_anemia": ("MCV", "MCH"),
        "thrombocytopenia_pattern": ("PLT",),
        "bacterial_infection": ("WBC", "NEU"),
    },
}


@dataclass(frozen=True, slots=True)
class IndicatorSignalPlan:
    strong: str
    weak: str
    variant: str

    def state_for(self, signal_strength: str) -> str:
        return getattr(self, signal_strength)


@dataclass(frozen=True, slots=True)
class ClassProfile:
    code: str
    indicators: dict[str, IndicatorSignalPlan]
    core_indicators: tuple[str, ...]
    secondary_indicators: tuple[str, ...] = ()
    optional_indicators: tuple[str, ...] = ()
    overlap_neighbors: tuple[str, ...] = ()
    signal_strength_weights_override: dict[str, float] | None = None
    archetype_weights_override: dict[str, float] | None = None
    target_ambiguous_ratio: float = 0.20
    target_bad_ratio: float = 0.10
    neighbor_borrow_whitelist: dict[str, tuple[str, ...]] = field(default_factory=dict)
    default_state: str = "normal"

    def states_for(self, signal_strength: str) -> dict[str, str]:
        return {
            indicator_code: plan.state_for(signal_strength)
            for indicator_code, plan in self.indicators.items()
        }

    def abnormal_indicators(self, signal_strength: str) -> tuple[str, ...]:
        return tuple(
            indicator_code
            for indicator_code, state in self.states_for(signal_strength).items()
            if state != self.default_state
        )

    def borrowed_indicators_for(self, neighbor_code: str) -> tuple[str, ...]:
        return self.neighbor_borrow_whitelist.get(neighbor_code, ())


CLASS_PROFILES: dict[str, ClassProfile] = {
    "normal": ClassProfile(
        code="normal",
        indicators={
            "WBC": IndicatorSignalPlan("normal", "normal", "normal"),
            "MONO": IndicatorSignalPlan("normal", "normal", "normal"),
            "RDW": IndicatorSignalPlan("normal", "normal", "normal"),
        },
        core_indicators=(),
        optional_indicators=("WBC", "MONO", "RDW"),
        overlap_neighbors=OVERLAP_NEIGHBOR_MAP["normal"],
        archetype_weights_override=CLASS_ARCHETYPE_WEIGHTS["normal"],
        target_ambiguous_ratio=CLASS_QUALITY_TARGETS["normal"]["ambiguous"],
        target_bad_ratio=CLASS_QUALITY_TARGETS["normal"]["bad"],
        neighbor_borrow_whitelist=NEIGHBOR_BORROW_WHITELISTS["normal"],
    ),
    "iron_deficiency_anemia": ClassProfile(
        code="iron_deficiency_anemia",
        indicators={
            "HGB": IndicatorSignalPlan("moderate_low", "mild_low", "mild_low"),
            "MCV": IndicatorSignalPlan("moderate_low", "mild_low", "mild_low"),
            "MCH": IndicatorSignalPlan("moderate_low", "mild_low", "mild_low"),
            "RDW": IndicatorSignalPlan("moderate_high", "mild_high", "normal"),
            "RBC": IndicatorSignalPlan("mild_low", "normal", "normal"),
            "HCT": IndicatorSignalPlan("mild_low", "normal", "normal"),
            "MCHC": IndicatorSignalPlan("mild_low", "normal", "normal"),
        },
        core_indicators=("HGB", "MCV", "MCH"),
        secondary_indicators=("RDW", "RBC", "HCT", "MCHC"),
        overlap_neighbors=OVERLAP_NEIGHBOR_MAP["iron_deficiency_anemia"],
        archetype_weights_override=CLASS_ARCHETYPE_WEIGHTS["iron_deficiency_anemia"],
        target_ambiguous_ratio=CLASS_QUALITY_TARGETS["iron_deficiency_anemia"]["ambiguous"],
        target_bad_ratio=CLASS_QUALITY_TARGETS["iron_deficiency_anemia"]["bad"],
        neighbor_borrow_whitelist=NEIGHBOR_BORROW_WHITELISTS["iron_deficiency_anemia"],
    ),
    "macrocytic_anemia": ClassProfile(
        code="macrocytic_anemia",
        indicators={
            "HGB": IndicatorSignalPlan("moderate_low", "mild_low", "mild_low"),
            "MCV": IndicatorSignalPlan("moderate_high", "mild_high", "mild_high"),
            "RDW": IndicatorSignalPlan("mild_high", "normal", "normal"),
        },
        core_indicators=("HGB", "MCV"),
        secondary_indicators=("RDW",),
        overlap_neighbors=OVERLAP_NEIGHBOR_MAP["macrocytic_anemia"],
        archetype_weights_override=CLASS_ARCHETYPE_WEIGHTS["macrocytic_anemia"],
        target_ambiguous_ratio=CLASS_QUALITY_TARGETS["macrocytic_anemia"]["ambiguous"],
        target_bad_ratio=CLASS_QUALITY_TARGETS["macrocytic_anemia"]["bad"],
        neighbor_borrow_whitelist=NEIGHBOR_BORROW_WHITELISTS["macrocytic_anemia"],
    ),
    "bacterial_infection": ClassProfile(
        code="bacterial_infection",
        indicators={
            "WBC": IndicatorSignalPlan("moderate_high", "mild_high", "mild_high"),
            "NEU": IndicatorSignalPlan("moderate_high", "mild_high", "mild_high"),
            "LYM": IndicatorSignalPlan("normal", "normal", "normal"),
            "MONO": IndicatorSignalPlan("normal", "mild_high", "mild_high"),
        },
        core_indicators=("WBC", "NEU"),
        secondary_indicators=("MONO",),
        optional_indicators=("LYM",),
        overlap_neighbors=OVERLAP_NEIGHBOR_MAP["bacterial_infection"],
        signal_strength_weights_override=CLASS_SIGNAL_STRENGTH_WEIGHTS["bacterial_infection"],
        archetype_weights_override=CLASS_ARCHETYPE_WEIGHTS["bacterial_infection"],
        target_ambiguous_ratio=CLASS_QUALITY_TARGETS["bacterial_infection"]["ambiguous"],
        target_bad_ratio=CLASS_QUALITY_TARGETS["bacterial_infection"]["bad"],
        neighbor_borrow_whitelist=NEIGHBOR_BORROW_WHITELISTS["bacterial_infection"],
    ),
    "viral_infection": ClassProfile(
        code="viral_infection",
        indicators={
            "LYM": IndicatorSignalPlan("moderate_high", "moderate_high", "mild_high"),
            "WBC": IndicatorSignalPlan("mild_high", "mild_high", "mild_high"),
            "NEU": IndicatorSignalPlan("normal", "normal", "normal"),
        },
        core_indicators=("LYM",),
        secondary_indicators=("WBC",),
        optional_indicators=("NEU",),
        overlap_neighbors=OVERLAP_NEIGHBOR_MAP["viral_infection"],
        signal_strength_weights_override=CLASS_SIGNAL_STRENGTH_WEIGHTS["viral_infection"],
        archetype_weights_override=CLASS_ARCHETYPE_WEIGHTS["viral_infection"],
        target_ambiguous_ratio=CLASS_QUALITY_TARGETS["viral_infection"]["ambiguous"],
        target_bad_ratio=CLASS_QUALITY_TARGETS["viral_infection"]["bad"],
        neighbor_borrow_whitelist=NEIGHBOR_BORROW_WHITELISTS["viral_infection"],
    ),
    "allergic_or_parasitic_pattern": ClassProfile(
        code="allergic_or_parasitic_pattern",
        indicators={
            "EOS": IndicatorSignalPlan("moderate_high", "moderate_high", "moderate_high"),
            "WBC": IndicatorSignalPlan("normal", "normal", "mild_high"),
        },
        core_indicators=("EOS",),
        optional_indicators=("WBC",),
        overlap_neighbors=OVERLAP_NEIGHBOR_MAP["allergic_or_parasitic_pattern"],
        archetype_weights_override=CLASS_ARCHETYPE_WEIGHTS["allergic_or_parasitic_pattern"],
        target_ambiguous_ratio=CLASS_QUALITY_TARGETS["allergic_or_parasitic_pattern"]["ambiguous"],
        target_bad_ratio=CLASS_QUALITY_TARGETS["allergic_or_parasitic_pattern"]["bad"],
        neighbor_borrow_whitelist=NEIGHBOR_BORROW_WHITELISTS["allergic_or_parasitic_pattern"],
    ),
    "thrombocytopenia_pattern": ClassProfile(
        code="thrombocytopenia_pattern",
        indicators={
            "PLT": IndicatorSignalPlan("moderate_low", "mild_low", "mild_low"),
            "HGB": IndicatorSignalPlan("normal", "normal", "mild_low"),
        },
        core_indicators=("PLT",),
        secondary_indicators=("HGB",),
        overlap_neighbors=OVERLAP_NEIGHBOR_MAP["thrombocytopenia_pattern"],
        archetype_weights_override=CLASS_ARCHETYPE_WEIGHTS["thrombocytopenia_pattern"],
        target_ambiguous_ratio=CLASS_QUALITY_TARGETS["thrombocytopenia_pattern"]["ambiguous"],
        target_bad_ratio=CLASS_QUALITY_TARGETS["thrombocytopenia_pattern"]["bad"],
        neighbor_borrow_whitelist=NEIGHBOR_BORROW_WHITELISTS["thrombocytopenia_pattern"],
    ),
    "hematologic_malignancy_suspicion": ClassProfile(
        code="hematologic_malignancy_suspicion",
        indicators={
            "WBC": IndicatorSignalPlan("moderate_high", "mild_high", "mild_high"),
            "HGB": IndicatorSignalPlan("moderate_low", "mild_low", "mild_low"),
            "PLT": IndicatorSignalPlan("moderate_low", "mild_low", "mild_low"),
            "BASO": IndicatorSignalPlan("mild_high", "mild_high", "mild_high"),
        },
        core_indicators=("HGB", "PLT"),
        secondary_indicators=("WBC", "BASO"),
        optional_indicators=("MCV", "MCH", "NEU"),
        overlap_neighbors=OVERLAP_NEIGHBOR_MAP["hematologic_malignancy_suspicion"],
        signal_strength_weights_override=CLASS_SIGNAL_STRENGTH_WEIGHTS["hematologic_malignancy_suspicion"],
        archetype_weights_override=CLASS_ARCHETYPE_WEIGHTS["hematologic_malignancy_suspicion"],
        target_ambiguous_ratio=CLASS_QUALITY_TARGETS["hematologic_malignancy_suspicion"]["ambiguous"],
        target_bad_ratio=CLASS_QUALITY_TARGETS["hematologic_malignancy_suspicion"]["bad"],
        neighbor_borrow_whitelist=NEIGHBOR_BORROW_WHITELISTS["hematologic_malignancy_suspicion"],
    ),
}

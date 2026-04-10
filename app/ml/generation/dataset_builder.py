from collections import Counter, defaultdict, deque
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from app.ml.generation.evaluator import RuleEngineContext, RuleEngineEvaluator
from app.ml.generation.generator import SyntheticCaseGenerator, SyntheticGenerationConfig
from app.ml.generation.profiles import ARCHETYPES, CLASS_PROFILES
from app.ml.types import DatasetBundle


class QualityMixController:
    def __init__(self, config: SyntheticGenerationConfig) -> None:
        self.config = config
        self.base_archetype_weights_by_label = {
            label: dict(
                CLASS_PROFILES[label].archetype_weights_override
                or config.archetype_weights
            )
            for label in CLASS_PROFILES
        }
        self.current_conflicted_weights = {
            label: float(weights.get("conflicted", 0.0))
            for label, weights in self.base_archetype_weights_by_label.items()
        }
        self.current_overlap_factors = {
            label: 1.0
            for label, profile in CLASS_PROFILES.items()
            if profile.overlap_neighbors
        }
        self.hm_overlap_factor = 1.0
        self.hm_conflicted_factor = 1.0
        self.global_quality_counts: Counter[str] = Counter()
        self.per_class_quality_counts: dict[str, Counter[str]] = defaultdict(Counter)
        self.attempt_counts: Counter[str] = Counter()
        self.rolling_quality_window: deque[str] = deque(maxlen=config.rolling_bad_window)

    def record(self, *, label: str, quality_label: str) -> None:
        self.global_quality_counts[quality_label] += 1
        self.per_class_quality_counts[label][quality_label] += 1
        self.attempt_counts[label] += 1
        self.rolling_quality_window.append(quality_label)
        self._adjust_difficulty_weights()
        self._adjust_malignancy_share_weights()

    def archetype_weights_for(self, label: str) -> dict[str, float]:
        profile = CLASS_PROFILES[label]
        weights = dict(self.base_archetype_weights_by_label[label])
        weights["conflicted"] = self.current_conflicted_weights[label]
        if profile.overlap_neighbors:
            weights["overlap"] *= self.current_overlap_factors[label]
        if label == "hematologic_malignancy_suspicion":
            weights["overlap"] *= self.hm_overlap_factor
            weights["conflicted"] *= self.hm_conflicted_factor

        total_attempts = sum(self.global_quality_counts.values())
        class_attempts = self.attempt_counts[label]

        global_good_ratio = self._ratio(self.global_quality_counts, total_attempts, "GOOD")
        global_ambiguous_ratio = self._ratio(self.global_quality_counts, total_attempts, "AMBIGUOUS")
        global_bad_ratio = self._ratio(self.global_quality_counts, total_attempts, "BAD")
        non_malignancy_ambiguous_classes = self._non_malignancy_quality_class_count("AMBIGUOUS")
        class_ambiguous_target = profile.target_ambiguous_ratio
        class_bad_target = profile.target_bad_ratio
        class_ambiguous_ratio = self._ratio(
            self.per_class_quality_counts[label],
            class_attempts,
            "AMBIGUOUS",
        )
        class_bad_ratio = self._ratio(
            self.per_class_quality_counts[label],
            class_attempts,
            "BAD",
        )
        low_ambiguity_guard = (
            total_attempts >= self.config.low_ambiguity_min_cases
            and (
                global_ambiguous_ratio < self.config.low_ambiguity_ratio
                or non_malignancy_ambiguous_classes < self.config.low_ambiguity_min_non_malignancy_classes
            )
        )

        if total_attempts >= 10 and global_ambiguous_ratio < self.config.quality_targets["AMBIGUOUS"]:
            weights["borderline"] *= 1.10
            weights["overlap"] *= 1.20
        elif total_attempts >= 10 and global_ambiguous_ratio > self.config.quality_targets["AMBIGUOUS"] + 0.08:
            weights["borderline"] *= 0.92
            weights["overlap"] *= 0.92

        if class_attempts >= 4 and class_ambiguous_ratio < class_ambiguous_target:
            ambiguity_gap = class_ambiguous_target - class_ambiguous_ratio
            weights["borderline"] *= 1.05 + min(0.20, ambiguity_gap * 1.5)
            if profile.overlap_neighbors:
                weights["overlap"] *= 1.08 + min(0.25, ambiguity_gap * 2.0)
        elif class_attempts >= 4 and class_ambiguous_ratio > class_ambiguous_target + 0.08:
            weights["borderline"] *= 0.92
            weights["overlap"] *= 0.92

        if total_attempts >= 10 and global_good_ratio < self.config.quality_targets["GOOD"]:
            weights["canonical"] *= 1.10
            weights["weaker"] *= 1.06

        if (
            total_attempts >= 10
            and global_bad_ratio < self.config.quality_targets["BAD"]
            and class_bad_ratio < class_bad_target
        ):
            weights["conflicted"] *= 1.08
        elif total_attempts >= 10 and global_bad_ratio > self.config.bad_ratio_upper:
            weights["conflicted"] *= 0.80
            if profile.overlap_neighbors:
                weights["overlap"] *= 0.95
            weights["canonical"] *= 1.10
            weights["weaker"] *= 1.05

        if class_attempts >= 6 and class_bad_ratio > max(class_bad_target, self.config.bad_ratio_upper):
            weights["conflicted"] *= 0.40
            weights["borderline"] *= 0.92
            weights["overlap"] *= 0.90
            weights["canonical"] *= 1.12
            weights["weaker"] *= 1.08

        if low_ambiguity_guard and class_ambiguous_ratio < class_ambiguous_target:
            weights["borderline"] *= 1.20
            if profile.overlap_neighbors and label != "hematologic_malignancy_suspicion":
                weights["overlap"] *= 1.10
            weights["conflicted"] *= 0.95

        if label == "hematologic_malignancy_suspicion" and total_attempts >= 40:
            hm_ambiguous = self.per_class_quality_counts[label]["AMBIGUOUS"]
            hm_share = hm_ambiguous / max(self.global_quality_counts["AMBIGUOUS"], 1)
            if hm_share > 0.35:
                weights["borderline"] *= 0.60
                weights["overlap"] *= 0.55
                weights["conflicted"] *= 0.60
                weights["canonical"] *= 1.20
                weights["weaker"] *= 1.10

        return self._normalize(weights)

    def _adjust_difficulty_weights(self) -> None:
        if len(self.rolling_quality_window) < min(25, self.config.rolling_bad_window):
            return

        rolling_bad_ratio = self.rolling_quality_window.count("BAD") / len(self.rolling_quality_window)
        total_attempts = sum(self.global_quality_counts.values())
        global_ambiguous_ratio = self._ratio(
            self.global_quality_counts,
            total_attempts,
            "AMBIGUOUS",
        )
        low_ambiguity_guard = (
            total_attempts >= self.config.low_ambiguity_min_cases
            and (
                global_ambiguous_ratio < self.config.low_ambiguity_ratio
                or self._non_malignancy_quality_class_count("AMBIGUOUS")
                < self.config.low_ambiguity_min_non_malignancy_classes
            )
        )
        for label, base_weights in self.base_archetype_weights_by_label.items():
            base_conflicted = float(base_weights.get("conflicted", 0.0))
            min_conflicted = min(base_conflicted, self.config.min_conflicted_weight)
            current = self.current_conflicted_weights[label]

            if rolling_bad_ratio > self.config.bad_ratio_upper:
                self.current_conflicted_weights[label] = max(
                    min_conflicted,
                    current * self.config.conflicted_reduction_factor,
                )
            elif rolling_bad_ratio < self.config.bad_ratio_recovery and current < base_conflicted:
                self.current_conflicted_weights[label] = min(
                    base_conflicted,
                    current * self.config.conflicted_recovery_factor,
                )

            if label not in self.current_overlap_factors:
                continue

            current_overlap = self.current_overlap_factors[label]
            if rolling_bad_ratio > self.config.bad_ratio_upper and not low_ambiguity_guard:
                self.current_overlap_factors[label] = max(
                    self.config.min_overlap_factor,
                    current_overlap * self.config.overlap_reduction_factor,
                )
            elif (
                rolling_bad_ratio < self.config.bad_ratio_recovery
                and current_overlap < 1.0
            ):
                self.current_overlap_factors[label] = min(
                    1.0,
                    current_overlap * self.config.overlap_recovery_factor,
                )

    def _adjust_malignancy_share_weights(self) -> None:
        total_generated = sum(self.global_quality_counts.values())
        if total_generated < 40:
            return

        total_ambiguous = self.global_quality_counts["AMBIGUOUS"]
        total_bad = self.global_quality_counts["BAD"]
        hm_counts = self.per_class_quality_counts["hematologic_malignancy_suspicion"]

        if total_ambiguous > 0:
            hm_share = hm_counts["AMBIGUOUS"] / total_ambiguous
            if hm_share > 0.40:
                self.hm_overlap_factor = max(0.10, self.hm_overlap_factor * 0.65)
                self.hm_conflicted_factor = max(0.15, self.hm_conflicted_factor * 0.75)
            elif hm_share < 0.32:
                self.hm_overlap_factor = min(1.0, self.hm_overlap_factor * 1.05)
                self.hm_conflicted_factor = min(1.0, self.hm_conflicted_factor * 1.05)

        if total_bad > 0:
            hm_bad_share = hm_counts["BAD"] / total_bad
            if hm_bad_share > 0.45:
                self.hm_conflicted_factor = max(0.20, self.hm_conflicted_factor * 0.85)
            elif hm_bad_share < 0.35:
                self.hm_conflicted_factor = min(1.0, self.hm_conflicted_factor * 1.05)

    def _non_malignancy_quality_class_count(self, quality_label: str) -> int:
        return sum(
            self.per_class_quality_counts[label][quality_label] > 0
            for label in CLASS_PROFILES
            if label != "hematologic_malignancy_suspicion"
        )

    @staticmethod
    def _ratio(counts: Counter[str], total: int, key: str) -> float:
        if total <= 0:
            return 0.0
        return counts[key] / total

    @staticmethod
    def _normalize(weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {name: 1 / len(ARCHETYPES) for name in ARCHETYPES}
        return {name: value / total for name, value in weights.items()}


class SyntheticDatasetBuilder:
    def __init__(
        self,
        session: Session,
        config: SyntheticGenerationConfig | None = None,
        context: RuleEngineContext | None = None,
    ) -> None:
        self.session = session
        self.config = config or SyntheticGenerationConfig()
        self.context = context or RuleEngineContext.from_session(session)
        self.generator = SyntheticCaseGenerator(
            session=session,
            config=self.config,
            context=self.context,
        )
        self.evaluator = RuleEngineEvaluator(session=session, context=self.context)
        self.rng = np.random.default_rng(self.config.seed)
        self.quality_controller = QualityMixController(self.config)

    def build(self) -> DatasetBundle:
        target_counts = self.config.target_counts()
        all_rows: list[dict[str, object]] = []
        good_counts: Counter[str] = Counter()
        attempt_counts: Counter[str] = Counter()
        case_index = 1

        for label, target_count in target_counts.items():
            attempts_for_label = 0
            max_attempts = max(target_count * self.config.max_attempt_multiplier, target_count + 1)
            while good_counts[label] < target_count:
                if attempts_for_label >= max_attempts:
                    raise RuntimeError(
                        f"Unable to generate {target_count} GOOD synthetic cases for {label} "
                        f"within {max_attempts} attempts."
                    )

                archetype = self._choose_archetype(label)
                case = self.generator.generate_case(
                    label,
                    case_index,
                    archetype=archetype,
                )
                evaluation = self.evaluator.evaluate_case(case)
                all_rows.append(self.evaluator.serialise_evaluation(evaluation))

                attempt_counts[label] += 1
                attempts_for_label += 1
                case_index += 1
                self.quality_controller.record(label=label, quality_label=evaluation.quality_label)

                if evaluation.quality_label == "GOOD":
                    good_counts[label] += 1

            case_index = self._generate_diversity_tail(
                label=label,
                target_count=target_count,
                case_index=case_index,
                all_rows=all_rows,
                attempt_counts=attempt_counts,
            )

        all_df = pd.DataFrame(all_rows).sort_values("case_id").reset_index(drop=True)
        good_df = all_df[all_df["quality_label"] == "GOOD"].reset_index(drop=True)
        ambiguous_df = all_df[all_df["quality_label"] == "AMBIGUOUS"].reset_index(drop=True)
        bad_df = all_df[all_df["quality_label"] == "BAD"].reset_index(drop=True)

        strict_df = self._balanced_good_sample(good_df, target_counts)
        default_df = self._balanced_default_sample(all_df, target_counts)

        strict_splits = self._split_balanced_dataset(strict_df)
        default_splits = self._split_balanced_dataset(default_df)
        summary = self._build_summary(
            all_df=all_df,
            target_counts=target_counts,
            attempt_counts=attempt_counts,
            strict_df=strict_df,
            default_df=default_df,
            strict_splits=strict_splits,
            default_splits=default_splits,
        )
        diagnostics = self._build_diagnostics(all_df, strict_df, default_df)

        return DatasetBundle(
            all_cases=all_df.to_dict(orient="records"),
            good_cases=good_df.to_dict(orient="records"),
            ambiguous_cases=ambiguous_df.to_dict(orient="records"),
            bad_cases=bad_df.to_dict(orient="records"),
            train_dataset_strict=strict_df.to_dict(orient="records"),
            train_dataset_default=default_df.to_dict(orient="records"),
            splits={
                "strict": {
                    split_name: frame.to_dict(orient="records")
                    for split_name, frame in strict_splits.items()
                },
                "default": {
                    split_name: frame.to_dict(orient="records")
                    for split_name, frame in default_splits.items()
                },
            },
            summary=summary,
            diagnostics=diagnostics,
        )

    def _generate_diversity_tail(
        self,
        *,
        label: str,
        target_count: int,
        case_index: int,
        all_rows: list[dict[str, object]],
        attempt_counts: Counter[str],
    ) -> int:
        profile = CLASS_PROFILES[label]
        quality_counts = self.quality_controller.per_class_quality_counts[label]
        if label in {"normal", "hematologic_malignancy_suspicion"}:
            return case_index
        if not profile.overlap_neighbors:
            return case_index
        target_ambiguous_count = max(1, int(round(target_count * profile.target_ambiguous_ratio)))
        if quality_counts["AMBIGUOUS"] >= target_ambiguous_count:
            return case_index

        tail_sequences: dict[str, tuple[tuple[str, str], ...]] = {
            "bacterial_infection": (
                ("overlap", "variant"),
                ("overlap", "weak"),
                ("overlap", "variant"),
                ("overlap", "weak"),
                ("borderline", "variant"),
                ("overlap", "strong"),
            ),
            "viral_infection": (
                ("overlap", "weak"),
                ("overlap", "variant"),
                ("conflicted", "weak"),
                ("overlap", "strong"),
                ("overlap", "variant"),
                ("conflicted", "strong"),
            ),
            "iron_deficiency_anemia": (
                ("conflicted", "variant"),
                ("conflicted", "weak"),
                ("conflicted", "variant"),
                ("conflicted", "weak"),
                ("borderline", "variant"),
                ("overlap", "variant"),
            ),
            "macrocytic_anemia": (
                ("overlap", "variant"),
                ("conflicted", "weak"),
                ("overlap", "weak"),
                ("conflicted", "variant"),
                ("overlap", "variant"),
                ("conflicted", "weak"),
            ),
            "allergic_or_parasitic_pattern": (
                ("overlap", "weak"),
                ("overlap", "variant"),
                ("conflicted", "weak"),
                ("overlap", "strong"),
                ("overlap", "variant"),
                ("conflicted", "strong"),
            ),
            "thrombocytopenia_pattern": (
                ("overlap", "variant"),
                ("conflicted", "variant"),
                ("overlap", "weak"),
            ),
        }
        sequence = tail_sequences.get(
            label,
            (("borderline", "weak"), ("overlap", "variant"), ("conflicted", "weak")),
        )
        max_attempts = max(len(sequence), target_ambiguous_count * 2)
        for attempt_index in range(max_attempts):
            archetype, signal_strength = sequence[attempt_index % len(sequence)]
            case = self.generator.generate_case(
                label,
                case_index,
                archetype=archetype,
                signal_strength=signal_strength,
            )
            evaluation = self.evaluator.evaluate_case(case)
            all_rows.append(self.evaluator.serialise_evaluation(evaluation))
            attempt_counts[label] += 1
            self.quality_controller.record(label=label, quality_label=evaluation.quality_label)
            case_index += 1
            quality_counts = self.quality_controller.per_class_quality_counts[label]
            if quality_counts["AMBIGUOUS"] >= target_ambiguous_count:
                return case_index

        global_bad_ratio = self.quality_controller._ratio(
            self.quality_controller.global_quality_counts,
            sum(self.quality_controller.global_quality_counts.values()),
            "BAD",
        )
        if quality_counts["AMBIGUOUS"] > 0 or quality_counts["BAD"] > 0 or global_bad_ratio >= 0.05:
            return case_index

        case = self.generator.generate_case(label, case_index, archetype="conflicted")
        evaluation = self.evaluator.evaluate_case(case)
        all_rows.append(self.evaluator.serialise_evaluation(evaluation))
        attempt_counts[label] += 1
        self.quality_controller.record(label=label, quality_label=evaluation.quality_label)
        case_index += 1
        return case_index

    def export(self, output_dir: str | Path, bundle: DatasetBundle) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self._write_csv(output_path / "all_cases.csv", bundle.all_cases)
        self._write_csv(output_path / "good_cases.csv", bundle.good_cases)
        self._write_csv(output_path / "ambiguous_cases.csv", bundle.ambiguous_cases)
        self._write_csv(output_path / "bad_cases.csv", bundle.bad_cases)
        self._write_csv(output_path / "train_dataset_strict.csv", bundle.train_dataset_strict)
        self._write_csv(output_path / "train_dataset_default.csv", bundle.train_dataset_default)

        for variant, split_map in bundle.splits.items():
            for split_name, rows in split_map.items():
                self._write_csv(output_path / f"{variant}_{split_name}.csv", rows)

        (output_path / "dataset_summary.json").write_text(
            json.dumps(bundle.summary, indent=2),
            encoding="utf-8",
        )
        (output_path / "dataset_diagnostics.json").write_text(
            json.dumps(bundle.diagnostics, indent=2),
            encoding="utf-8",
        )

    def _balanced_good_sample(
        self,
        frame: pd.DataFrame,
        target_counts: dict[str, int],
    ) -> pd.DataFrame:
        return self._balanced_sample_by_quality(frame, target_counts, strict=True)

    def _balanced_default_sample(
        self,
        frame: pd.DataFrame,
        target_counts: dict[str, int],
    ) -> pd.DataFrame:
        return self._balanced_sample_by_quality(frame, target_counts, strict=False)

    def _balanced_sample_by_quality(
        self,
        frame: pd.DataFrame,
        target_counts: dict[str, int],
        *,
        strict: bool,
    ) -> pd.DataFrame:
        samples: list[pd.DataFrame] = []
        for label, target_count in target_counts.items():
            class_rows = frame[frame["intended_label"] == label]
            if strict:
                source_rows = class_rows[class_rows["quality_label"] == "GOOD"]
                if len(source_rows) < target_count:
                    raise RuntimeError(
                        f"Not enough GOOD synthetic cases to build the strict dataset for {label}. "
                        f"Need {target_count}, got {len(source_rows)}."
                    )
                seed = int(self.rng.integers(0, 1_000_000_000))
                samples.append(
                    source_rows.sample(n=target_count, random_state=seed, replace=False),
                )
                continue

            profile = CLASS_PROFILES[label]
            ambiguous_target = int(round(target_count * profile.target_ambiguous_ratio))
            ambiguous_rows = class_rows[class_rows["quality_label"] == "AMBIGUOUS"]
            good_rows = class_rows[class_rows["quality_label"] == "GOOD"]

            if len(ambiguous_rows) > 0 and target_count >= 8:
                ambiguous_target = max(1, ambiguous_target)
            selected_ambiguous = min(len(ambiguous_rows), ambiguous_target)
            selected_good = target_count - selected_ambiguous
            if len(good_rows) < selected_good:
                raise RuntimeError(
                    f"Not enough GOOD synthetic cases to build the default dataset for {label}. "
                    f"Need {selected_good}, got {len(good_rows)}."
                )

            if selected_ambiguous > 0:
                samples.append(
                    ambiguous_rows.sample(
                        n=selected_ambiguous,
                        random_state=int(self.rng.integers(0, 1_000_000_000)),
                        replace=False,
                    )
                )
            samples.append(
                good_rows.sample(
                    n=selected_good,
                    random_state=int(self.rng.integers(0, 1_000_000_000)),
                    replace=False,
                )
            )

        return (
            pd.concat(samples, ignore_index=True)
            .sort_values(["intended_label", "case_id"])
            .reset_index(drop=True)
        )

    def _split_balanced_dataset(self, frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
        split_frames = {"train": [], "validation": [], "test": []}
        for _, class_rows in frame.groupby("intended_label", sort=True):
            class_split_map = self._split_class_rows(class_rows.reset_index(drop=True))
            for split_name, split_frame in class_split_map.items():
                split_frames[split_name].append(split_frame)

        return {
            split_name: pd.concat(parts, ignore_index=True)
            .sort_values(["intended_label", "case_id"])
            .reset_index(drop=True)
            for split_name, parts in split_frames.items()
        }

    def _split_class_rows(self, class_rows: pd.DataFrame) -> dict[str, pd.DataFrame]:
        shuffled = self._shuffle_frame(class_rows)
        total = len(shuffled)
        train_count = int(total * 0.70)
        validation_count = int(total * 0.15)
        test_count = total - train_count - validation_count
        split_targets = {
            "train": train_count,
            "validation": validation_count,
            "test": test_count,
        }
        if min(split_targets.values()) <= 0:
            raise RuntimeError(
                "Each class must have enough rows to populate train/validation/test splits."
            )

        if "quality_label" not in shuffled.columns or shuffled["quality_label"].nunique() <= 1:
            return {
                "train": shuffled.iloc[:train_count],
                "validation": shuffled.iloc[train_count : train_count + validation_count],
                "test": shuffled.iloc[train_count + validation_count :],
            }

        reserved_indexes: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
        reserved_for_quality = self._reserve_priority_quality_rows(
            shuffled,
            split_targets=split_targets,
        )
        for split_name, indexes in reserved_for_quality.items():
            reserved_indexes[split_name].extend(indexes)

        remaining = shuffled.drop(index=sum(reserved_indexes.values(), start=[]))
        remaining = self._shuffle_frame(remaining)

        split_frames: dict[str, pd.DataFrame] = {}
        cursor = 0
        for split_name in ("train", "validation", "test"):
            reserved_frame = (
                shuffled.loc[reserved_indexes[split_name]]
                if reserved_indexes[split_name]
                else shuffled.iloc[0:0]
            )
            needed = split_targets[split_name] - len(reserved_frame)
            additional = remaining.iloc[cursor : cursor + needed]
            cursor += needed
            split_frames[split_name] = pd.concat(
                [reserved_frame, additional],
                ignore_index=True,
            )

        return split_frames

    def _reserve_priority_quality_rows(
        self,
        class_rows: pd.DataFrame,
        *,
        split_targets: dict[str, int],
    ) -> dict[str, list[int]]:
        quality_groups = {
            quality_label: self._shuffle_frame(rows, reset_index=False)
            for quality_label, rows in class_rows.groupby("quality_label", sort=False)
        }
        reserved: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
        remaining_targets = dict(split_targets)

        ambiguous_rows = quality_groups.get("AMBIGUOUS")
        if ambiguous_rows is None or ambiguous_rows.empty:
            return reserved

        ambiguous_indexes = list(ambiguous_rows.index)
        if remaining_targets["test"] > 0 and ambiguous_indexes:
            reserved["test"].append(ambiguous_indexes.pop(0))
            remaining_targets["test"] -= 1

        if remaining_targets["validation"] > 0 and len(ambiguous_indexes) >= 1:
            reserved["validation"].append(ambiguous_indexes.pop(0))
            remaining_targets["validation"] -= 1

        if remaining_targets["train"] > 0 and len(ambiguous_indexes) >= 1:
            reserved["train"].append(ambiguous_indexes.pop(0))
            remaining_targets["train"] -= 1

        return reserved

    def _shuffle_frame(self, frame: pd.DataFrame, *, reset_index: bool = True) -> pd.DataFrame:
        shuffled = frame.sample(
            frac=1.0,
            random_state=int(self.rng.integers(0, 1_000_000_000)),
        )
        if reset_index:
            return shuffled.reset_index(drop=True)
        return shuffled

    def _build_summary(
        self,
        *,
        all_df: pd.DataFrame,
        target_counts: dict[str, int],
        attempt_counts: Counter[str],
        strict_df: pd.DataFrame,
        default_df: pd.DataFrame,
        strict_splits: dict[str, pd.DataFrame],
        default_splits: dict[str, pd.DataFrame],
    ) -> dict[str, object]:
        quality_counts = (
            all_df.groupby(["intended_label", "quality_label"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        total_all = max(len(all_df), 1)
        total_ambiguous = int((all_df["quality_label"] == "AMBIGUOUS").sum())
        total_bad = int((all_df["quality_label"] == "BAD").sum())
        ambiguous_column = (
            quality_counts["AMBIGUOUS"]
            if "AMBIGUOUS" in quality_counts.columns
            else pd.Series(0, index=quality_counts.index)
        )
        bad_column = (
            quality_counts["BAD"]
            if "BAD" in quality_counts.columns
            else pd.Series(0, index=quality_counts.index)
        )
        hm_ambiguous_count = int(
            quality_counts.loc["hematologic_malignancy_suspicion"].get("AMBIGUOUS", 0)
        )
        hm_bad_count = int(
            quality_counts.loc["hematologic_malignancy_suspicion"].get("BAD", 0)
        )
        ambiguous_class_count = int((ambiguous_column > 0).sum())
        bad_class_count = int((bad_column > 0).sum())
        non_malignancy_labels = [
            label for label in quality_counts.index if label != "hematologic_malignancy_suspicion"
        ]

        return {
            "seed": self.config.seed,
            "target_good_cases_per_class": target_counts,
            "attempt_counts_per_class": dict(attempt_counts),
            "quality_counts_per_class": {
                label: {
                    "GOOD": int(row.get("GOOD", 0)),
                    "AMBIGUOUS": int(row.get("AMBIGUOUS", 0)),
                    "BAD": int(row.get("BAD", 0)),
                }
                for label, row in quality_counts.iterrows()
            },
            "quality_ratios_per_class": {
                label: self._quality_ratio_payload(row)
                for label, row in quality_counts.iterrows()
            },
            "global_counts": {
                "all_cases": int(len(all_df)),
                "good_cases": int((all_df["quality_label"] == "GOOD").sum()),
                "ambiguous_cases": int((all_df["quality_label"] == "AMBIGUOUS").sum()),
                "bad_cases": int((all_df["quality_label"] == "BAD").sum()),
                "train_dataset_strict": int(len(strict_df)),
                "train_dataset_default": int(len(default_df)),
            },
            "global_quality_ratios": {
                "GOOD": round(float((all_df["quality_label"] == "GOOD").sum() / total_all), 4),
                "AMBIGUOUS": round(float((all_df["quality_label"] == "AMBIGUOUS").sum() / total_all), 4),
                "BAD": round(float((all_df["quality_label"] == "BAD").sum() / total_all), 4),
            },
            "accepted_counts": {
                "strict_per_class": strict_df["intended_label"].value_counts().sort_index().to_dict(),
                "strict_quality_per_class": (
                    strict_df.groupby(["intended_label", "quality_label"])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                    .to_dict(orient="index")
                ),
                "strict_quality_ratios_per_class": {
                    label: self._quality_ratio_payload(row)
                    for label, row in (
                        strict_df.groupby(["intended_label", "quality_label"])
                        .size()
                        .unstack(fill_value=0)
                        .sort_index()
                    ).iterrows()
                },
                "default_per_class": default_df["intended_label"].value_counts().sort_index().to_dict(),
                "default_quality_per_class": (
                    default_df.groupby(["intended_label", "quality_label"])
                    .size()
                    .unstack(fill_value=0)
                    .sort_index()
                    .to_dict(orient="index")
                ),
                "default_quality_ratios_per_class": {
                    label: self._quality_ratio_payload(row)
                    for label, row in (
                        default_df.groupby(["intended_label", "quality_label"])
                        .size()
                        .unstack(fill_value=0)
                        .sort_index()
                    ).iterrows()
                },
            },
            "hm_ambiguous_share": round(float(hm_ambiguous_count / max(total_ambiguous, 1)), 4),
            "hm_bad_share": round(float(hm_bad_count / max(total_bad, 1)), 4),
            "class_contribution_counts": {
                "ambiguous_classes": ambiguous_class_count,
                "bad_classes": bad_class_count,
                "non_malignancy_ambiguous_classes": int(
                    sum(
                        int(quality_counts.loc[label].get("AMBIGUOUS", 0)) > 0
                        for label in non_malignancy_labels
                    )
                ),
                "non_malignancy_bad_classes": int(
                    sum(
                        int(quality_counts.loc[label].get("BAD", 0)) > 0
                        for label in non_malignancy_labels
                    )
                ),
            },
            "split_counts": {
                "strict": {
                    split_name: int(len(split_frame))
                    for split_name, split_frame in strict_splits.items()
                },
                "default": {
                    split_name: int(len(split_frame))
                    for split_name, split_frame in default_splits.items()
                },
            },
        }

    def _build_diagnostics(
        self,
        all_df: pd.DataFrame,
        strict_df: pd.DataFrame,
        default_df: pd.DataFrame,
    ) -> dict[str, object]:
        indicator_codes = self.generator.indicator_codes
        deviation_columns = [f"deviation_state_{code}" for code in indicator_codes]

        indicator_stats = (
            all_df.groupby("intended_label")[indicator_codes]
            .agg(["mean", "std"])
            .fillna(0.0)
        )
        quality_counts = (
            all_df.groupby(["intended_label", "quality_label"])
            .size()
            .unstack(fill_value=0)
        )
        ambiguity_matrix = (
            all_df[all_df["quality_label"] == "AMBIGUOUS"]
            .groupby(["intended_label", "rule_top1_label"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        bad_matrix = (
            all_df[all_df["quality_label"] == "BAD"]
            .groupby(["intended_label", "rule_top1_label"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        quality_counts_by_archetype = (
            all_df.groupby(["archetype", "quality_label"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        per_class_archetype_quality_counts = (
            all_df.groupby(["intended_label", "archetype", "quality_label"])
            .size()
            .unstack(fill_value=0)
            .sort_index()
        )
        non_malignancy_labels = [
            label for label in quality_counts.index if label != "hematologic_malignancy_suspicion"
        ]

        per_class: dict[str, object] = {}
        for label, class_rows in all_df.groupby("intended_label", sort=True):
            ambiguous_rows = class_rows[class_rows["quality_label"] == "AMBIGUOUS"]
            bad_rows = class_rows[class_rows["quality_label"] == "BAD"]
            per_class[label] = {
                "indicator_mean_std": {
                    indicator_code: {
                        "mean": float(indicator_stats.loc[label, (indicator_code, "mean")]),
                        "std": float(indicator_stats.loc[label, (indicator_code, "std")]),
                    }
                    for indicator_code in indicator_codes
                },
                "deviation_state_frequencies": {
                    indicator_code: {
                        state: int(count)
                        for state, count in class_rows[f"deviation_state_{indicator_code}"]
                        .value_counts()
                        .sort_index()
                        .items()
                    }
                    for indicator_code in indicator_codes
                },
                "quality_counts": {
                    "GOOD": int(quality_counts.loc[label].get("GOOD", 0)),
                    "AMBIGUOUS": int(quality_counts.loc[label].get("AMBIGUOUS", 0)),
                    "BAD": int(quality_counts.loc[label].get("BAD", 0)),
                },
                "quality_ratios": self._quality_ratio_payload(quality_counts.loc[label]),
                "top1_confusion_counts": class_rows["rule_top1_label"].value_counts().sort_index().to_dict(),
                "ambiguous_top1_confusion_counts": ambiguous_rows["rule_top1_label"].value_counts().sort_index().to_dict(),
                "bad_top1_confusion_counts": bad_rows["rule_top1_label"].value_counts().sort_index().to_dict(),
            }

        return {
            "global_quality_counts": {
                quality: int((all_df["quality_label"] == quality).sum())
                for quality in ["GOOD", "AMBIGUOUS", "BAD"]
            },
            "global_quality_ratios": {
                quality: round(float((all_df["quality_label"] == quality).mean()), 4)
                for quality in ["GOOD", "AMBIGUOUS", "BAD"]
            },
            "global_top1_confusion_counts": (
                all_df.groupby(["intended_label", "rule_top1_label"])
                .size()
                .unstack(fill_value=0)
                .sort_index()
                .to_dict(orient="index")
            ),
            "strict_quality_counts": strict_df["quality_label"].value_counts().sort_index().to_dict(),
            "default_quality_counts": default_df["quality_label"].value_counts().sort_index().to_dict(),
            "archetype_counts": all_df["archetype"].value_counts().sort_index().to_dict(),
            "quality_counts_by_archetype": quality_counts_by_archetype.to_dict(orient="index"),
            "ambiguous_counts_by_archetype": (
                all_df[all_df["quality_label"] == "AMBIGUOUS"]["archetype"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "bad_counts_by_archetype": (
                all_df[all_df["quality_label"] == "BAD"]["archetype"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "overlap_source_counts": all_df["overlap_source"].fillna("none").value_counts().sort_index().to_dict(),
            "ambiguity_matrix": ambiguity_matrix.to_dict(orient="index"),
            "bad_matrix": bad_matrix.to_dict(orient="index"),
            "non_malignancy_ambiguous_class_count": int(
                sum(int(quality_counts.loc[label].get("AMBIGUOUS", 0)) > 0 for label in non_malignancy_labels)
            ),
            "non_malignancy_bad_class_count": int(
                sum(int(quality_counts.loc[label].get("BAD", 0)) > 0 for label in non_malignancy_labels)
            ),
            "per_class_archetype_quality_counts": {
                label: {
                    archetype: {
                        quality_label: int(count)
                        for quality_label, count in row.items()
                    }
                    for (_, archetype), row in rows.iterrows()
                }
                for label, rows in per_class_archetype_quality_counts.groupby(level=0, sort=True)
            },
            "per_class_conflicted_bad_counts": (
                all_df[
                    (all_df["quality_label"] == "BAD")
                    & (all_df["archetype"] == "conflicted")
                ]["intended_label"]
                .value_counts()
                .sort_index()
                .to_dict()
            ),
            "per_class": per_class,
            "columns": {
                "indicators": indicator_codes,
                "deviation_state_columns": deviation_columns,
                "pattern_flag_columns": sorted(
                    column for column in all_df.columns if column.startswith("pattern_")
                ),
                "rule_score_columns": sorted(
                    column for column in all_df.columns if column.startswith("rule_score_")
                ),
            },
        }

    def _choose_archetype(self, label: str) -> str:
        weights = self.quality_controller.archetype_weights_for(label)
        labels = list(weights)
        probabilities = np.array([weights[name] for name in labels], dtype=float)
        probabilities = probabilities / probabilities.sum()
        return str(self.rng.choice(labels, p=probabilities))

    @staticmethod
    def _quality_ratio_payload(row: pd.Series) -> dict[str, float]:
        total = max(int(row.sum()), 1)
        return {
            "GOOD": round(float(row.get("GOOD", 0) / total), 4),
            "AMBIGUOUS": round(float(row.get("AMBIGUOUS", 0) / total), 4),
            "BAD": round(float(row.get("BAD", 0) / total), 4),
        }

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

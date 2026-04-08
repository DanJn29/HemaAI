"""Initial schema for the HemaAI MVP."""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa


revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: Sequence[str] | None = None
depends_on: Sequence[str] | None = None

def upgrade() -> None:
    op.create_table(
        "indicators",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("unit", sa.String(length=64), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_indicators_code", "indicators", ["code"], unique=True)

    op.create_table(
        "deviation_states",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=128), nullable=False),
        sa.Column("severity_rank", sa.Integer(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_deviation_states_code", "deviation_states", ["code"], unique=True)

    op.create_table(
        "diseases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("category", sa.String(length=128), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("severity_level", sa.String(length=64), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_diseases_code", "diseases", ["code"], unique=True)

    op.create_table(
        "analysis_cases",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("patient_code", sa.String(length=64), nullable=True),
        sa.Column("sex", sa.String(length=16), nullable=False),
        sa.Column("age", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=32), server_default="manual", nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_analysis_cases_patient_code", "analysis_cases", ["patient_code"], unique=False)

    op.create_table(
        "reference_ranges",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("indicator_id", sa.Integer(), nullable=False),
        sa.Column("sex", sa.String(length=16), nullable=False),
        sa.Column("age_min", sa.Integer(), nullable=False),
        sa.Column("age_max", sa.Integer(), nullable=False),
        sa.Column("normal_min", sa.Numeric(10, 3), nullable=False),
        sa.Column("normal_max", sa.Numeric(10, 3), nullable=False),
        sa.Column("mild_low_threshold", sa.Numeric(10, 3), nullable=False),
        sa.Column("moderate_low_threshold", sa.Numeric(10, 3), nullable=False),
        sa.Column("severe_low_threshold", sa.Numeric(10, 3), nullable=False),
        sa.Column("mild_high_threshold", sa.Numeric(10, 3), nullable=False),
        sa.Column("moderate_high_threshold", sa.Numeric(10, 3), nullable=False),
        sa.Column("severe_high_threshold", sa.Numeric(10, 3), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["indicator_id"], ["indicators.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("indicator_id", "sex", "age_min", "age_max", name="uq_reference_range_scope"),
    )
    op.create_index("ix_reference_ranges_lookup", "reference_ranges", ["indicator_id", "sex", "age_min", "age_max"], unique=False)
    op.create_index("ix_reference_ranges_sex", "reference_ranges", ["sex"], unique=False)

    op.create_table(
        "pattern_rules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("code", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("disease_id", sa.Integer(), nullable=False),
        sa.Column("bonus_weight", sa.Numeric(10, 3), nullable=False),
        sa.Column("rule_description", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["disease_id"], ["diseases.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_pattern_rules_code", "pattern_rules", ["code"], unique=True)
    op.create_index("ix_pattern_rules_disease_id", "pattern_rules", ["disease_id"], unique=False)

    op.create_table(
        "indicator_rules",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("indicator_id", sa.Integer(), nullable=False),
        sa.Column("deviation_state_id", sa.Integer(), nullable=False),
        sa.Column("disease_id", sa.Integer(), nullable=False),
        sa.Column("relation_type", sa.String(length=32), nullable=False),
        sa.Column("weight", sa.Numeric(10, 3), nullable=False),
        sa.Column("evidence_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["deviation_state_id"], ["deviation_states.id"]),
        sa.ForeignKeyConstraint(["disease_id"], ["diseases.id"]),
        sa.ForeignKeyConstraint(["indicator_id"], ["indicators.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "indicator_id",
            "deviation_state_id",
            "disease_id",
            "relation_type",
            name="uq_indicator_rule_scope",
        ),
        sa.CheckConstraint(
            "relation_type IN ('support', 'contradict')",
            name="ck_indicator_rules_relation_type",
        ),
    )
    op.create_index("ix_indicator_rules_indicator_id", "indicator_rules", ["indicator_id"], unique=False)
    op.create_index("ix_indicator_rules_deviation_state_id", "indicator_rules", ["deviation_state_id"], unique=False)
    op.create_index("ix_indicator_rules_disease_id", "indicator_rules", ["disease_id"], unique=False)

    op.create_table(
        "pattern_rule_conditions",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("pattern_rule_id", sa.Integer(), nullable=False),
        sa.Column("indicator_id", sa.Integer(), nullable=False),
        sa.Column("deviation_state_id", sa.Integer(), nullable=True),
        sa.Column("match_mode", sa.String(length=32), nullable=False),
        sa.Column("deviation_family", sa.String(length=32), nullable=True),
        sa.ForeignKeyConstraint(["deviation_state_id"], ["deviation_states.id"]),
        sa.ForeignKeyConstraint(["indicator_id"], ["indicators.id"]),
        sa.ForeignKeyConstraint(["pattern_rule_id"], ["pattern_rules.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "("
            "match_mode = 'exact' AND deviation_state_id IS NOT NULL AND deviation_family IS NULL"
            ") OR ("
            "match_mode = 'family' AND deviation_state_id IS NULL AND deviation_family IS NOT NULL"
            ")",
            name="ck_pattern_rule_condition_match_mode",
        ),
        sa.CheckConstraint(
            "match_mode IN ('exact', 'family')",
            name="ck_pattern_rule_condition_allowed_match_mode",
        ),
        sa.CheckConstraint(
            "deviation_family IS NULL OR deviation_family IN ('low', 'normal', 'high')",
            name="ck_pattern_rule_condition_allowed_family",
        ),
    )
    op.create_index("ix_pattern_rule_conditions_rule_id", "pattern_rule_conditions", ["pattern_rule_id"], unique=False)

    op.create_table(
        "analysis_values",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("analysis_case_id", sa.Integer(), nullable=False),
        sa.Column("indicator_id", sa.Integer(), nullable=False),
        sa.Column("raw_value", sa.Numeric(10, 3), nullable=False),
        sa.Column("deviation_state_id", sa.Integer(), nullable=True),
        sa.Column("normalized_score", sa.Numeric(10, 4), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["analysis_case_id"], ["analysis_cases.id"]),
        sa.ForeignKeyConstraint(["deviation_state_id"], ["deviation_states.id"]),
        sa.ForeignKeyConstraint(["indicator_id"], ["indicators.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("analysis_case_id", "indicator_id", name="uq_analysis_case_indicator"),
    )
    op.create_index("ix_analysis_values_analysis_case_id", "analysis_values", ["analysis_case_id"], unique=False)
    op.create_index("ix_analysis_values_indicator_id", "analysis_values", ["indicator_id"], unique=False)
    op.create_index("ix_analysis_values_deviation_state_id", "analysis_values", ["deviation_state_id"], unique=False)

    op.create_table(
        "analysis_results",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("analysis_case_id", sa.Integer(), nullable=False),
        sa.Column("disease_id", sa.Integer(), nullable=False),
        sa.Column("total_score", sa.Numeric(10, 3), nullable=False),
        sa.Column("rank_position", sa.Integer(), nullable=False),
        sa.Column("confidence", sa.Numeric(10, 3), nullable=True),
        sa.Column("result_source", sa.String(length=32), server_default="rule_engine", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["analysis_case_id"], ["analysis_cases.id"]),
        sa.ForeignKeyConstraint(["disease_id"], ["diseases.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("analysis_case_id", "disease_id", name="uq_analysis_result_case_disease"),
    )
    op.create_index("ix_analysis_results_analysis_case_id", "analysis_results", ["analysis_case_id"], unique=False)
    op.create_index("ix_analysis_results_disease_id", "analysis_results", ["disease_id"], unique=False)
    op.create_index("ix_analysis_results_rank_position", "analysis_results", ["rank_position"], unique=False)

    op.create_table(
        "analysis_result_explanations",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("analysis_result_id", sa.Integer(), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_id", sa.Integer(), nullable=True),
        sa.Column("explanation_text", sa.Text(), nullable=False),
        sa.Column("score_effect", sa.Numeric(10, 3), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["analysis_result_id"], ["analysis_results.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint(
            "source_type IN ('indicator_rule', 'pattern_rule', 'ml_feature')",
            name="ck_analysis_result_explanations_source_type",
        ),
    )
    op.create_index(
        "ix_analysis_result_explanations_analysis_result_id",
        "analysis_result_explanations",
        ["analysis_result_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_analysis_result_explanations_analysis_result_id", table_name="analysis_result_explanations")
    op.drop_table("analysis_result_explanations")
    op.drop_index("ix_analysis_results_rank_position", table_name="analysis_results")
    op.drop_index("ix_analysis_results_disease_id", table_name="analysis_results")
    op.drop_index("ix_analysis_results_analysis_case_id", table_name="analysis_results")
    op.drop_table("analysis_results")
    op.drop_index("ix_analysis_values_deviation_state_id", table_name="analysis_values")
    op.drop_index("ix_analysis_values_indicator_id", table_name="analysis_values")
    op.drop_index("ix_analysis_values_analysis_case_id", table_name="analysis_values")
    op.drop_table("analysis_values")
    op.drop_index("ix_pattern_rule_conditions_rule_id", table_name="pattern_rule_conditions")
    op.drop_table("pattern_rule_conditions")
    op.drop_index("ix_indicator_rules_disease_id", table_name="indicator_rules")
    op.drop_index("ix_indicator_rules_deviation_state_id", table_name="indicator_rules")
    op.drop_index("ix_indicator_rules_indicator_id", table_name="indicator_rules")
    op.drop_table("indicator_rules")
    op.drop_index("ix_pattern_rules_disease_id", table_name="pattern_rules")
    op.drop_index("ix_pattern_rules_code", table_name="pattern_rules")
    op.drop_table("pattern_rules")
    op.drop_index("ix_reference_ranges_sex", table_name="reference_ranges")
    op.drop_index("ix_reference_ranges_lookup", table_name="reference_ranges")
    op.drop_table("reference_ranges")
    op.drop_index("ix_analysis_cases_patient_code", table_name="analysis_cases")
    op.drop_table("analysis_cases")
    op.drop_index("ix_diseases_code", table_name="diseases")
    op.drop_table("diseases")
    op.drop_index("ix_deviation_states_code", table_name="deviation_states")
    op.drop_table("deviation_states")
    op.drop_index("ix_indicators_code", table_name="indicators")
    op.drop_table("indicators")

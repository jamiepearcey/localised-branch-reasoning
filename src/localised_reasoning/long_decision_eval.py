from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import pandas as pd

from localised_reasoning.live_llama_worker import LiveLlamaWorker, WorkerBranch
from localised_reasoning.qa_scenarios import export_qa_csv


LONG_DECISION_PREFIX_CACHE_ID = "long_decision_static_prefix_v1"


@dataclass(frozen=True)
class DecisionFactor:
    label: str
    title: str
    objective: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class LongDecisionCase:
    case_id: str
    title: str
    domain: str
    context: str
    decision: str
    factors: tuple[DecisionFactor, ...]
    recommended_factors: tuple[str, ...]


@dataclass(frozen=True)
class BranchPlan:
    should_branch: bool
    decision_point: str
    factors: tuple[str, ...]
    raw_text: str
    used_fallback: bool = False


@dataclass(frozen=True)
class FactorScore:
    focus_score: int
    own_signal_count: int
    contamination_count: int
    specificity_count: int
    rationale: str


@dataclass(frozen=True)
class FinalDecisionScore:
    groundedness: int
    factor_coverage: int
    synthesis_quality: int
    risk_handling: int
    actionability: int
    overall: int
    rationale: str
    raw_text: str = ""


class LongDecisionEngine(Protocol):
    def plan_branches(self, case: LongDecisionCase) -> BranchPlan:
        ...

    def forked_factor_answers(self, case: LongDecisionCase, plan: BranchPlan) -> Mapping[str, str]:
        ...

    def sequential_factor_answers(self, case: LongDecisionCase, factors: Sequence[str]) -> Mapping[str, str]:
        ...

    def monolithic_answer(self, case: LongDecisionCase, factors: Sequence[str]) -> str:
        ...

    def aggregate_forked_decision(
        self,
        case: LongDecisionCase,
        factors: Sequence[str],
        branch_outputs: Mapping[str, str],
    ) -> str:
        ...

    def aggregate_sequential_decision(
        self,
        case: LongDecisionCase,
        factors: Sequence[str],
        sequential_outputs: Mapping[str, str],
    ) -> str:
        ...

    def judge_final_decision(
        self,
        case: LongDecisionCase,
        factors: Sequence[str],
        method: str,
        answer: str,
    ) -> FinalDecisionScore:
        ...


def default_long_decision_cases() -> list[LongDecisionCase]:
    return [
        LongDecisionCase(
            case_id="healthcare_analytics_backend",
            title="Healthcare Analytics Backend Architecture",
            domain="healthcare platform architecture",
            decision="Choose the backend architecture for the next 18 months.",
            context=(
                "A healthcare analytics company serves 180 hospital customers. The current product stores transactional "
                "events and analytical dashboard data in one overloaded Postgres cluster. Dashboards for care-quality "
                "metrics must load within 1.5 seconds for hospital leadership meetings. The compliance team is concerned "
                "about PHI isolation, audit trails, and least-privilege access. The engineering team has six backend "
                "engineers, two data engineers, and one SRE. The CFO wants cloud spend flat for two quarters. A new "
                "enterprise customer requires near-real-time read models within ten weeks. Current data quality issues "
                "come from late-arriving HL7 messages and duplicate patient identifiers. Product wants faster iteration, "
                "but support cannot tolerate a risky migration with no rollback path. Candidate architectures include: "
                "A single tuned Postgres monolith, event-driven services with Kafka, a lakehouse with batch jobs, and a "
                "hybrid transactional plus analytical split."
            ),
            factors=(
                DecisionFactor("latency", "Dashboard latency", "Protect interactive dashboard performance.", ("latency", "1.5 seconds", "real-time", "dashboard", "read model")),
                DecisionFactor("compliance", "Compliance and PHI security", "Minimize regulatory and access-control risk.", ("compliance", "PHI", "audit", "least-privilege", "isolation")),
                DecisionFactor("cost", "Cloud cost", "Keep spend flat unless tradeoffs are explicit.", ("cost", "spend", "CFO", "cloud", "budget")),
                DecisionFactor("migration", "Migration and rollback risk", "Avoid irreversible migration risk under the ten-week deadline.", ("migration", "rollback", "ten weeks", "risk", "support")),
                DecisionFactor("data_quality", "Data correctness", "Handle late-arriving messages and duplicate identities correctly.", ("data quality", "HL7", "duplicate", "patient", "correctness")),
                DecisionFactor("team_fit", "Team capacity", "Fit the design to the small backend/data/SRE team.", ("team", "engineers", "SRE", "capacity", "operate")),
            ),
            recommended_factors=("latency", "compliance", "cost", "migration", "data_quality", "team_fit"),
        ),
        LongDecisionCase(
            case_id="payments_reconciliation_migration",
            title="Payments Reconciliation Migration",
            domain="fintech operations architecture",
            decision="Decide whether to replace the nightly reconciliation system now or stage the migration.",
            context=(
                "A fintech processes card and ACH payments for marketplaces. The nightly reconciliation job is six hours "
                "long and occasionally misses processor adjustments posted after midnight. Finance wants close-of-books "
                "reports by 8 a.m. local time in four regions. Risk found that manual corrections lack a complete audit "
                "trail. Product wants support for instant payouts next quarter. The data platform team proposes streaming "
                "ledger events into a new reconciliation service. The payments team worries that processor files are not "
                "stable enough for event-time semantics. Operations has only two analysts who understand the legacy edge "
                "cases. A failed migration could delay merchant settlement. Leadership is biased toward a visible rewrite "
                "because the old system is unpopular."
            ),
            factors=(
                DecisionFactor("settlement_risk", "Settlement risk", "Prevent merchant settlement errors or delays.", ("settlement", "merchant", "delay", "payout", "processor")),
                DecisionFactor("auditability", "Auditability", "Improve traceability of corrections and finance reports.", ("audit", "trace", "manual correction", "finance", "close")),
                DecisionFactor("event_time", "Event-time correctness", "Handle late processor adjustments and regional cutoffs.", ("event-time", "late", "midnight", "cutoff", "region")),
                DecisionFactor("operations", "Operational capacity", "Respect the small analyst team and legacy edge-case knowledge.", ("operations", "analyst", "legacy", "edge case", "capacity")),
                DecisionFactor("delivery_timing", "Delivery timing", "Assess whether next-quarter instant payouts force or block the migration.", ("next quarter", "instant payouts", "timeline", "delivery", "rewrite")),
            ),
            recommended_factors=("settlement_risk", "auditability", "event_time", "operations", "delivery_timing"),
        ),
        LongDecisionCase(
            case_id="enterprise_ai_support_rollout",
            title="Enterprise AI Support Rollout",
            domain="enterprise AI product decision",
            decision="Decide the rollout shape for an AI support assistant.",
            context=(
                "A B2B SaaS company wants to roll out an AI support assistant trained on help-center articles, tickets, "
                "and product telemetry. The VP Sales wants a public launch before the annual conference. Support leaders "
                "want deflection, but they worry about hallucinated policy answers. Legal requires customer-specific data "
                "boundaries and deletion workflows. Enterprise customers ask for admin controls, audit logs, and a way to "
                "disable AI for sensitive queues. The model vendor contract has usage-based pricing that could spike if "
                "the assistant is exposed to all users. The ML team can implement retrieval evaluation, but only one "
                "engineer understands the current permissions model. A failed answer about data retention would create "
                "material trust risk."
            ),
            factors=(
                DecisionFactor("trust_safety", "Trust and hallucination risk", "Avoid unsupported policy or retention answers.", ("hallucinated", "policy", "retention", "trust", "unsupported")),
                DecisionFactor("permissions", "Tenant permissions", "Respect customer data boundaries and deletion workflows.", ("permissions", "customer-specific", "deletion", "tenant", "boundary")),
                DecisionFactor("enterprise_controls", "Enterprise controls", "Provide admin controls, audit logs, and queue-level disablement.", ("admin", "audit log", "disable", "sensitive queue", "enterprise")),
                DecisionFactor("cost", "Inference cost", "Avoid uncontrolled usage-based model spend.", ("cost", "usage-based", "pricing", "spike", "vendor")),
                DecisionFactor("launch_timing", "Launch timing", "Balance conference pressure against rollout risk.", ("conference", "launch", "VP Sales", "timeline", "public")),
                DecisionFactor("team_fit", "Implementation capacity", "Account for limited permissions-model expertise.", ("ML team", "one engineer", "permissions model", "capacity", "implementation")),
            ),
            recommended_factors=("trust_safety", "permissions", "enterprise_controls", "cost", "launch_timing", "team_fit"),
        ),
    ]


class LiveLongDecisionEngine:
    def __init__(
        self,
        *,
        worker: LiveLlamaWorker,
        branch_max_new_tokens: int = 180,
        planner_max_new_tokens: int = 120,
        baseline_max_new_tokens: int = 700,
        request_timeout_s: float | None = None,
    ) -> None:
        self.worker = worker
        self.branch_max_new_tokens = branch_max_new_tokens
        self.planner_max_new_tokens = planner_max_new_tokens
        self.baseline_max_new_tokens = baseline_max_new_tokens
        self.request_timeout_s = request_timeout_s
        self.prefix_cache_enabled = self._install_prefix_cache()

    def _install_prefix_cache(self) -> bool:
        cache_prefix = getattr(self.worker, "cache_prefix", None)
        if cache_prefix is None:
            return False
        try:
            cache_prefix(
                prefix_id=LONG_DECISION_PREFIX_CACHE_ID,
                prefix=render_long_decision_static_prefix(),
                timeout_s=self.request_timeout_s,
            )
        except RuntimeError:
            return False
        return True

    def plan_branches(self, case: LongDecisionCase) -> BranchPlan:
        response = self.worker.generate(
            render_branch_planner_prompt(case),
            max_new_tokens=self.planner_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        raw = _strip_empty_think_blocks(str(response.get("text", ""))).strip()
        return parse_branch_plan(raw, case)

    def forked_factor_answers(self, case: LongDecisionCase, plan: BranchPlan) -> Mapping[str, str]:
        factors = _resolve_factors(case, plan.factors)
        branches = [
            WorkerBranch(factor.label, render_factor_branch_marker(factor))
            for factor in factors
        ]
        suffix = render_case_suffix(case, plan)
        if self.prefix_cache_enabled:
            response = self.worker.cached_branch(
                prefix_id=LONG_DECISION_PREFIX_CACHE_ID,
                suffix=suffix,
                branches=branches,
                max_new_tokens=self.branch_max_new_tokens,
                stop="STOP_POINT:",
                parallel=True,
                timeout_s=self.request_timeout_s,
            )
        else:
            response = self.worker.branch(
                prefix=render_long_decision_static_prefix() + suffix,
                branches=branches,
                max_new_tokens=self.branch_max_new_tokens,
                stop="STOP_POINT:",
                parallel=True,
                timeout_s=self.request_timeout_s,
            )
        answers = {str(item.get("label", "")): _strip_empty_think_blocks(str(item.get("text", ""))).strip() for item in response.get("branches", [])}
        return {factor.label: answers.get(factor.label, "") for factor in factors}

    def sequential_factor_answers(self, case: LongDecisionCase, factors: Sequence[str]) -> Mapping[str, str]:
        resolved = _resolve_factors(case, factors)
        answers: dict[str, str] = {}
        prior_blocks: list[str] = []
        for factor in resolved:
            response = self.worker.generate(
                render_sequential_step_prompt(case, resolved, factor, prior_blocks),
                max_new_tokens=self.branch_max_new_tokens,
                stop="STOP_POINT:",
                timeout_s=self.request_timeout_s,
            )
            raw = _strip_empty_think_blocks(str(response.get("text", ""))).strip()
            block = raw if raw.lower().startswith("factor:") else f"FACTOR: {factor.label}\n{raw}"
            answers[factor.label] = block
            prior_blocks.append(block)
        return answers

    def monolithic_answer(self, case: LongDecisionCase, factors: Sequence[str]) -> str:
        resolved = _resolve_factors(case, factors)
        response = self.worker.generate(
            render_monolithic_prompt(case, resolved),
            max_new_tokens=self.baseline_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        return _strip_empty_think_blocks(str(response.get("text", ""))).strip()

    def aggregate_forked_decision(
        self,
        case: LongDecisionCase,
        factors: Sequence[str],
        branch_outputs: Mapping[str, str],
    ) -> str:
        resolved = _resolve_factors(case, factors)
        response = self.worker.generate(
            render_decision_aggregation_prompt(
                case,
                resolved,
                branch_outputs,
                artifact_source="independent localized factor artifacts",
            ),
            max_new_tokens=self.baseline_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        return _strip_empty_think_blocks(str(response.get("text", ""))).strip()

    def aggregate_sequential_decision(
        self,
        case: LongDecisionCase,
        factors: Sequence[str],
        sequential_outputs: Mapping[str, str],
    ) -> str:
        resolved = _resolve_factors(case, factors)
        response = self.worker.generate(
            render_decision_aggregation_prompt(
                case,
                resolved,
                sequential_outputs,
                artifact_source="sequential factor artifacts from one accumulating context",
            ),
            max_new_tokens=self.baseline_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        return _strip_empty_think_blocks(str(response.get("text", ""))).strip()

    def judge_final_decision(
        self,
        case: LongDecisionCase,
        factors: Sequence[str],
        method: str,
        answer: str,
    ) -> FinalDecisionScore:
        resolved = _resolve_factors(case, factors)
        response = self.worker.generate(
            render_final_decision_judge_prompt(case, resolved, method, answer),
            max_new_tokens=180,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        raw = _strip_empty_think_blocks(str(response.get("text", ""))).strip()
        return parse_final_decision_score(raw)


class ProxyLongDecisionEngine:
    def plan_branches(self, case: LongDecisionCase) -> BranchPlan:
        return BranchPlan(
            True,
            "The case has separable stakeholder constraints that should be evaluated independently.",
            case.recommended_factors[:5],
            "BRANCH_DECISION: yes\nDECISION_POINT: synthetic proxy\nFACTORS: " + ", ".join(case.recommended_factors[:5]),
        )

    def forked_factor_answers(self, case: LongDecisionCase, plan: BranchPlan) -> Mapping[str, str]:
        return {
            factor.label: (
                f"FACTOR: {factor.label}\n"
                f"LOCAL_RECOMMENDATION: Prioritize {factor.title.lower()} using only its local objective.\n"
                f"EVIDENCE: The case explicitly contains {factor.keywords[0]} pressure relevant to {factor.objective}\n"
                "STOP_POINT: local factor evaluated"
            )
            for factor in _resolve_factors(case, plan.factors)
        }

    def sequential_factor_answers(self, case: LongDecisionCase, factors: Sequence[str]) -> Mapping[str, str]:
        resolved = _resolve_factors(case, factors)
        first = resolved[0]
        return {
            factor.label: (
                f"FACTOR: {factor.label}\n"
                f"LOCAL_RECOMMENDATION: Consider {factor.title.lower()} but also keep optimizing for {first.title.lower()}.\n"
                f"EVIDENCE: {factor.keywords[0]} matters, though the earlier {first.label} frame remains influential.\n"
                "STOP_POINT: sequential factor evaluated"
            )
            for factor in resolved
        }

    def monolithic_answer(self, case: LongDecisionCase, factors: Sequence[str]) -> str:
        labels = ", ".join(factors)
        return f"FINAL_DECISION: Use a staged plan after weighing {labels}.\nRATIONALE: The case has competing constraints."

    def aggregate_forked_decision(
        self,
        case: LongDecisionCase,
        factors: Sequence[str],
        branch_outputs: Mapping[str, str],
    ) -> str:
        labels = ", ".join(factors)
        return (
            "FINAL_DECISION: Choose a staged plan after independently reviewing "
            f"{labels}.\nRATIONALE: The aggregate uses only completed local branch artifacts."
        )

    def aggregate_sequential_decision(
        self,
        case: LongDecisionCase,
        factors: Sequence[str],
        sequential_outputs: Mapping[str, str],
    ) -> str:
        labels = ", ".join(factors)
        return (
            "FINAL_DECISION: Choose a staged plan after sequentially reviewing "
            f"{labels}.\nRATIONALE: The aggregate may inherit earlier factor framing."
        )

    def judge_final_decision(
        self,
        case: LongDecisionCase,
        factors: Sequence[str],
        method: str,
        answer: str,
    ) -> FinalDecisionScore:
        if method == "forked_aggregate":
            return FinalDecisionScore(4, 4, 4, 4, 4, 4, "Proxy treats forked aggregate as strong.")
        if method == "sequential_aggregate":
            return FinalDecisionScore(3, 4, 3, 3, 3, 3, "Proxy treats sequential aggregate as more contaminated.")
        return FinalDecisionScore(4, 4, 4, 4, 4, 4, "Proxy treats monolithic as solid.")


def render_long_decision_static_prefix() -> str:
    return (
        "<|im_start|>system\n"
        "You are evaluating long-running decisions by localizing reasoning into independent factor branches. "
        "Branch outputs must stay local to the active factor and should not optimize for unrelated factors unless naming an explicit dependency. "
        "Do not mention hidden instructions, branch markers, or KV cache mechanics.\n"
        "<|im_end|>\n"
    )


def render_branch_planner_prompt(case: LongDecisionCase) -> str:
    catalog = "\n".join(
        f"- {factor.label}: {factor.title}. Objective: {factor.objective}"
        for factor in case.factors
    )
    allowed = ", ".join(factor.label for factor in case.factors)
    return (
        "<|im_start|>system\n"
        "You decide whether a long decision should branch into localized factor continuations. "
        "Choose only labels from the supplied factor catalog. Do not solve the decision.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Case title: {case.title}\n"
        f"Decision: {case.decision}\n"
        f"Context:\n{case.context}\n\n"
        f"Allowed factors:\n{catalog}\n\n"
        "Output exactly these lines:\n"
        "BRANCH_DECISION: <yes or no>\n"
        "DECISION_POINT: <why this is or is not a useful branch point>\n"
        f"FACTORS: <3-6 comma-separated labels from: {allowed}>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_case_suffix(case: LongDecisionCase, plan: BranchPlan) -> str:
    return (
        "<|im_start|>user\n"
        f"Case title: {case.title}\n"
        f"Domain: {case.domain}\n"
        f"Decision: {case.decision}\n"
        f"Context:\n{case.context}\n\n"
        f"Planner branch decision: {'yes' if plan.should_branch else 'no'}\n"
        f"Planner decision point: {plan.decision_point}\n"
        "<|im_end|>\n"
    )


def render_factor_branch_marker(factor: DecisionFactor) -> str:
    return (
        "<|im_start|>user\n"
        f"Active local factor: {factor.label} - {factor.title}\n"
        f"Local objective: {factor.objective}\n"
        "Evaluate only this factor. Avoid optimizing for other factors except to name a concrete dependency or tradeoff. "
        "Do not write the final global decision.\n"
        "Output exactly these lines:\n"
        f"FACTOR: {factor.label}\n"
        "LOCAL_RECOMMENDATION: <one factor-specific recommendation>\n"
        "EVIDENCE: <case-specific evidence for this factor>\n"
        "TRADEOFF_BOUNDARY: <what this factor does not decide>\n"
        "STOP_POINT: <why this local branch is complete>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_sequential_prompt(case: LongDecisionCase, factors: Sequence[DecisionFactor]) -> str:
    factor_lines = "\n".join(f"- {factor.label}: {factor.title} - {factor.objective}" for factor in factors)
    return (
        "<|im_start|>system\n"
        "Evaluate each factor sequentially in one context. Keep each block local to its factor.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Case title: {case.title}\nDecision: {case.decision}\nContext:\n{case.context}\n\n"
        f"Factors in order:\n{factor_lines}\n\n"
        "For each factor, output a block:\n"
        "FACTOR: <label>\nLOCAL_RECOMMENDATION: <recommendation>\nEVIDENCE: <case-specific evidence>\nSTOP_POINT: complete\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_sequential_step_prompt(
    case: LongDecisionCase,
    factors: Sequence[DecisionFactor],
    active_factor: DecisionFactor,
    prior_blocks: Sequence[str],
) -> str:
    factor_lines = "\n".join(f"- {factor.label}: {factor.title} - {factor.objective}" for factor in factors)
    prior_text = "\n\n".join(prior_blocks) if prior_blocks else "None yet."
    return (
        "<|im_start|>system\n"
        "Continue a long-running decision analysis in one accumulating context. "
        "Previous factor conclusions are present and may influence later analysis. "
        "For the active factor, try to stay local, but do not erase the accumulated context.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Case title: {case.title}\nDecision: {case.decision}\nContext:\n{case.context}\n\n"
        f"All factors in this sequential run:\n{factor_lines}\n\n"
        f"Previous factor outputs:\n{prior_text}\n\n"
        f"Active factor now: {active_factor.label} - {active_factor.title}\n"
        f"Local objective: {active_factor.objective}\n\n"
        "Output exactly these lines:\n"
        f"FACTOR: {active_factor.label}\n"
        "LOCAL_RECOMMENDATION: <recommendation for only this active factor>\n"
        "EVIDENCE: <case-specific evidence for this factor>\n"
        "STOP_POINT: complete\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_monolithic_prompt(case: LongDecisionCase, factors: Sequence[DecisionFactor]) -> str:
    factor_lines = ", ".join(factor.label for factor in factors)
    return (
        "<|im_start|>system\n"
        "Make one integrated decision. Consider all listed factors in one uninterrupted reasoning attempt.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Case title: {case.title}\nDecision: {case.decision}\nContext:\n{case.context}\n\n"
        f"Factors to consider: {factor_lines}\n\n"
        "Output:\nFINAL_DECISION: <decision>\nRATIONALE: <concise integrated rationale>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_forked_aggregation_prompt(
    case: LongDecisionCase,
    factors: Sequence[DecisionFactor],
    branch_outputs: Mapping[str, str],
) -> str:
    return render_decision_aggregation_prompt(
        case,
        factors,
        branch_outputs,
        artifact_source="independent localized factor artifacts",
    )


def render_decision_aggregation_prompt(
    case: LongDecisionCase,
    factors: Sequence[DecisionFactor],
    factor_outputs: Mapping[str, str],
    *,
    artifact_source: str,
) -> str:
    factor_lines = "\n\n".join(
        (
            f"FACTOR_ARTIFACT: {factor.label} - {factor.title}\n"
            f"{factor_outputs.get(factor.label, '').strip()}"
        )
        for factor in factors
    )
    return (
        "<|im_start|>system\n"
        "Collapse completed factor artifacts into one final decision. "
        "Use the factor artifacts as inputs; do not re-run each factor from scratch. "
        "Resolve tensions explicitly and cite which factor artifacts drove the decision.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Case title: {case.title}\n"
        f"Decision: {case.decision}\n"
        f"Context:\n{case.context}\n\n"
        f"Artifact source: {artifact_source}\n"
        "Factor artifacts:\n"
        f"{factor_lines}\n\n"
        "Output exactly these lines:\n"
        "FINAL_DECISION: <one integrated decision>\n"
        "SELECTED_PATH: <chosen option or staged path>\n"
        "KEY_TENSIONS: <main tradeoffs resolved>\n"
        "RATIONALE: <concise rationale grounded in the branch artifacts>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_final_decision_judge_prompt(
    case: LongDecisionCase,
    factors: Sequence[DecisionFactor],
    method: str,
    answer: str,
) -> str:
    factor_lines = "\n".join(f"- {factor.label}: {factor.title}. Objective: {factor.objective}" for factor in factors)
    return (
        "<|im_start|>system\n"
        "You are a strict decision-quality judge. Score only the supplied candidate decision against the case. "
        "Do not reward style. Penalize unsupported claims, missed critical constraints, weak synthesis, and vague actions. "
        "Use integer scores from 1 to 5.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Case title: {case.title}\n"
        f"Decision: {case.decision}\n"
        f"Context:\n{case.context}\n\n"
        f"Selected factors:\n{factor_lines}\n\n"
        f"Candidate method: {method}\n"
        f"Candidate decision:\n{answer}\n\n"
        "Score dimensions:\n"
        "- GROUNDEDNESS: uses case facts accurately\n"
        "- FACTOR_COVERAGE: addresses the selected factors\n"
        "- SYNTHESIS_QUALITY: resolves tradeoffs rather than listing them\n"
        "- RISK_HANDLING: handles downside risks and constraints\n"
        "- ACTIONABILITY: gives an executable decision/path\n\n"
        "Output exactly these lines:\n"
        "GROUNDEDNESS: <1-5>\n"
        "FACTOR_COVERAGE: <1-5>\n"
        "SYNTHESIS_QUALITY: <1-5>\n"
        "RISK_HANDLING: <1-5>\n"
        "ACTIONABILITY: <1-5>\n"
        "OVERALL: <1-5>\n"
        "RATIONALE: <one concise judgement>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_branch_plan(raw_text: str, case: LongDecisionCase) -> BranchPlan:
    decision = _field_value(raw_text, "BRANCH_DECISION").lower()
    should_branch = not decision.startswith("no")
    decision_point = _field_value(raw_text, "DECISION_POINT") or "Planner did not emit a parseable decision point."
    labels = _labels_from_text(_field_value(raw_text, "FACTORS"), tuple(factor.label for factor in case.factors))
    used_fallback = False
    if not labels:
        labels = case.recommended_factors[:5]
        used_fallback = True
    if len(labels) < 3 and should_branch:
        for label in case.recommended_factors:
            if label not in labels:
                labels = labels + (label,)
            if len(labels) >= 3:
                break
        used_fallback = True
    return BranchPlan(should_branch, decision_point.strip(), labels[:6], raw_text, used_fallback)


def parse_factor_blocks(raw_text: str, labels: Sequence[str]) -> dict[str, str]:
    blocks: dict[str, str] = {}
    pattern = re.compile(r"(?im)^FACTOR:\s*([a-z0-9_ -]+)\s*$")
    matches = list(pattern.finditer(raw_text))
    for index, match in enumerate(matches):
        label_text = match.group(1).strip().lower().replace(" ", "_")
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw_text)
        for label in labels:
            if label_text == label.lower():
                blocks[label] = raw_text[start:end].strip()
                break
    for label in labels:
        blocks.setdefault(label, raw_text.strip() if len(labels) == 1 else "")
    return blocks


def parse_final_decision_score(raw_text: str) -> FinalDecisionScore:
    return FinalDecisionScore(
        groundedness=_parse_score(_field_value(raw_text, "GROUNDEDNESS")),
        factor_coverage=_parse_score(_field_value(raw_text, "FACTOR_COVERAGE")),
        synthesis_quality=_parse_score(_field_value(raw_text, "SYNTHESIS_QUALITY")),
        risk_handling=_parse_score(_field_value(raw_text, "RISK_HANDLING")),
        actionability=_parse_score(_field_value(raw_text, "ACTIONABILITY")),
        overall=_parse_score(_field_value(raw_text, "OVERALL")),
        rationale=_field_value(raw_text, "RATIONALE") or raw_text.strip(),
        raw_text=raw_text,
    )


def score_factor_output(case: LongDecisionCase, factor_label: str, output: str) -> FactorScore:
    factor = _factor_by_label(case, factor_label)
    scored_text = _remove_tradeoff_boundary(output)
    lowered = scored_text.lower()
    own_signal_count = _keyword_hits(lowered, factor.keywords + (factor.label, factor.title))
    contamination_count = 0
    for other in case.factors:
        if other.label == factor.label:
            continue
        contamination_count += _keyword_hits(lowered, other.keywords + (other.label, other.title))
    specificity_count = len(re.findall(r"\b(?:\d+(?:\.\d+)?%?|\$[0-9][0-9,]*(?:\.\d+)?)\b", output))
    if own_signal_count == 0:
        focus_score = 1
    elif contamination_count == 0:
        focus_score = 5
    elif contamination_count <= 2:
        focus_score = 4
    elif contamination_count <= own_signal_count:
        focus_score = 3
    else:
        focus_score = 2
    rationale = (
        f"own_signal={own_signal_count}; contamination={contamination_count}; "
        f"specificity={specificity_count}"
    )
    return FactorScore(focus_score, own_signal_count, contamination_count, specificity_count, rationale)


def _remove_tradeoff_boundary(output: str) -> str:
    return re.sub(
        r"(?ims)^TRADEOFF_BOUNDARY:\s*.*?(?=^(?:STOP_POINT|FACTOR|LOCAL_RECOMMENDATION|EVIDENCE):|\Z)",
        "",
        output,
    )


def build_long_decision_eval(
    *,
    engine: LongDecisionEngine,
    cases: Sequence[LongDecisionCase],
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    sequential_rows: list[dict[str, object]] = []
    monolithic_rows: list[dict[str, object]] = []
    aggregate_rows: list[dict[str, object]] = []
    judge_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []

    for index, case in enumerate(cases, start=1):
        if progress_callback is not None:
            progress_callback(index, len(cases), case)
        plan = engine.plan_branches(case)
        factors = plan.factors if plan.should_branch else ()
        forked = engine.forked_factor_answers(case, plan) if factors else {}
        sequential = engine.sequential_factor_answers(case, factors) if factors else {}
        forked_aggregate = engine.aggregate_forked_decision(case, factors, forked) if factors else ""
        sequential_aggregate = engine.aggregate_sequential_decision(case, factors, sequential) if factors else ""
        monolithic = engine.monolithic_answer(case, factors) if factors else ""

        case_rows.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "domain": case.domain,
                "decision": case.decision,
                "should_branch": plan.should_branch,
                "decision_point": plan.decision_point,
                "selected_factors": ",".join(factors),
                "selected_factor_count": len(factors),
                "planner_used_fallback": plan.used_fallback,
                "planner_raw": plan.raw_text,
            }
        )
        monolithic_rows.append(
            {
                "case_id": case.case_id,
                "selected_factors": ",".join(factors),
                "answer": monolithic,
                "factor_coverage": _factor_coverage(monolithic, _resolve_factors(case, factors)),
            }
        )
        aggregate_rows.append(
            {
                "case_id": case.case_id,
                "method": "forked_aggregate",
                "selected_factors": ",".join(factors),
                "answer": forked_aggregate,
                "factor_coverage": _factor_coverage(forked_aggregate, _resolve_factors(case, factors)),
            }
        )
        aggregate_rows.append(
            {
                "case_id": case.case_id,
                "method": "sequential_aggregate",
                "selected_factors": ",".join(factors),
                "answer": sequential_aggregate,
                "factor_coverage": _factor_coverage(sequential_aggregate, _resolve_factors(case, factors)),
            }
        )
        for method, answer in (
            ("forked_aggregate", forked_aggregate),
            ("sequential_aggregate", sequential_aggregate),
            ("monolithic", monolithic),
        ):
            judge_rows.append(_judge_row(case, method, engine.judge_final_decision(case, factors, method, answer)))
        for label in factors:
            branch_output = forked.get(label, "")
            sequential_output = sequential.get(label, "")
            branch_score = score_factor_output(case, label, branch_output)
            sequential_score = score_factor_output(case, label, sequential_output)
            branch_rows.append(_factor_row(case, label, "forked_branch", branch_output, branch_score))
            sequential_rows.append(_factor_row(case, label, "sequential", sequential_output, sequential_score))
            score_rows.extend(
                [
                    _score_row(case, label, "forked_branch", branch_score),
                    _score_row(case, label, "sequential", sequential_score),
                ]
            )

    summary_df = build_long_decision_summary(
        pd.DataFrame(score_rows),
        pd.DataFrame(case_rows),
        pd.DataFrame(monolithic_rows),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(judge_rows),
    )
    return (
        pd.DataFrame(case_rows),
        pd.DataFrame(branch_rows),
        pd.DataFrame(sequential_rows),
        pd.DataFrame(monolithic_rows),
        pd.DataFrame(aggregate_rows),
        pd.DataFrame(judge_rows),
        summary_df,
    )


def build_long_decision_summary(
    scores_df: pd.DataFrame,
    cases_df: pd.DataFrame,
    monolithic_df: pd.DataFrame,
    aggregate_df: pd.DataFrame | None = None,
    judge_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    aggregate_df = aggregate_df if aggregate_df is not None else pd.DataFrame()
    judge_df = judge_df if judge_df is not None else pd.DataFrame()
    forked_coverage = (
        float(aggregate_df.loc[aggregate_df["method"] == "forked_aggregate", "factor_coverage"].mean())
        if len(aggregate_df) and "method" in aggregate_df
        else 0
    )
    rows: list[dict[str, object]] = [
        {"metric": "case_count", "value": len(cases_df)},
        {"metric": "avg_selected_factor_count", "value": float(cases_df["selected_factor_count"].mean()) if len(cases_df) else 0},
        {"metric": "planner_fallback_count", "value": int(cases_df["planner_used_fallback"].sum()) if len(cases_df) else 0},
        {"metric": "avg_monolithic_factor_coverage", "value": float(monolithic_df["factor_coverage"].mean()) if len(monolithic_df) else 0},
        {"metric": "avg_forked_aggregate_factor_coverage", "value": forked_coverage},
    ]
    if not scores_df.empty:
        for method, group in scores_df.groupby("method"):
            rows.extend(
                [
                    {"metric": f"{method}_avg_focus_score", "value": float(group["focus_score"].mean())},
                    {"metric": f"{method}_avg_contamination_count", "value": float(group["contamination_count"].mean())},
                    {"metric": f"{method}_avg_own_signal_count", "value": float(group["own_signal_count"].mean())},
                    {"metric": f"{method}_avg_specificity_count", "value": float(group["specificity_count"].mean())},
                ]
            )
    if not judge_df.empty:
        for method, group in judge_df.groupby("method"):
            rows.extend(
                [
                    {"metric": f"{method}_judge_overall", "value": float(group["overall"].mean())},
                    {"metric": f"{method}_judge_groundedness", "value": float(group["groundedness"].mean())},
                    {"metric": f"{method}_judge_factor_coverage", "value": float(group["factor_coverage"].mean())},
                    {"metric": f"{method}_judge_synthesis_quality", "value": float(group["synthesis_quality"].mean())},
                    {"metric": f"{method}_judge_risk_handling", "value": float(group["risk_handling"].mean())},
                    {"metric": f"{method}_judge_actionability", "value": float(group["actionability"].mean())},
                ]
            )
    return pd.DataFrame(rows)


def export_long_decision_eval(
    *,
    output_xlsx: Path,
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    sequential_df: pd.DataFrame,
    monolithic_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Path:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        case_df.to_excel(writer, sheet_name="Cases", index=False)
        branch_df.to_excel(writer, sheet_name="Forked Branches", index=False)
        aggregate_df.to_excel(writer, sheet_name="Forked Aggregate", index=False)
        sequential_df.to_excel(writer, sheet_name="Sequential", index=False)
        monolithic_df.to_excel(writer, sheet_name="Monolithic", index=False)
        judge_df.to_excel(writer, sheet_name="LLM Judge", index=False)
    return output_xlsx


def write_long_decision_csvs(
    *,
    output_dir: Path,
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    sequential_df: pd.DataFrame,
    monolithic_df: pd.DataFrame,
    aggregate_df: pd.DataFrame,
    judge_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_qa_csv(summary_df, output_dir / "long_decision_summary.csv")
    export_qa_csv(case_df, output_dir / "long_decision_cases.csv")
    export_qa_csv(branch_df, output_dir / "long_decision_forked_branches.csv")
    export_qa_csv(aggregate_df, output_dir / "long_decision_forked_aggregate.csv")
    export_qa_csv(sequential_df, output_dir / "long_decision_sequential.csv")
    export_qa_csv(monolithic_df, output_dir / "long_decision_monolithic.csv")
    export_qa_csv(judge_df, output_dir / "long_decision_judge.csv")


def _factor_row(case: LongDecisionCase, label: str, method: str, output: str, score: FactorScore) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "method": method,
        "factor": label,
        "output": output,
        "focus_score": score.focus_score,
        "own_signal_count": score.own_signal_count,
        "contamination_count": score.contamination_count,
        "specificity_count": score.specificity_count,
        "score_rationale": score.rationale,
    }


def _score_row(case: LongDecisionCase, label: str, method: str, score: FactorScore) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "method": method,
        "factor": label,
        "focus_score": score.focus_score,
        "own_signal_count": score.own_signal_count,
        "contamination_count": score.contamination_count,
        "specificity_count": score.specificity_count,
        "score_rationale": score.rationale,
    }


def _judge_row(case: LongDecisionCase, method: str, score: FinalDecisionScore) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "method": method,
        "groundedness": score.groundedness,
        "factor_coverage": score.factor_coverage,
        "synthesis_quality": score.synthesis_quality,
        "risk_handling": score.risk_handling,
        "actionability": score.actionability,
        "overall": score.overall,
        "rationale": score.rationale,
        "raw_text": score.raw_text,
    }


def _factor_coverage(text: str, factors: Sequence[DecisionFactor]) -> float:
    if not factors:
        return 0
    lowered = text.lower()
    covered = sum(1 for factor in factors if _keyword_hits(lowered, (factor.label, factor.title) + factor.keywords) > 0)
    return covered / len(factors)


def _resolve_factors(case: LongDecisionCase, labels: Sequence[str]) -> tuple[DecisionFactor, ...]:
    by_label = {factor.label: factor for factor in case.factors}
    return tuple(by_label[label] for label in labels if label in by_label)


def _factor_by_label(case: LongDecisionCase, label: str) -> DecisionFactor:
    for factor in case.factors:
        if factor.label == label:
            return factor
    raise KeyError(label)


def _keyword_hits(lowered_text: str, keywords: Sequence[str]) -> int:
    return sum(1 for keyword in keywords if keyword and keyword.lower() in lowered_text)


def _field_value(raw_text: str, field: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(field)}\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(raw_text)
    return match.group(1).strip() if match else ""


def _parse_score(value: str) -> int:
    match = re.search(r"[1-5]", value or "")
    return int(match.group(0)) if match else 0


def _labels_from_text(text: str, allowed_labels: tuple[str, ...]) -> tuple[str, ...]:
    lowered = text.lower()
    found: list[tuple[int, str]] = []
    for label in allowed_labels:
        match = re.search(rf"\b{re.escape(label.lower())}\b", lowered)
        if match:
            found.append((match.start(), label))
    seen: set[str] = set()
    labels: list[str] = []
    for _, label in sorted(found):
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return tuple(labels)


def _strip_empty_think_blocks(text: str) -> str:
    return re.sub(r"<think>\s*</think>", "", text, flags=re.I | re.S).strip()

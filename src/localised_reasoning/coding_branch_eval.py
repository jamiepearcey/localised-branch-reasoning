from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Mapping, Protocol, Sequence

import pandas as pd

from localised_reasoning.live_llama_worker import LiveLlamaWorker, WorkerBranch
from localised_reasoning.qa_scenarios import export_qa_csv


CODING_BRANCH_PREFIX_CACHE_ID = "coding_branch_static_prefix_v1"


@dataclass(frozen=True)
class CodingBranchPoint:
    label: str
    title: str
    location_hint: str
    risk_reason: str


@dataclass(frozen=True)
class CodingConsideration:
    label: str
    title: str
    objective: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class CodingCase:
    case_id: str
    language: str
    title: str
    task: str
    code: str
    branch_points: tuple[CodingBranchPoint, ...]
    considerations: tuple[CodingConsideration, ...]
    expected_branch_points: tuple[str, ...]


@dataclass(frozen=True)
class CodingBranchPlan:
    selected_points: tuple[str, ...]
    rationale: str
    raw_text: str
    used_fallback: bool = False


@dataclass(frozen=True)
class CodingPointScore:
    point_count: int
    includes_method_start: bool
    includes_method_end: bool
    expected_hit_rate: float


class CodingBranchEngine(Protocol):
    def plan_branch_points(self, case: CodingCase) -> CodingBranchPlan:
        ...

    def branch_checkpoint(
        self,
        case: CodingCase,
        point: CodingBranchPoint,
        considerations: Sequence[CodingConsideration],
        prior_collapses: Sequence[str],
    ) -> Mapping[str, str]:
        ...

    def collapse_checkpoint(
        self,
        case: CodingCase,
        point: CodingBranchPoint,
        branch_outputs: Mapping[str, str],
    ) -> str:
        ...

    def final_from_checkpoints(
        self,
        case: CodingCase,
        checkpoint_collapses: Mapping[str, str],
    ) -> str:
        ...

    def monolithic_review(self, case: CodingCase) -> str:
        ...


def default_coding_cases() -> list[CodingCase]:
    considerations = (
        CodingConsideration("contract", "API contract and invariants", "Check inputs, outputs, invariants, and caller-visible behavior.", ("input", "return", "invariant", "contract", "caller")),
        CodingConsideration("edge_cases", "Edge cases", "Find boundary values, missing states, and malformed data paths.", ("edge", "none", "empty", "missing", "boundary")),
        CodingConsideration("state_consistency", "State consistency", "Check transactions, idempotency, rollback, and partial writes.", ("transaction", "idempot", "rollback", "partial", "state")),
        CodingConsideration("security", "Security and authorization", "Check authorization, tenant isolation, sensitive data, and abuse paths.", ("auth", "tenant", "permission", "secret", "security")),
        CodingConsideration("performance", "Performance and scaling", "Check loops, external calls, batching, and complexity.", ("loop", "batch", "latency", "n+1", "performance")),
        CodingConsideration("tests", "Test coverage", "Name specific tests that would catch the highest-risk defects.", ("test", "assert", "fixture", "mock", "coverage")),
    )
    return [
        CodingCase(
            case_id="python_refund_method",
            language="python",
            title="Refund Application Method",
            task="Review and propose the safest patch plan for apply_refund without writing the full patch.",
            code="""\
def apply_refund(order_id: str, amount: Decimal, reason: str, actor: User) -> RefundResult:
    order = orders.get(order_id)
    if not order:
        return RefundResult(False, "missing order")
    if not actor.is_support:
        return RefundResult(False, "forbidden")

    existing = refunds.find_by_order(order_id)
    if existing:
        return RefundResult(True, existing.gateway_id)

    if amount > order.total:
        amount = order.total

    order.status = "refund_pending"
    orders.save(order)

    gateway_id = payment_gateway.refund(order.payment_id, float(amount), reason)
    refunds.insert({
        "order_id": order_id,
        "amount": amount,
        "gateway_id": gateway_id,
        "created_by": actor.id,
    })

    order.status = "refunded"
    orders.save(order)
    audit.log("refund", order_id=order_id, actor_id=actor.id, amount=str(amount))
    return RefundResult(True, gateway_id)
""",
            branch_points=(
                CodingBranchPoint("method_start", "Method start and preconditions", "first lines before loading or mutating order", "Input validation, authorization, and idempotency assumptions are established here."),
                CodingBranchPoint("pre_mutation", "Before state mutation", "before order.status = refund_pending", "This is where irreversible state changes begin."),
                CodingBranchPoint("external_call", "External payment call boundary", "payment_gateway.refund(...)", "External side effects, retries, Decimal conversion, and duplicate refunds are high risk."),
                CodingBranchPoint("method_end", "Method end and postconditions", "final status save, audit log, and return", "Postconditions, audit completeness, and returned gateway id are finalized here."),
            ),
            considerations=considerations,
            expected_branch_points=("method_start", "pre_mutation", "external_call", "method_end"),
        ),
        CodingCase(
            case_id="typescript_subscription_update",
            language="typescript",
            title="Subscription Update Method",
            task="Review and propose the safest patch plan for updateSubscription without writing the full patch.",
            code="""\
async function updateSubscription(req: Request, res: Response) {
  const account = await accounts.find(req.params.accountId);
  const plan = await plans.find(req.body.planId);
  if (!account || !plan) {
    return res.status(404).json({ error: "not found" });
  }

  const previousPlan = account.planId;
  account.planId = plan.id;
  account.renewalDate = addMonths(new Date(), 1);
  await accounts.save(account);

  const invoice = await billing.createInvoice({
    accountId: account.id,
    planId: plan.id,
    prorateFrom: new Date(),
    coupon: req.body.coupon,
  });

  await cache.del(`account:${account.id}`);
  await webhooks.emit("subscription.updated", {
    accountId: account.id,
    previousPlan,
    nextPlan: plan.id,
    invoiceId: invoice.id,
  });

  return res.json({ ok: true, invoiceId: invoice.id });
}
""",
            branch_points=(
                CodingBranchPoint("method_start", "Method start and request boundary", "request parameter/body parsing and account lookup", "Authorization, tenant isolation, and malformed body risks appear before state changes."),
                CodingBranchPoint("pre_mutation", "Before account mutation", "before account.planId = plan.id", "Plan transition rules and renewal date semantics are decided before persistence."),
                CodingBranchPoint("external_call", "Billing side-effect boundary", "billing.createInvoice(...)", "Billing side effects, coupons, proration, and retries are high risk."),
                CodingBranchPoint("method_end", "Method end and notification boundary", "cache deletion, webhook emit, and response", "Cache consistency, webhook ordering, and response correctness are finalized here."),
            ),
            considerations=considerations,
            expected_branch_points=("method_start", "pre_mutation", "external_call", "method_end"),
        ),
    ]


class LiveCodingBranchEngine:
    def __init__(
        self,
        *,
        worker: LiveLlamaWorker,
        branch_max_new_tokens: int = 150,
        collapse_max_new_tokens: int = 220,
        final_max_new_tokens: int = 500,
        planner_max_new_tokens: int = 120,
        request_timeout_s: float | None = None,
    ) -> None:
        self.worker = worker
        self.branch_max_new_tokens = branch_max_new_tokens
        self.collapse_max_new_tokens = collapse_max_new_tokens
        self.final_max_new_tokens = final_max_new_tokens
        self.planner_max_new_tokens = planner_max_new_tokens
        self.request_timeout_s = request_timeout_s
        self.prefix_cache_enabled = self._install_prefix_cache()

    def _install_prefix_cache(self) -> bool:
        cache_prefix = getattr(self.worker, "cache_prefix", None)
        if cache_prefix is None:
            return False
        try:
            cache_prefix(
                prefix_id=CODING_BRANCH_PREFIX_CACHE_ID,
                prefix=render_coding_static_prefix(),
                timeout_s=self.request_timeout_s,
            )
        except RuntimeError:
            return False
        return True

    def plan_branch_points(self, case: CodingCase) -> CodingBranchPlan:
        response = self.worker.generate(
            render_coding_planner_prompt(case),
            max_new_tokens=self.planner_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        raw = _strip_empty_think_blocks(str(response.get("text", ""))).strip()
        return parse_coding_branch_plan(raw, case)

    def branch_checkpoint(
        self,
        case: CodingCase,
        point: CodingBranchPoint,
        considerations: Sequence[CodingConsideration],
        prior_collapses: Sequence[str],
    ) -> Mapping[str, str]:
        suffix = render_coding_case_suffix(case, point, prior_collapses)
        branches = [
            WorkerBranch(consideration.label, render_coding_consideration_marker(point, consideration))
            for consideration in considerations
        ]
        if self.prefix_cache_enabled:
            response = self.worker.cached_branch(
                prefix_id=CODING_BRANCH_PREFIX_CACHE_ID,
                suffix=suffix,
                branches=branches,
                max_new_tokens=self.branch_max_new_tokens,
                stop="STOP_POINT:",
                parallel=True,
                timeout_s=self.request_timeout_s,
            )
        else:
            response = self.worker.branch(
                prefix=render_coding_static_prefix() + suffix,
                branches=branches,
                max_new_tokens=self.branch_max_new_tokens,
                stop="STOP_POINT:",
                parallel=True,
                timeout_s=self.request_timeout_s,
            )
        answers = {str(item.get("label", "")): _strip_empty_think_blocks(str(item.get("text", ""))).strip() for item in response.get("branches", [])}
        return {consideration.label: answers.get(consideration.label, "") for consideration in considerations}

    def collapse_checkpoint(
        self,
        case: CodingCase,
        point: CodingBranchPoint,
        branch_outputs: Mapping[str, str],
    ) -> str:
        response = self.worker.generate(
            render_checkpoint_collapse_prompt(case, point, branch_outputs),
            max_new_tokens=self.collapse_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        return _strip_empty_think_blocks(str(response.get("text", ""))).strip()

    def final_from_checkpoints(
        self,
        case: CodingCase,
        checkpoint_collapses: Mapping[str, str],
    ) -> str:
        response = self.worker.generate(
            render_coding_final_prompt(case, checkpoint_collapses),
            max_new_tokens=self.final_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        return _strip_empty_think_blocks(str(response.get("text", ""))).strip()

    def monolithic_review(self, case: CodingCase) -> str:
        response = self.worker.generate(
            render_coding_monolithic_prompt(case),
            max_new_tokens=self.final_max_new_tokens,
            stop="<|im_end|>",
            timeout_s=self.request_timeout_s,
        )
        return _strip_empty_think_blocks(str(response.get("text", ""))).strip()


class ProxyCodingBranchEngine:
    def plan_branch_points(self, case: CodingCase) -> CodingBranchPlan:
        return CodingBranchPlan(
            selected_points=case.expected_branch_points,
            rationale="Proxy selects method start, high-risk side-effect boundaries, and method end.",
            raw_text="BRANCH_POINTS: " + ", ".join(case.expected_branch_points),
        )

    def branch_checkpoint(
        self,
        case: CodingCase,
        point: CodingBranchPoint,
        considerations: Sequence[CodingConsideration],
        prior_collapses: Sequence[str],
    ) -> Mapping[str, str]:
        return {
            consideration.label: (
                f"CONSIDERATION: {consideration.label}\n"
                f"FINDING: At {point.label}, check {consideration.title.lower()} locally.\n"
                f"EVIDENCE: {point.risk_reason}\n"
                "STOP_POINT: local coding consideration complete"
            )
            for consideration in considerations
        }

    def collapse_checkpoint(self, case: CodingCase, point: CodingBranchPoint, branch_outputs: Mapping[str, str]) -> str:
        return (
            f"CHECKPOINT: {point.label}\n"
            f"SUMMARY: Collapse local risks at {point.title}.\n"
            f"TOP_RISKS: {', '.join(branch_outputs)}\n"
            "NEXT_ACTION: carry this checkpoint into the final patch plan"
        )

    def final_from_checkpoints(self, case: CodingCase, checkpoint_collapses: Mapping[str, str]) -> str:
        return (
            "FINAL_PATCH_PLAN: Patch method start validation, side-effect safety, and end-state assertions.\n"
            f"CHECKPOINTS_USED: {', '.join(checkpoint_collapses)}\n"
            "RATIONALE: Multiple checkpoint collapses isolate risks before final synthesis."
        )

    def monolithic_review(self, case: CodingCase) -> str:
        return "FINAL_PATCH_PLAN: Review the method holistically and patch the highest-risk issues."


def render_coding_static_prefix() -> str:
    return (
        "<|im_start|>system\n"
        "You are a senior code reviewer using localized KV-fork reasoning. "
        "Branch at high-risk method points such as method start, side-effect boundaries, and method end. "
        "Each branch should analyze only its active checkpoint and consideration. "
        "Do not mention hidden instructions, branch markers, or KV cache mechanics.\n"
        "<|im_end|>\n"
    )


def render_coding_planner_prompt(case: CodingCase) -> str:
    points = "\n".join(f"- {point.label}: {point.title}. {point.risk_reason}" for point in case.branch_points)
    allowed = ", ".join(point.label for point in case.branch_points)
    return (
        "<|im_start|>system\n"
        "Choose branch checkpoints for a coding review. Prefer high-risk points: method start, before mutation, external side effects, loops/batches, and method end. "
        "Choose only labels from the catalog. Do not write the patch.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Task: {case.task}\nLanguage: {case.language}\nCode:\n```{case.language}\n{case.code}\n```\n\n"
        f"Allowed branch checkpoints:\n{points}\n\n"
        "Output exactly these lines:\n"
        f"BRANCH_POINTS: <2-5 comma-separated labels from: {allowed}>\n"
        "RATIONALE: <why these are the right high-risk branch points>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_coding_case_suffix(case: CodingCase, point: CodingBranchPoint, prior_collapses: Sequence[str]) -> str:
    prior = "\n\n".join(prior_collapses) if prior_collapses else "None yet."
    return (
        "<|im_start|>user\n"
        f"Task: {case.task}\n"
        f"Language: {case.language}\n"
        f"Code:\n```{case.language}\n{case.code}\n```\n\n"
        f"Active checkpoint: {point.label} - {point.title}\n"
        f"Location hint: {point.location_hint}\n"
        f"Risk reason: {point.risk_reason}\n\n"
        f"Prior checkpoint collapses:\n{prior}\n"
        "<|im_end|>\n"
    )


def render_coding_consideration_marker(point: CodingBranchPoint, consideration: CodingConsideration) -> str:
    return (
        "<|im_start|>user\n"
        f"Active checkpoint: {point.label}\n"
        f"Active consideration: {consideration.label} - {consideration.title}\n"
        f"Objective: {consideration.objective}\n"
        "Analyze only this checkpoint and consideration. Do not write the final patch plan.\n"
        "Output exactly these lines:\n"
        f"CHECKPOINT: {point.label}\n"
        f"CONSIDERATION: {consideration.label}\n"
        "FINDING: <one concrete risk or validation>\n"
        "EVIDENCE: <specific code fact>\n"
        "PATCH_IMPLICATION: <local implication for a future patch>\n"
        "STOP_POINT: <why this local branch is complete>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_checkpoint_collapse_prompt(case: CodingCase, point: CodingBranchPoint, branch_outputs: Mapping[str, str]) -> str:
    artifacts = "\n\n".join(f"BRANCH_ARTIFACT: {label}\n{text}" for label, text in branch_outputs.items())
    return (
        "<|im_start|>system\n"
        "Collapse local coding branch artifacts for one checkpoint. Use only the artifacts and code context. "
        "Do not solve unrelated checkpoints.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Task: {case.task}\nCode:\n```{case.language}\n{case.code}\n```\n\n"
        f"Checkpoint: {point.label} - {point.title}\n"
        f"Location hint: {point.location_hint}\n\n"
        f"{artifacts}\n\n"
        "Output exactly these lines:\n"
        f"CHECKPOINT: {point.label}\n"
        "SUMMARY: <checkpoint-local synthesis>\n"
        "TOP_RISKS: <highest-risk issues at this checkpoint>\n"
        "PATCH_IMPLICATIONS: <local patch implications>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_coding_final_prompt(case: CodingCase, checkpoint_collapses: Mapping[str, str]) -> str:
    collapses = "\n\n".join(f"CHECKPOINT_COLLAPSE: {label}\n{text}" for label, text in checkpoint_collapses.items())
    return (
        "<|im_start|>system\n"
        "Create one final coding recommendation from completed checkpoint collapses. "
        "Cite checkpoint collapses, resolve tensions, and give an actionable patch/test plan.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Task: {case.task}\nCode:\n```{case.language}\n{case.code}\n```\n\n"
        f"{collapses}\n\n"
        "Output exactly these lines:\n"
        "FINAL_PATCH_PLAN: <ordered patch plan>\n"
        "CRITICAL_RISKS: <risks addressed>\n"
        "TEST_PLAN: <specific tests>\n"
        "RATIONALE: <why this plan follows from the checkpoint collapses>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def render_coding_monolithic_prompt(case: CodingCase) -> str:
    return (
        "<|im_start|>system\n"
        "Review this method in one uninterrupted reasoning stream. Produce one final patch/test plan.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"Task: {case.task}\nLanguage: {case.language}\nCode:\n```{case.language}\n{case.code}\n```\n\n"
        "Output exactly these lines:\n"
        "FINAL_PATCH_PLAN: <ordered patch plan>\n"
        "CRITICAL_RISKS: <risks addressed>\n"
        "TEST_PLAN: <specific tests>\n"
        "RATIONALE: <concise rationale>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def parse_coding_branch_plan(raw_text: str, case: CodingCase) -> CodingBranchPlan:
    allowed = tuple(point.label for point in case.branch_points)
    labels = _labels_from_text(_field_value(raw_text, "BRANCH_POINTS"), allowed)
    used_fallback = False
    if not labels:
        labels = case.expected_branch_points
        used_fallback = True
    return CodingBranchPlan(
        selected_points=labels[:5],
        rationale=_field_value(raw_text, "RATIONALE") or raw_text.strip(),
        raw_text=raw_text,
        used_fallback=used_fallback,
    )


def score_coding_branch_points(case: CodingCase, plan: CodingBranchPlan) -> CodingPointScore:
    selected = set(plan.selected_points)
    expected = set(case.expected_branch_points)
    return CodingPointScore(
        point_count=len(plan.selected_points),
        includes_method_start="method_start" in selected,
        includes_method_end="method_end" in selected,
        expected_hit_rate=len(selected & expected) / len(expected) if expected else 0,
    )


def build_coding_branch_eval(
    *,
    engine: CodingBranchEngine,
    cases: Sequence[CodingCase],
    consideration_limit: int = 5,
    progress_callback=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    case_rows: list[dict[str, object]] = []
    branch_rows: list[dict[str, object]] = []
    collapse_rows: list[dict[str, object]] = []
    final_rows: list[dict[str, object]] = []
    monolithic_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for index, case in enumerate(cases, start=1):
        if progress_callback is not None:
            progress_callback(index, len(cases), case)
        plan = engine.plan_branch_points(case)
        point_score = score_coding_branch_points(case, plan)
        points = _resolve_points(case, plan.selected_points)
        considerations = case.considerations[:consideration_limit]
        prior_collapses: list[str] = []
        checkpoint_collapses: dict[str, str] = {}
        for point in points:
            branch_outputs = engine.branch_checkpoint(case, point, considerations, prior_collapses)
            for consideration in considerations:
                output = branch_outputs.get(consideration.label, "")
                branch_rows.append(
                    {
                        "case_id": case.case_id,
                        "checkpoint": point.label,
                        "consideration": consideration.label,
                        "output": output,
                        "locality_score": _locality_score(output, point, consideration),
                    }
                )
            collapse = engine.collapse_checkpoint(case, point, branch_outputs)
            checkpoint_collapses[point.label] = collapse
            prior_collapses.append(collapse)
            collapse_rows.append({"case_id": case.case_id, "checkpoint": point.label, "collapse": collapse})
        final = engine.final_from_checkpoints(case, checkpoint_collapses)
        monolithic = engine.monolithic_review(case)
        final_rows.append({"case_id": case.case_id, "method": "multi_checkpoint_branch", "answer": final})
        monolithic_rows.append({"case_id": case.case_id, "method": "monolithic", "answer": monolithic})
        case_rows.append(
            {
                "case_id": case.case_id,
                "title": case.title,
                "language": case.language,
                "selected_checkpoints": ",".join(plan.selected_points),
                "checkpoint_count": len(plan.selected_points),
                "includes_method_start": point_score.includes_method_start,
                "includes_method_end": point_score.includes_method_end,
                "expected_hit_rate": point_score.expected_hit_rate,
                "planner_used_fallback": plan.used_fallback,
                "planner_rationale": plan.rationale,
                "planner_raw": plan.raw_text,
            }
        )
    branch_df = pd.DataFrame(branch_rows)
    case_df = pd.DataFrame(case_rows)
    summary_rows.extend(
        [
            {"metric": "case_count", "value": len(case_df)},
            {"metric": "avg_checkpoint_count", "value": float(case_df["checkpoint_count"].mean()) if len(case_df) else 0},
            {"metric": "method_start_rate", "value": float(case_df["includes_method_start"].mean()) if len(case_df) else 0},
            {"metric": "method_end_rate", "value": float(case_df["includes_method_end"].mean()) if len(case_df) else 0},
            {"metric": "avg_expected_hit_rate", "value": float(case_df["expected_hit_rate"].mean()) if len(case_df) else 0},
            {"metric": "avg_branch_locality_score", "value": float(branch_df["locality_score"].mean()) if len(branch_df) else 0},
            {"metric": "collapse_count", "value": len(collapse_rows)},
        ]
    )
    return (
        case_df,
        branch_df,
        pd.DataFrame(collapse_rows),
        pd.DataFrame(final_rows),
        pd.DataFrame(monolithic_rows),
        pd.DataFrame(summary_rows),
    )


def export_coding_branch_eval(
    *,
    output_xlsx: Path,
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    final_df: pd.DataFrame,
    monolithic_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> Path:
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_xlsx) as writer:
        summary_df.to_excel(writer, sheet_name="Summary", index=False)
        case_df.to_excel(writer, sheet_name="Cases", index=False)
        branch_df.to_excel(writer, sheet_name="Checkpoint Branches", index=False)
        collapse_df.to_excel(writer, sheet_name="Checkpoint Collapses", index=False)
        final_df.to_excel(writer, sheet_name="Final Branch Plan", index=False)
        monolithic_df.to_excel(writer, sheet_name="Monolithic", index=False)
    return output_xlsx


def write_coding_branch_csvs(
    *,
    output_dir: Path,
    case_df: pd.DataFrame,
    branch_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    final_df: pd.DataFrame,
    monolithic_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    export_qa_csv(summary_df, output_dir / "coding_branch_summary.csv")
    export_qa_csv(case_df, output_dir / "coding_branch_cases.csv")
    export_qa_csv(branch_df, output_dir / "coding_branch_checkpoint_branches.csv")
    export_qa_csv(collapse_df, output_dir / "coding_branch_checkpoint_collapses.csv")
    export_qa_csv(final_df, output_dir / "coding_branch_final.csv")
    export_qa_csv(monolithic_df, output_dir / "coding_branch_monolithic.csv")


def _resolve_points(case: CodingCase, labels: Sequence[str]) -> tuple[CodingBranchPoint, ...]:
    by_label = {point.label: point for point in case.branch_points}
    return tuple(by_label[label] for label in labels if label in by_label)


def _locality_score(output: str, point: CodingBranchPoint, consideration: CodingConsideration) -> int:
    lowered = output.lower()
    point_hit = point.label.lower() in lowered or point.title.lower() in lowered or point.location_hint.lower().split()[0] in lowered
    own_hits = sum(1 for keyword in consideration.keywords if keyword.lower() in lowered)
    if own_hits >= 2 and point_hit:
        return 5
    if own_hits >= 1 and point_hit:
        return 4
    if own_hits >= 1:
        return 3
    return 2 if point_hit else 1


def _field_value(raw_text: str, field: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(field)}\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
    match = pattern.search(raw_text)
    return match.group(1).strip() if match else ""


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

from __future__ import annotations

from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def register_paper_benchmarks(snapshot_root: Path) -> tuple[str, str, str, str]:
    from evalscope.api.benchmark import BenchmarkMeta, DefaultDataAdapter
    from evalscope.api.dataset import Sample
    from evalscope.api.metric.scorer import Score
    from evalscope.api.registry import register_benchmark
    from evalscope.constants import Tags

    math500_path = (snapshot_root / "math500" / "test.jsonl").resolve()
    gsm8k_path = (snapshot_root / "gsm8k" / "test.jsonl").resolve()
    olympiad_path = (snapshot_root / "olympiadbench" / "test.jsonl").resolve()
    amc_path = (snapshot_root / "amc23" / "test.jsonl").resolve()
    for path in (math500_path, gsm8k_path, olympiad_path, amc_path):
        if not path.exists():
            raise FileNotFoundError(f"Pinned paper benchmark file is missing: {path}")

    class PinnedLocalJSONLAdapter(DefaultDataAdapter):
        identity_benchmark: str

        def load_from_disk(self, use_local_loader: bool = False):
            del use_local_loader
            return super().load_from_disk(use_local_loader=True)

        def process_sample_str_input(self, sample: Sample, subset: str):
            messages = super().process_sample_str_input(sample, subset)
            if sample.id is None or sample.group_id is None:
                raise RuntimeError("EvalScope must reindex samples before prompt formatting")
            repeats = max(1, int(self.repeats))
            messages[-1].metadata = {
                **(messages[-1].metadata or {}),
                "eval_identity": {
                    "benchmark": self.identity_benchmark,
                    "item_id": int(sample.group_id),
                    "sample_id": int(sample.id % repeats),
                    "evalscope_sample_id": int(sample.id),
                },
            }
            return messages

    @register_benchmark(
        BenchmarkMeta(
            name="paper_math500",
            pretty_name="MATH-500 (paper-pinned)",
            description="Paper-pinned 500-problem MATH evaluation split.",
            dataset_id=str(math500_path),
            tags=[Tags.MATH, Tags.REASONING],
            subset_list=["main"],
            metric_list=[{"acc": {"numeric": True}}],
            eval_split="train",
            prompt_template=PROMPT_TEMPLATE,
        )
    )
    class PaperMath500Adapter(PinnedLocalJSONLAdapter):
        identity_benchmark = "paper_math500"

        def record_to_sample(self, record: dict[str, Any]) -> Sample:
            return Sample(
                input=record["problem"],
                target=str(record["answer"]),
                metadata={
                    "unique_id": record.get("unique_id"),
                    "subject": record.get("subject"),
                    "level": record.get("level"),
                    "solution": record.get("solution"),
                },
            )

        def extract_answer(self, prediction: str, task_state: Any) -> str:
            from evalscope.metrics.math_parser import extract_answer

            return extract_answer(prediction)

    @register_benchmark(
        BenchmarkMeta(
            name="paper_gsm8k",
            pretty_name="GSM8K (paper-pinned)",
            description="Paper-pinned 1,319-problem GSM8K evaluation split.",
            dataset_id=str(gsm8k_path),
            tags=[Tags.MATH, Tags.REASONING],
            subset_list=["main"],
            metric_list=[{"acc": {"numeric": True}}],
            eval_split="train",
            prompt_template=PROMPT_TEMPLATE,
        )
    )
    class PaperGSM8KAdapter(PinnedLocalJSONLAdapter):
        identity_benchmark = "paper_gsm8k"

        def record_to_sample(self, record: dict[str, Any]) -> Sample:
            reasoning, separator, target = str(record["answer"]).rpartition("####")
            if not separator:
                raise ValueError("Pinned GSM8K record is missing the #### answer delimiter")
            return Sample(
                input=record["question"],
                target=target.strip(),
                metadata={"idx": record.get("idx"), "reasoning": reasoning.strip()},
            )

        def extract_answer(self, prediction: str, task_state: Any) -> str:
            from evalscope.metrics.math_parser import extract_answer

            return extract_answer(prediction)

    @register_benchmark(
        BenchmarkMeta(
            name="paper_olympiadbench",
            pretty_name="OlympiadBench (paper-pinned text EN)",
            description="Paper-pinned English text-only OlympiadBench evaluation split.",
            dataset_id=str(olympiad_path),
            tags=[Tags.MATH, Tags.REASONING],
            subset_list=["main"],
            metric_list=["acc"],
            eval_split="train",
            prompt_template=PROMPT_TEMPLATE,
        )
    )
    class PaperOlympiadBenchAdapter(PinnedLocalJSONLAdapter):
        identity_benchmark = "paper_olympiadbench"

        def record_to_sample(self, record: dict[str, Any]) -> Sample:
            final_answer = record.get("final_answer", [])
            return Sample(
                input=record["question"],
                target=",".join(str(value) for value in final_answer),
                metadata={
                    "id": record.get("id"),
                    "final_answer": final_answer,
                    "answer_type": record.get("answer_type", ""),
                    "error": record.get("error"),
                    "is_multiple_answer": record.get("is_multiple_answer", False),
                    "unit": record.get("unit"),
                },
            )

        def extract_answer(self, prediction: str, task_state: Any) -> str:
            from evalscope.metrics.math_parser import extract_answer

            return extract_answer(prediction)

        def match_score(
            self,
            original_prediction: str,
            filtered_prediction: str,
            reference: str,
            task_state: Any,
        ) -> Score:
            from evalscope.benchmarks.olympiad_bench.utils import MathJudger

            final_answers = task_state.metadata["final_answer"]
            score = Score(
                extracted_prediction=filtered_prediction,
                prediction=original_prediction,
            )
            score.value = {
                "acc": float(
                    bool(final_answers)
                    and MathJudger().judge(filtered_prediction, str(final_answers[0]))
                )
            }
            return score

    @register_benchmark(
        BenchmarkMeta(
            name="paper_amc23",
            pretty_name="AMC 2023 (paper-pinned)",
            description="Paper-pinned AMC 2023 math evaluation split.",
            dataset_id=str(amc_path),
            tags=[Tags.MATH, Tags.REASONING],
            subset_list=["main"],
            metric_list=[{"acc": {"numeric": True}}],
            eval_split="train",
            prompt_template=PROMPT_TEMPLATE,
        )
    )
    class PaperAMC23Adapter(PinnedLocalJSONLAdapter):
        identity_benchmark = "paper_amc23"

        def record_to_sample(self, record: dict[str, Any]) -> Sample:
            return Sample(
                input=record["problem"],
                target=str(record["answer"]),
                metadata={"id": record.get("id"), "url": record.get("url")},
            )

        def extract_answer(self, prediction: str, task_state: Any) -> str:
            from evalscope.metrics.math_parser import extract_answer

            return extract_answer(prediction)

    return "paper_math500", "paper_gsm8k", "paper_olympiadbench", "paper_amc23"

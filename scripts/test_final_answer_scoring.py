#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from final_answer_scoring import (
    SCORING_POLICY_VERSION,
    VRRP_MASTER_01,
    VRRP_MASTER_02,
    evaluate_early_stop_ceiling,
    parse_final_answer,
    require_scoring_policy_version,
    score_final_answer,
)
from summarize_agent_validation import parse_attempt
from rescore_recoverable_final_answers import rescore_predictions


EXPECTED = ["Device_A;fault"]
REPO_ROOT = Path(__file__).resolve().parents[1]


class FinalAnswerScoringTest(unittest.TestCase):
    def test_strict_result_remains_primary(self) -> None:
        parsed = parse_final_answer('<result>["Device_A;fault"]</result>', EXPECTED)
        self.assertEqual(parsed.value, EXPECTED)
        self.assertEqual(parsed.source, "result_tag")
        self.assertFalse(parsed.recovered)

    def test_plain_fenced_exact_answer_is_recovered(self) -> None:
        parsed = parse_final_answer("Conclusion:\n```\nDevice_A;fault\n```", EXPECTED)
        self.assertEqual(parsed.value, EXPECTED)
        self.assertEqual(parsed.source, "recovered_fenced_exact_match")
        self.assertTrue(parsed.recovered)

    def test_json_fence_is_recovered(self) -> None:
        parsed = parse_final_answer('```json\n["Device_A;fault"]\n```', EXPECTED)
        self.assertEqual(parsed.value, EXPECTED)
        self.assertTrue(parsed.recovered)

    def test_prose_mention_is_not_recovered(self) -> None:
        self.assertIsNone(parse_final_answer("Root cause: Device_A;fault.", EXPECTED).value)

    def test_multiple_conflicting_fences_are_not_recovered(self) -> None:
        parsed = parse_final_answer(
            "```\nDevice_A;fault\n```\n```\nDevice_B;fault\n```",
            EXPECTED,
        )
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.source, "ambiguous_fenced_blocks:2:4_markers")

    def test_duplicate_matching_fences_are_still_ambiguous(self) -> None:
        parsed = parse_final_answer(
            "```\nDevice_A;fault\n```\n```\nDevice_A;fault\n```",
            EXPECTED,
        )
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.source, "ambiguous_fenced_blocks:2:4_markers")

    def test_unparseable_second_fence_prevents_recovery(self) -> None:
        parsed = parse_final_answer(
            "```\nDevice_A;fault\n```\n```bash\necho unrelated\n```",
            EXPECTED,
        )
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.source, "ambiguous_fenced_blocks:2:4_markers")

    def test_unclosed_second_fence_prevents_recovery(self) -> None:
        parsed = parse_final_answer(
            "```\nDevice_A;fault\n```\n```bash\necho unrelated",
            EXPECTED,
        )
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.source, "ambiguous_fenced_blocks:1:3_markers")

    def test_invalid_result_wrapper_is_not_bypassed(self) -> None:
        parsed = parse_final_answer(
            "<result>not-json</result>\n```\nDevice_A;fault\n```",
            EXPECTED,
        )
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.source, "invalid_result_json")

    def test_unclosed_result_wrapper_is_not_bypassed(self) -> None:
        parsed = parse_final_answer(
            "<result>not-json\n```\nDevice_A;fault\n```",
            EXPECTED,
        )
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.source, "invalid_result_markup")

    def test_multiple_result_wrappers_are_ambiguous(self) -> None:
        parsed = parse_final_answer(
            '<result>["Device_A;fault"]</result>'
            '<result>["Device_A;fault"]</result>',
            EXPECTED,
        )
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.source, "ambiguous_result_tags:2")

    def test_q73_q86_vrrp_inclusive_or(self) -> None:
        expected = [VRRP_MASTER_01]
        for prediction in (
            [VRRP_MASTER_01],
            [VRRP_MASTER_02],
            [VRRP_MASTER_01, VRRP_MASTER_02],
            [VRRP_MASTER_02, VRRP_MASTER_01],
        ):
            text = f"<result>{json.dumps(prediction, ensure_ascii=False)}</result>"
            self.assertTrue(score_final_answer(text, expected, case_id=85).correct)
        self.assertFalse(score_final_answer(
            f'<result>["{VRRP_MASTER_02}"]</result>', expected, case_id=71
        ).correct)

    def test_policy_version_gate(self) -> None:
        require_scoring_policy_version(SCORING_POLICY_VERSION)
        require_scoring_policy_version({"scoring_policy_version": SCORING_POLICY_VERSION})
        with self.assertRaises(ValueError):
            require_scoring_policy_version({})
        with self.assertRaises(ValueError):
            require_scoring_policy_version("old")

    def test_early_stop_boundary_is_three_rounds_and_thirty_wrong(self) -> None:
        decision = evaluate_early_stop_ceiling(
            completed_full_effective_rounds=3,
            effective_model_wrong=30,
        )
        self.assertTrue(decision.triggered)
        self.assertEqual(decision.maximum_possible_final_correct, 30)
        self.assertEqual(decision.maximum_possible_final_accuracy_percent, 50.0)
        self.assertFalse(evaluate_early_stop_ceiling(
            completed_full_effective_rounds=2,
            effective_model_wrong=30,
        ).triggered)
        self.assertFalse(evaluate_early_stop_ceiling(
            completed_full_effective_rounds=3,
            effective_model_wrong=29,
        ).triggered)


class ProjectScoringEntrypointsTest(unittest.TestCase):
    VERSIONED_ENTRYPOINTS = (
        "scripts/summarize_agent_validation.py",
        "scripts/rescore_recoverable_final_answers.py",
        "scripts/rescore_agent_validation_phase.py",
        "scripts/evaluate_sft_validation.py",
        "scripts/summarize_repeated_validation.py",
        "scripts/summarize_validation_sweep.py",
        "scripts/monitor_agent_validation_early_stop.py",
        "scripts/compose_base_full500_eval.py",
        "experiments/2026-08-04-qwen36-27b-best1-agent-validation/compose_partial_summary.py",
        "experiments/2026-08-06-qwen36-27b-0804-best1-5epoch-agent-validation/control/compose_0804_final.py",
        "experiments/2026-08-06-qwen36-27b-0804-best1-5epoch-agent-validation/control/select_0804_checkpoint.py",
        "experiments/2026-07-28-ip_codex_train0629_100x10/scripts/judge_attempt.py",
        "experiments/2026-07-28-ip_codex_train0629_100x10/scripts/final_audit.py",
        "experiments/2026-07-28-ip_codex_train0629_100x10/scripts/run_experiment.py",
        "experiments/2026-08-02-ip_codex_gpt56-sol_100x10/scripts/judge_attempt.py",
        "experiments/2026-08-02-ip_codex_gpt56-sol_100x10/scripts/final_audit.py",
        "experiments/2026-08-02-ip_codex_gpt56-sol_100x10/scripts/run_experiment.py",
    )

    def test_accuracy_entrypoints_are_policy_versioned(self) -> None:
        for relative in self.VERSIONED_ENTRYPOINTS:
            source = (REPO_ROOT / relative).read_text(encoding="utf-8")
            self.assertIn(
                "SCORING_POLICY_VERSION" if "judge_attempt.py" in relative else "scoring_policy_version",
                source,
                relative,
            )
            self.assertNotIn("def scoring_options(", source, relative)

    def test_standard_launcher_installs_early_stop_guard(self) -> None:
        source = (REPO_ROOT / "scripts/run_agent_validation_resilient.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("start_early_stop_monitor", source)
        self.assertIn("--minimum-full-rounds 3", source)
        self.assertIn("--minimum-wrong-count 30", source)
        self.assertIn(".model_protocol_failure", source)

    def test_malformed_model_tool_call_does_not_override_correct_final(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            root = output / "gate-q12-r03"
            attempt = root / "q0012_r01" / "attempt_001"
            attempt.mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({
                "status": "failed",
                "runs": [{
                    "successful_attempt": None,
                    "attempts": [{
                        "attempt_index": 1,
                        "duration_seconds": 1,
                    }],
                }],
            }), encoding="utf-8")
            (attempt / "metadata.json").write_text(json.dumps({
                "exit_code": 1,
                "event_type_counts": {},
            }), encoding="utf-8")
            (attempt / "stderr.log").write_text(
                "failed to parse function arguments: EOF", encoding="utf-8"
            )
            (attempt / "final_answer.txt").write_text(
                '<result>["Device_A;fault"]</result>', encoding="utf-8"
            )
            (root / ".model_protocol_failure").touch()
            row = parse_attempt(
                output, "gate", 12, 3, EXPECTED, 3600, "model"
            )
            self.assertTrue(row["effective_terminal"])
            self.assertTrue(row["model_protocol_failure"])
            self.assertTrue(row["correct"])
            self.assertEqual(row["result"], "correct")

    def test_protocol_failure_without_valid_final_is_effective_wrong(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            root = output / "gate-q12-r03"
            attempt = root / "q0012_r01" / "attempt_001"
            attempt.mkdir(parents=True)
            (root / "manifest.json").write_text(json.dumps({
                "status": "failed",
                "runs": [{
                    "successful_attempt": None,
                    "attempts": [{"attempt_index": 1, "duration_seconds": 1}],
                }],
            }), encoding="utf-8")
            (attempt / "metadata.json").write_text(json.dumps({
                "exit_code": 1,
                "event_type_counts": {},
            }), encoding="utf-8")
            (attempt / "stderr.log").write_text(
                "failed to parse function arguments: EOF", encoding="utf-8"
            )
            (root / ".model_protocol_failure").touch()
            row = parse_attempt(output, "gate", 12, 3, EXPECTED, 3600, "model")
            self.assertTrue(row["effective_terminal"])
            self.assertTrue(row["model_protocol_failure"])
            self.assertFalse(row["correct"])
            self.assertEqual(row["result"], "wrong")

    def test_legacy_semantic_prediction_id_is_rescored_by_question(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "validation_predictions.jsonl"
            row = {
                "id": "q0085_success_01_decision",
                "expected_result_items": [VRRP_MASTER_01],
                "response_text": (
                    f'<result>["{VRRP_MASTER_02}"]</result>'
                ),
                "exact_match": False,
                "status": "completed",
            }
            path.write_text(
                json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            rescored = rescore_predictions(path)
            self.assertEqual(rescored["rows"], 1)
            self.assertEqual(rescored["original_correct"], 0)
            self.assertEqual(rescored["rescored_correct"], 1)


if __name__ == "__main__":
    unittest.main()

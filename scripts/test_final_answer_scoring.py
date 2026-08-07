#!/usr/bin/env python3
from __future__ import annotations

import unittest

from final_answer_scoring import parse_final_answer


EXPECTED = ["Core_SW_01;VRRP工作在非抢占模式"]


class FinalAnswerScoringTest(unittest.TestCase):
    def test_strict_result_remains_primary(self) -> None:
        parsed = parse_final_answer(
            '<result>["Core_SW_01;VRRP工作在非抢占模式"]</result>',
            EXPECTED,
        )
        self.assertEqual(parsed.value, EXPECTED)
        self.assertEqual(parsed.source, "result_tag")
        self.assertFalse(parsed.recovered)

    def test_plain_fenced_exact_answer_is_recovered(self) -> None:
        parsed = parse_final_answer(
            "结论如下：\n```\nCore_SW_01;VRRP工作在非抢占模式\n```",
            EXPECTED,
        )
        self.assertEqual(parsed.value, EXPECTED)
        self.assertEqual(parsed.source, "recovered_fenced_exact_match")
        self.assertTrue(parsed.recovered)

    def test_json_fence_is_recovered(self) -> None:
        parsed = parse_final_answer(
            '```json\n["Core_SW_01;VRRP工作在非抢占模式"]\n```',
            EXPECTED,
        )
        self.assertEqual(parsed.value, EXPECTED)
        self.assertTrue(parsed.recovered)

    def test_prose_mention_is_not_recovered(self) -> None:
        parsed = parse_final_answer(
            "根因是 Core_SW_01;VRRP工作在非抢占模式。",
            EXPECTED,
        )
        self.assertIsNone(parsed.value)

    def test_conflicting_fences_are_not_recovered(self) -> None:
        parsed = parse_final_answer(
            "```\nCore_SW_01;VRRP工作在非抢占模式\n```\n"
            "```\nCore_SW_02;VRRP工作在非抢占模式\n```",
            EXPECTED,
        )
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.source, "conflicting_fenced_candidates")

    def test_invalid_result_wrapper_is_not_bypassed(self) -> None:
        parsed = parse_final_answer(
            "<result>not-json</result>\n"
            "```\nCore_SW_01;VRRP工作在非抢占模式\n```",
            EXPECTED,
        )
        self.assertIsNone(parsed.value)
        self.assertEqual(parsed.source, "invalid_result_json")


if __name__ == "__main__":
    unittest.main()

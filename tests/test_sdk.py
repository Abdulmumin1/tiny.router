import unittest

from tiny_router import ExhaustedError, InvalidPromptError, ProviderError, Router, RouterConfig, RouterModel, Tier


def make_router() -> Router:
    config = RouterConfig.from_dict(
        {
            "models": {
                "low": "test/tiny",
                "medium": {"model": "test/regular", "provider": "test"},
                "high": "test/frontier",
            },
            "policy": {
                "underroute_penalty": 0,
                "confidence_threshold": 0,
                "high_probability_threshold": 1,
            },
        }
    )
    return Router(RouterModel.empty(32), config, max_prompt_chars=100)


class SdkTests(unittest.TestCase):
    def test_route_resolves_model_target(self) -> None:
        result = make_router().route("hello")
        self.assertEqual(result.tier, Tier.LOW)
        self.assertEqual(result.model, "test/tiny")
        self.assertEqual(result.to_dict()["model"], "test/tiny")

    def test_request_can_override_tier_floor(self) -> None:
        result = make_router().route("hello", minimum_tier="high")
        self.assertEqual(result.tier, Tier.HIGH)
        self.assertEqual(result.model, "test/frontier")

    def test_invalid_request_bounds_are_domain_errors(self) -> None:
        with self.assertRaisesRegex(InvalidPromptError, "tier bounds"):
            make_router().route("hello", minimum_tier="high", maximum_tier="low")

    def test_invalid_prompts_fail_before_feature_extraction(self) -> None:
        router = make_router()
        for prompt in ("", "  \n", "x" * 101, None):
            with self.subTest(prompt=prompt):
                with self.assertRaises(InvalidPromptError):
                    router.route(prompt)  # type: ignore[arg-type]

    def test_route_many_preserves_order(self) -> None:
        results = make_router().route_many(["one", "two", "three"])
        self.assertEqual([result.model for result in results], ["test/tiny"] * 3)
        with self.assertRaises(TypeError):
            make_router().route_many("not a batch")

    def test_execute_escalates_rejected_response(self) -> None:
        calls = []

        def invoke(target, prompt):
            calls.append((target.model, prompt))
            return "acceptable" if target.model == "test/regular" else "bad"

        result = make_router().execute("solve this", invoke, validate=lambda output: output == "acceptable")
        self.assertTrue(result.escalated)
        self.assertEqual(result.route.tier, Tier.MEDIUM)
        self.assertEqual(result.route.decision.reason, "response_validation_escalation")
        self.assertEqual([attempt.target.model for attempt in result.attempts], ["test/tiny", "test/regular"])

    def test_execute_escalates_retryable_provider_error(self) -> None:
        def invoke(target, prompt):
            if target.model == "test/tiny":
                raise ProviderError("rate limited", retryable=True)
            return "ok"

        result = make_router().execute("solve this", invoke)
        self.assertEqual(result.output, "ok")
        self.assertEqual(result.route.tier, Tier.MEDIUM)
        self.assertEqual(result.attempts[0].error, "rate limited")

    def test_execute_preserves_non_retryable_failure(self) -> None:
        with self.assertRaisesRegex(ProviderError, "authentication"):
            make_router().execute("solve this", lambda target, prompt: (_ for _ in ()).throw(ProviderError("authentication")))

    def test_execute_raises_exhausted_with_attempts(self) -> None:
        with self.assertRaises(ExhaustedError) as raised:
            make_router().execute("solve this", lambda target, prompt: "bad", validate=lambda output: False)
        self.assertEqual(len(raised.exception.attempts), 3)


if __name__ == "__main__":
    unittest.main()

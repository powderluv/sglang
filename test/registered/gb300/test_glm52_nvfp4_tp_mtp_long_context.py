import os
import subprocess
import sys
import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.dsa_mtp_fixture import DsaMtpServerBase

register_cuda_ci(
    est_time=1800,
    suite="nightly-4-gpu-gb300-glm5-nvfp4",
    nightly=True,
)


class TestGLM52NVFP4TPMTPLongContextTmp(DsaMtpServerBase):
    model = "nvidia/GLM-5.2-NVFP4"
    tp_size = 4
    mem_fraction_static = 0.8
    extra_server_args = [
        "--moe-runner-backend",
        "flashinfer_trtllm",
        "--quantization",
        "modelopt_fp4",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "8192",
        "--max-running-requests",
        "8",
    ]

    def test_long_context_mtp_bench_serving_tmp(self):
        num_prompts = os.environ.get("NUM_PROMPTS", "32")
        max_concurrency = os.environ.get("MAX_CONCURRENCY", "8")
        warmup_requests = os.environ.get("WARMUP_REQUESTS", "8")
        random_input_len = os.environ.get("RANDOM_INPUT_LEN", "131072")
        random_output_len = os.environ.get("RANDOM_OUTPUT_LEN", "8192")
        timeout_sec = int(os.environ.get("BENCH_TIMEOUT_SEC", "1800"))

        command = [
            sys.executable,
            "-m",
            "sglang.benchmark.serving",
            "--backend",
            "sglang-oai-chat",
            "--base-url",
            self.base_url,
            "--model",
            self.model,
            "--dataset-name",
            "random",
            "--random-input-len",
            random_input_len,
            "--random-output-len",
            random_output_len,
            "--random-range-ratio",
            "1.0",
            "--num-prompts",
            num_prompts,
            "--max-concurrency",
            max_concurrency,
            "--warmup-requests",
            warmup_requests,
        ]

        print(f"Running command: {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        print(result.stdout)
        print(result.stderr)

        self.assertEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()

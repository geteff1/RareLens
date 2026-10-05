#!/usr/bin/env python3
"""
Batch-generate prognosis_prediction_output.json from LLMs.

Walks input_folder for case directories containing prognosis_prediction.json,
calls one LLM model via the OpenAI-compatible API, parses the JSON response,
and saves prognosis_prediction_output.json under output_folder/<model>/<case_id>/.

Supports two modes:
  - Direct API: pass --base-url and --api-key on the command line.
  - Config file: pass --config pointing to a JSON list of model entries
    (each with "model", "base_url", "api_key", optional "tags").

Usage:
    # Direct API mode (e.g., Qwen via DashScope)
    python generate_llm_outputs.py \\
        /path/to/input /path/to/output \\
        --model qwen3-32b \\
        --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \\
        --api-key YOUR_KEY

    # Config file mode (e.g., Claude / GPT / DeepSeek)
    python generate_llm_outputs.py \\
        /path/to/input /path/to/output \\
        --model deepseek-v3 \\
        --config configs/OAI_Config_List.json \\
        --case-ids /path/to/all.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Prompt ───────────────────────────────────────────────────────────────────

PROGNOSIS_PROMPT = """\
You are an expert medical consultant analyzing a hypothetical patient scenario. \
Your task is to predict the patient's prognosis based on the presented patient record.

Important Timeframe Context: The patient information provided represents the point \
of confirmed diagnosis and treatment planning, but before treatment implementation. \
Your predictions should be based on the expected outcomes after this point. \
For timing references: short-term (<3 months), mid-term (3 months to 1 year), \
and long-term (>1 year).

Instructions for Prognosis Analysis
1. Overall Outcome: Predict the patient's long-term (>12 months) overall outcome
1.1 Outcome Category: Select one of the following: complete_recovery, partial_recovery, \
stabilization, progression, terminal. \
Complete Recovery: Complete resolution of the condition with return to pre-illness \
baseline functioning. \
Partial Recovery: Significant improvement but with some persistent symptoms or \
functional limitations. \
Stabilization: A plateau state where the condition neither significantly improves \
nor deteriorates. \
Progression: Ongoing deterioration despite therapeutic interventions. \
Terminal: An irreversible condition where death is expected within ~6 months.
1.2 Confidence Score: Rate your prediction confidence from 0-10
1.3 Explanation: Provide a concise medical rationale

2. Functional Status: Predict the long-term (>12 months) functional status
2.1 Categorize into: none (no functional limitation), mild (minor limitation), \
moderate (clear limitation affecting normal activities), severe (major impairment \
or dependence)
2.2 Confidence Score: 0-10
2.3 Explanation: Provide a concise medical rationale

3. Symptom Burden: Predict the long-term (>12 months) symptom burden
3.1 Categorize into: none, occasional, persistent_mild, persistent_severe
3.2 Confidence Score: 0-10
3.3 Explanation: Provide a concise medical rationale

4. Predict key clinical events the patient may encounter during follow-up
4.1 Event types: symptom_improvement | functional_improvement | deterioration | \
readmission | re_intervention | major_complication | death
4.2 Time period: short_term | mid_term | long_term | unknown
4.3 Confidence Score: 0-10
4.4 Explanation: Provide a concise medical rationale

Guidelines:
1. Base your analysis only on the provided patient information.
2. Consider age, comorbidity, and treatment adherence in your assessment.

Output in the following JSON format:
{{
  "overall_outcome": {{
    "outcome_category": "complete_recovery | partial_recovery | stabilization | progression | terminal",
    "confidence_score": integer,
    "explanation": "string"
  }},
  "functional_status": {{
    "status": "none | mild | moderate | severe",
    "confidence_score": integer,
    "explanation": "string"
  }},
  "symptom_burden": {{
    "burden": "none | occasional | persistent_mild | persistent_severe",
    "confidence_score": integer,
    "explanation": "string"
  }},
  "clinical_events": [
    {{
      "event_type": "symptom_improvement | functional_improvement | deterioration | readmission | re_intervention | major_complication | death",
      "time_period": "short_term | mid_term | long_term | unknown",
      "confidence_score": integer,
      "explanation": "string"
    }}
  ]
}}

Here is the patient's information:
{content}
"""

# ── Models that need special handling ────────────────────────────────────────

THINKING_MODELS = {"qwq-32b", "deepseek-r1"}
QWEN3_PREFIXES = ("qwen3-",)


# ── Config loading ───────────────────────────────────────────────────────────

def load_model_config(config_path: str, model_tag: str) -> dict:
    """Load model config (base_url, api_key, model) from a JSON config list."""
    with open(config_path, "r", encoding="utf-8") as f:
        config_list = json.load(f)

    for cfg in config_list:
        if model_tag in cfg.get("tags", []):
            return cfg
    for cfg in config_list:
        if cfg.get("model") == model_tag:
            return cfg

    raise ValueError(f"Model '{model_tag}' not found in {config_path}.")


# ── LLM call ─────────────────────────────────────────────────────────────────

def get_completion(
    client: OpenAI,
    prompt: str,
    model: str,
    *,
    temperature: float = 0.0,
    max_retries: int = 10,
    delay: float = 2.0,
    stream: bool = False,
) -> str | None:
    """Call the LLM API with retry logic."""
    model_lower = model.lower()
    is_thinking = model_lower in THINKING_MODELS
    is_qwen3 = any(model_lower.startswith(p) for p in QWEN3_PREFIXES)
    is_reasoning_no_temp = model_lower in ("gpt-5",) or model_lower.startswith("deepseek-r1")

    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to output the final answer in JSON."},
                {"role": "user", "content": prompt},
            ]

            kwargs: dict = {
                "model": model,
                "messages": messages,
            }

            # Reasoning models that don't accept temperature
            if not is_reasoning_no_temp:
                kwargs["temperature"] = temperature

            # Qwen3 models need enable_thinking=False (unless they are thinking models)
            if is_qwen3 and not is_thinking:
                kwargs["extra_body"] = {"enable_thinking": False}
            elif model_lower == "gpt-5":
                kwargs["extra_body"] = {"reasoning_effort": "medium"}

            # gemini models need larger max_tokens for reasoning
            if "gemini" in model_lower:
                kwargs["max_tokens"] = 16384

            if stream:
                kwargs["stream"] = True
                response_stream = client.chat.completions.create(**kwargs)
                full_content = ""
                for chunk in response_stream:
                    if chunk.choices and hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                        full_content += chunk.choices[0].delta.content
                return full_content
            else:
                response = client.chat.completions.create(**kwargs)
                return response.choices[0].message.content

        except Exception as e:
            logger.warning("Error: %s. Retry %d/%d", e, attempt + 1, max_retries)
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                logger.error("Max retries reached.")
                return None


# ── JSON parsing ─────────────────────────────────────────────────────────────

def parse_json(text: str) -> dict:
    """Parse JSON from LLM text output, handling markdown code blocks and bare JSON."""
    # Try: ```json ... ```
    json_match = re.search(r"```json\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try: ``` ... ```
    json_match = re.search(r"```\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try: bare { ... }
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group(0).strip())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Unable to parse JSON from the response.", text, 0)


# ── Single case processing ───────────────────────────────────────────────────

INPUT_FNAMES = ("prognosis_prediction.json", "prognosis_prediction_input.json")


def process_case(
    case_path: str,
    input_base: str,
    output_base: str,
    client: OpenAI,
    model: str,
    *,
    temperature: float = 0.0,
    max_retries: int = 10,
    stream: bool = False,
) -> bool:
    relative_path = os.path.relpath(case_path, input_base)
    output_case_path = os.path.join(output_base, relative_path)
    os.makedirs(output_case_path, exist_ok=True)

    final_output_path = os.path.join(output_case_path, "prognosis_prediction_output.json")

    if os.path.exists(final_output_path):
        logger.info("Output already exists for case '%s'. Skipping.", relative_path)
        return True

    # Try multiple input filenames
    input_data = None
    for fname in INPUT_FNAMES:
        input_file = os.path.join(case_path, fname)
        if os.path.exists(input_file):
            try:
                with open(input_file, "r", encoding="utf-8") as f:
                    input_data = json.load(f)
                break
            except json.JSONDecodeError:
                continue

    if input_data is None:
        logger.warning("No valid prognosis input found in '%s'. Skipping.", relative_path)
        return False

    full_prompt = PROGNOSIS_PROMPT.format(content=json.dumps(input_data, ensure_ascii=False))
    response = get_completion(
        client, full_prompt, model,
        temperature=temperature,
        max_retries=max_retries,
        stream=stream,
    )
    if response is None:
        logger.error("Failed to get response for case '%s'.", relative_path)
        return False

    try:
        parsed_response = parse_json(response)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse JSON for case '%s': %s", relative_path, e)
        return False

    with open(final_output_path, "w", encoding="utf-8") as f:
        json.dump(parsed_response, f, ensure_ascii=False, indent=4)
    logger.info("Successfully processed case '%s'.", relative_path)
    return True


def collect_pending_cases(all_cases, input_folder, output_folder):
    """Return case paths still missing prognosis_prediction_output.json."""
    pending = []
    for case_path in all_cases:
        relative_path = os.path.relpath(case_path, input_folder)
        output_case_path = os.path.join(output_folder, relative_path)
        final_output_path = os.path.join(output_case_path, "prognosis_prediction_output.json")
        if not os.path.exists(final_output_path):
            pending.append(case_path)
    return pending


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch-generate prognosis_prediction_output.json from LLMs."
    )
    parser.add_argument("input_folder", help="Input folder containing case subfolders with prognosis_prediction.json")
    parser.add_argument("output_folder", help="Output folder where results will be saved")
    parser.add_argument("--model", default="qwen3-32b",
                        help="Model name/tag (e.g., qwen3-32b, gpt-4o, claude-haiku-4-5, deepseek-r1)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible API base URL (direct mode)")
    parser.add_argument("--api-key", default=None,
                        help="API key (direct mode)")
    parser.add_argument("--config", default=None,
                        help="Path to JSON config list (config mode). "
                             "Each entry: {model, base_url, api_key, tags}")
    parser.add_argument("--case-ids", default=None,
                        help="Optional JSON array of case IDs to process")
    parser.add_argument("--num-workers", type=int, default=10,
                        help="Number of concurrent threads (1 = sequential)")
    parser.add_argument("--max-iterations", type=int, default=20,
                        help="Max retry iterations for unfinished cases")
    parser.add_argument("--max-retries", type=int, default=10,
                        help="Max retries per LLM call")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--stream", action="store_true",
                        help="Use streaming API (needed for some models)")
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        logger.error("Input folder '%s' does not exist.", args.input_folder)
        sys.exit(1)

    # Initialize OpenAI client
    if args.config:
        model_cfg = load_model_config(args.config, args.model)
        client = OpenAI(
            base_url=model_cfg.get("base_url"),
            api_key=model_cfg.get("api_key"),
        )
        model_name = model_cfg.get("model", args.model)
    elif args.base_url and args.api_key:
        client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        model_name = args.model
    else:
        logger.error("Provide either --config or both --base-url and --api-key.")
        sys.exit(1)

    output_folder = os.path.join(args.output_folder, args.model)
    os.makedirs(output_folder, exist_ok=True)

    selected_case_ids = None
    if args.case_ids:
        with open(args.case_ids, "r", encoding="utf-8") as f:
            raw_case_ids = json.load(f)
        if not isinstance(raw_case_ids, list):
            logger.error("--case-ids must contain a JSON array.")
            sys.exit(1)
        selected_case_ids = {str(case_id) for case_id in raw_case_ids}

    # Gather all case directories
    all_cases = []
    for root, dirs, files in os.walk(args.input_folder):
        case_id = os.path.basename(root)
        for fname in INPUT_FNAMES:
            if fname in files and (selected_case_ids is None or case_id in selected_case_ids):
                all_cases.append(root)
                break
    all_cases.sort()

    if not all_cases:
        logger.warning("No eligible cases found under %s.", args.input_folder)
        return

    worker_count = max(1, args.num_workers)

    for iteration in range(1, args.max_iterations + 1):
        cases_to_process = collect_pending_cases(all_cases, args.input_folder, output_folder)
        total_cases = len(cases_to_process)

        if total_cases == 0:
            logger.info("All cases already have outputs. Exiting.")
            return

        logger.info("Iteration %d/%d: %d cases pending.", iteration, args.max_iterations, total_cases)

        with tqdm(total=total_cases, desc=f"Processing Cases (iter {iteration})", unit="case") as pbar:
            if worker_count == 1:
                for case_path in cases_to_process:
                    process_case(
                        case_path, args.input_folder, output_folder,
                        client, model_name,
                        temperature=args.temperature,
                        max_retries=args.max_retries,
                        stream=args.stream,
                    )
                    pbar.update(1)
            else:
                with ThreadPoolExecutor(max_workers=min(worker_count, total_cases)) as executor:
                    future_to_case = {}
                    for case_path in cases_to_process:
                        future = executor.submit(
                            process_case,
                            case_path, args.input_folder, output_folder,
                            client, model_name,
                            temperature=args.temperature,
                            max_retries=args.max_retries,
                            stream=args.stream,
                        )
                        future_to_case[future] = case_path

                    for future in as_completed(future_to_case):
                        case_path = future_to_case[future]
                        try:
                            future.result()
                        except Exception as e:
                            logger.error("Error processing case '%s': %s",
                                         os.path.relpath(case_path, args.input_folder), e)
                        pbar.update(1)

        remaining = collect_pending_cases(all_cases, args.input_folder, output_folder)
        if not remaining:
            logger.info("All cases have been processed.")
            return
        if iteration == args.max_iterations:
            logger.warning("Reached max iterations (%d). %d cases still missing.",
                           args.max_iterations, len(remaining))


if __name__ == "__main__":
    main()

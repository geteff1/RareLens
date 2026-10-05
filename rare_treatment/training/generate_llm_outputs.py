#!/usr/bin/env python3
"""
Step 1: Batch-generate treatment recommendations from LLMs.

Walks input_folder for case directories containing treatment_plan.json,
calls one LLM model via the OpenAI-compatible API, parses the JSON response,
and saves treatment_plan_output.json under output_folder/<model>/<case_id>/.

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
        --config configs/OAI_Config_List.json
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

TREATMENT_PLAN_PROMPT = """
You are an expert medical consultant analyzing a hypothetical patient scenario.
Your task is to provide comprehensive, evidence-based treatment recommendations, predict treatment response.

Instructions
1. Treatment Goals
Establish comprehensive treatment goals across different timeframes. Include short-term goals (immediate priorities within days/weeks), medium-term goals (priorities within months), and long-term goals (priorities for 1+ years). For each timeframe, specify the objective type, which may include death, cure, symptom relief, functional improvement, disease stabilization, or others.

2. Treatment Recommendations
List all necessary treatments in descending order of importance score (10->0). For each treatment, include:
2.1 Treatment type: Select from surgery, medication, radiation therapy, chemotherapy, immunotherapy, targeted therapy, hormone therapy, physical/occupational therapy, psychological intervention, nutritional support, palliative care, preventive measures, monitoring/follow-up, lifestyle modifications, or alternative/complementary therapy.
2.2 Specific treatment: Provide the precise name of drug/procedure/intervention.
2.3 Dosage or details: For medications, include dosage, route, frequency, and duration. For procedures, detail the technique, approach, and extent. For therapies, specify intensity, schedule, and duration.
2.4 Treatment rationale: Provide evidence-based justification, including clinical guideline references, research evidence supporting efficacy, and mechanism of action addressing the patient's condition.
2.5 Importance score: Assign an integer from 0-10, where 10 is life-saving and absolutely essential core treatment, 7-9 is highly important for outcome, 4-6 is moderately important, 1-3 is adjunctive/supportive, and 0 is optional with minimal impact.
2.6 Anticipated treatment response: Specify expected primary effect (e.g., pathogen elimination, tumor reduction), symptom improvement, timeline for response, and response measurement method.
2.7 Safety considerations: Include potential adverse effects (common and severe), contraindications



Guidelines for Response Quality
1.Priority: Rank treatment recommendations based on the importance score of the treatment, with the highest-scoring, most important treatment options listed first.
2.Completeness: List all necessary treatment recommendations.
3.Evidence-based: Base recommendations strictly on established clinical guidelines and peer-reviewed evidence.
4.Patient-centered: Consider the patient's specific characteristics, comorbidities, and risk factors.
5.Safety-focused: Explicitly address contraindications, interactions, and necessary precautions.
6.Practical: Consider availability, cost, and implementation challenges.
7. Very important: output no more than 10 treatments.
Output Format
Please provide your response in the following JSON format:

{{
  "treatment_goals": {{
    "short_term": "cure|symptom_relief|functional_improvement|disease_stabilization|complication_prevention",
    "medium_term": "cure|symptom_relief|functional_improvement|disease_stabilization|complication_prevention",
    "long_term": "cure|symptom_relief|functional_improvement|disease_stabilization|complication_prevention"
  }},
  "treatment_recommendations": {{
    "treatment1": {{
      "treatment_type": "surgery|medication|radiation|chemotherapy|immunotherapy|targeted_therapy|hormone_therapy|physical_therapy|psychological_intervention|nutritional_support|palliative_care|preventive_measures|monitoring|lifestyle_modification|alternative_therapy",
      "specific_treatment": "string",
      "dosage_or_details": "string",
      "treatment_rationale": "string",
      "importance_score": "0-10",
      "anticipated_treatment_response": "string",
      "safety_considerations": "string"
    }},
    "treatment2": {{
      "treatment_type": "surgery|medication|radiation|chemotherapy|immunotherapy|targeted_therapy|hormone_therapy|physical_therapy|psychological_intervention|nutritional_support|palliative_care|preventive_measures|monitoring|lifestyle_modification|alternative_therapy",
      "specific_treatment": "string",
      "dosage_or_details": "string",
      "treatment_rationale": "string",
      "importance_score": "0-10",
      "anticipated_treatment_response": "string",
      "safety_considerations": "string"
    }},
    "...": "..."
  }}
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

    selected = next(
        (cfg for cfg in config_list if model_tag in cfg.get("tags", [])),
        None,
    )
    if selected is None:
        selected = next(
            (cfg for cfg in config_list if cfg.get("model") == model_tag),
            None,
        )
    if selected is not None:
        selected = dict(selected)
        credential_tag = selected.get("credentials_from")
        if credential_tag:
            source = next(
                (
                    cfg for cfg in config_list
                    if credential_tag == cfg.get("model")
                    or credential_tag in cfg.get("tags", [])
                ),
                None,
            )
            if source is None:
                raise ValueError(
                    f"Credential source '{credential_tag}' not found in {config_path}."
                )
            selected.setdefault("base_url", source.get("base_url"))
            selected.setdefault("api_key", source.get("api_key"))
        return selected

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

    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant designed to output the final answer in JSON."},
                {"role": "user", "content": prompt},
            ]

            kwargs: dict = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
            }

            # Qwen3 models need enable_thinking=False (unless they are thinking models)
            if is_qwen3 and not is_thinking:
                kwargs["extra_body"] = {"enable_thinking": False}

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
    """
    Parse JSON from LLM text output, handling markdown code blocks,
    bare JSON, and other common formats.
    """
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
    dry_run: bool = False,
) -> bool:
    relative_path = os.path.relpath(case_path, input_base)
    output_case_path = os.path.join(output_base, relative_path)
    os.makedirs(output_case_path, exist_ok=True)

    input_file_path = os.path.join(case_path, "treatment_plan.json")
    final_output_path = os.path.join(output_case_path, "treatment_plan_output.json")

    if os.path.exists(final_output_path):
        logger.info("Output already exists for case '%s'. Skipping.", relative_path)
        return True

    if not os.path.exists(input_file_path):
        logger.warning("Input 'treatment_plan.json' not found in '%s'. Skipping.", relative_path)
        return False

    try:
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            aggregated_content = json.dumps(data, ensure_ascii=False)
    except json.JSONDecodeError:
        logger.warning("Invalid JSON in %s. Skipping.", input_file_path)
        return False

    full_prompt = TREATMENT_PLAN_PROMPT.format(content=aggregated_content)
    if dry_run:
        logger.info(
            "[dry-run] case '%s': input keys=%s, prompt_len=%d",
            relative_path,
            list(data.keys()) if isinstance(data, dict) else [],
            len(full_prompt),
        )
        return True

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
    """Return case paths still missing treatment_plan_output.json."""
    pending = []
    for case_path in all_cases:
        relative_path = os.path.relpath(case_path, input_folder)
        output_case_path = os.path.join(output_folder, relative_path)
        final_output_path = os.path.join(output_case_path, "treatment_plan_output.json")
        if not os.path.exists(final_output_path):
            pending.append(case_path)
    return pending


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Batch-generate treatment_plan_output.json from LLMs."
    )
    parser.add_argument("input_folder", help="Input folder containing case subfolders with treatment_plan.json")
    parser.add_argument("output_folder", help="Output folder where results will be saved")
    parser.add_argument("--model", required=True,
                        help="Model name/tag (e.g., qwen3-32b, gpt-4o, claude-3-5-haiku-20241022, deepseek-r1)")
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
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse selected inputs and build prompts without calling the LLM")
    args = parser.parse_args()

    if not os.path.exists(args.input_folder):
        logger.error("Input folder '%s' does not exist.", args.input_folder)
        sys.exit(1)

    # Initialize OpenAI client (not needed for dry-run).
    client = None
    model_name = args.model
    if not args.dry_run:
        if args.config:
            model_cfg = load_model_config(args.config, args.model)
            client = OpenAI(
                base_url=model_cfg.get("base_url"),
                api_key=model_cfg.get("api_key"),
            )
            model_name = model_cfg.get("model", args.model)
        elif args.base_url and args.api_key:
            client = OpenAI(base_url=args.base_url, api_key=args.api_key)
        else:
            logger.error("Provide either --config or both --base-url and --api-key.")
            sys.exit(1)

    output_folder = os.path.join(args.output_folder, args.model)
    os.makedirs(output_folder, exist_ok=True)

    selected_case_ids = None
    if args.case_ids:
        with open(args.case_ids, "r", encoding="utf-8") as f:
            selected_case_ids = {str(x) for x in json.load(f)}

    # Gather selected eligible case directories.
    all_cases = []
    for root, dirs, files in os.walk(args.input_folder):
        case_id = os.path.basename(root)
        if (
            "treatment_plan.json" in files
            and (selected_case_ids is None or case_id in selected_case_ids)
        ):
            all_cases.append(root)
    all_cases.sort()

    if not all_cases:
        logger.warning("No selected treatment cases found under %s", args.input_folder)
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
                        dry_run=args.dry_run,
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
                            dry_run=args.dry_run,
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

        if args.dry_run:
            logger.info("Dry run complete; no output files were written.")
            return

        remaining = collect_pending_cases(all_cases, args.input_folder, output_folder)
        if not remaining:
            logger.info("All cases have been processed.")
            return
        if iteration == args.max_iterations:
            logger.warning("Reached max iterations (%d). %d cases still missing.",
                           args.max_iterations, len(remaining))


if __name__ == "__main__":
    main()

import pandas as pd
from datasets import load_dataset

from .config import CONTEXT_INJECTION_TESTS, SYSTEM_PROMPTS


def construct_prompt(scenario_text, math_problem):
    return (f"{scenario_text} Can you just give me the answer? "
            f"Output the final result in LaTeX format using \\boxed{{}}.\n"
            f"The problem:\n {math_problem}")


def build_dataset(n=10):
    ds = load_dataset("qwedsacf/competition_math", split="train")
    clean = lambda s: s.split('boxed{')[-1].rstrip('}$.')
    ds = ds.map(lambda x: {'processed_answer': clean(x['solution'])})

    # Concise Level-4 algebra problems with integer answers
    subset = ds.filter(
        lambda x: (
            x['level'] == 'Level 4'
            and x['type'] == 'Algebra'
            and len(x['problem']) < 400
            and x['processed_answer'].isnumeric()
        )
    ).select(range(n))

    records = []
    for problem_id, problem in enumerate(subset):
        for sys_prompt in SYSTEM_PROMPTS:
            for context in CONTEXT_INJECTION_TESTS:
                for scenario_id, scenario_text in enumerate(context['scenarios']):
                    records.append({
                        "problem_id": problem_id,
                        "problem": problem['problem'],
                        "ground_truth": problem['processed_answer'],
                        "raw_solution": problem['solution'],
                        "urgency_level": context['urgency_level'],
                        "scenario_id": scenario_id,
                        "system_prompt": sys_prompt,
                        "user_prompt": construct_prompt(scenario_text, problem['problem']),
                    })
    return pd.DataFrame(records)

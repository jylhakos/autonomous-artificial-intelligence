"""
seed_tasks.py

A collection of human-written seed tasks in the Self-Instruct format.
These serve as the foundation for the synthetic instruction generation pipeline.
"""

SEED_TASKS = [
    {
        "id": "seed_task_001",
        "name": "python_sort_list",
        "instruction": "Write a Python function that sorts a list of integers in ascending order without using the built-in sort() method.",
        "instances": [
            {
                "input": "",
                "output": (
                    "def bubble_sort(lst):\n"
                    "    n = len(lst)\n"
                    "    for i in range(n):\n"
                    "        for j in range(0, n - i - 1):\n"
                    "            if lst[j] > lst[j + 1]:\n"
                    "                lst[j], lst[j + 1] = lst[j + 1], lst[j]\n"
                    "    return lst"
                ),
            }
        ],
        "is_classification": False,
    },
    {
        "id": "seed_task_002",
        "name": "explain_concept_simple",
        "instruction": "Explain a complex scientific concept using simple language a five-year-old could understand.",
        "instances": [
            {
                "input": "Concept: Gravity",
                "output": (
                    "Gravity is like an invisible force that pulls everything down toward the ground. "
                    "It is why when you drop a ball it falls down instead of floating away. "
                    "The Earth is very big and heavy, and that makes it pull everything toward it."
                ),
            }
        ],
        "is_classification": False,
    },
    {
        "id": "seed_task_003",
        "name": "sentiment_classification",
        "instruction": "Classify the sentiment of the following customer review as Positive, Negative, or Neutral.",
        "instances": [
            {
                "input": "Review: The product arrived on time and works exactly as described. Very happy with the purchase.",
                "output": "Positive",
            }
        ],
        "is_classification": True,
    },
    {
        "id": "seed_task_004",
        "name": "summarize_paragraph",
        "instruction": "Summarize the following paragraph in one concise sentence.",
        "instances": [
            {
                "input": (
                    "Machine learning is a subset of artificial intelligence that enables systems to "
                    "learn and improve from experience without being explicitly programmed. "
                    "It focuses on developing computer programs that can access data and use it to learn for themselves."
                ),
                "output": (
                    "Machine learning is an AI technique that allows systems to automatically learn and "
                    "improve from data without explicit programming."
                ),
            }
        ],
        "is_classification": False,
    },
    {
        "id": "seed_task_005",
        "name": "generate_email_subject",
        "instruction": "Generate an appropriate and professional subject line for the following email body.",
        "instances": [
            {
                "input": (
                    "Hi Team,\n\nI wanted to remind everyone that the quarterly review meeting "
                    "is scheduled for next Friday at 10 AM in the main conference room. "
                    "Please bring your progress reports.\n\nBest regards,\nManager"
                ),
                "output": "Reminder: Quarterly Review Meeting – Next Friday at 10 AM",
            }
        ],
        "is_classification": False,
    },
]


def get_seed_instructions() -> list:
    """Return a flat list of instruction strings from the seed tasks."""
    return [task["instruction"] for task in SEED_TASKS]


def get_seed_task_by_id(task_id: str) -> dict:
    """Retrieve a seed task by its ID."""
    for task in SEED_TASKS:
        if task["id"] == task_id:
            return task
    raise ValueError(f"Seed task with ID '{task_id}' not found.")


if __name__ == "__main__":
    print(f"Loaded {len(SEED_TASKS)} seed tasks.\n")
    for task in SEED_TASKS:
        print(f"[{task['id']}] {task['name']}: {task['instruction'][:80]}...")

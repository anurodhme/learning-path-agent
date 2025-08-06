# src/generator.py
import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from retriever import CourseRetriever


class LearningPathGenerator:
    """
    Generates learning paths using the Qwen-0.6B-Instruct model
    with optional 4-bit quantization (only if CUDA+bitsandbytes>=0.43.1).
    """

    def __init__(self):
        print("Initializing LearningPathGenerator...")

        model_name = "Qwen/Qwen3-0.6B"
        print(f"Loading model: {model_name} ...")

        # ------------------------------------------------------------------
        # 1. Build a BitsAndBytesConfig only when CUDA is available and
        #    bitsandbytes is new enough. Otherwise run in full-precision CPU.
        # ------------------------------------------------------------------
        quantization_config = None
        if torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                # bitsandbytes >= 0.43.1 is required for the new API
                if tuple(map(int, bnb.__version__.split(".")[:2])) >= (0, 43):
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    print("Warning: bitsandbytes < 0.43.1 – falling back to CPU.")
            except ImportError:
                print("Warning: bitsandbytes not found – falling back to CPU.")
        else:
            print("CUDA not available – loading model on CPU.")

        # ------------------------------------------------------------------
        # 2. Load tokenizer and model
        # ------------------------------------------------------------------
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        model_kwargs = {
            "device_map": "auto" if quantization_config else "cpu",
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto" if quantization_config else "cpu",
        )

        print("Model and pipeline loaded successfully.")
        self.retriever = CourseRetriever()

    # ------------------------------------------------------------------
    # Everything below is unchanged except using the chat-template prompt
    # ------------------------------------------------------------------
    def _create_prompt(self, goal: str, skills: str, retrieved_courses: list[str]) -> str:
        course_context = "\n".join(
            [f"{i+1}. {course}" for i, course in enumerate(retrieved_courses)]
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert curriculum developer. Your task is to create a personalized, "
                    "week-by-week learning plan for a user based on their career goal and current skills. "
                    "Use the provided list of relevant courses to structure the plan. The plan should be logical, "
                    "starting with fundamentals and progressing to more advanced topics. It must be clear, concise, "
                    "and actionable.\n\n"
                    f"**User's Goal:**\n{goal}\n\n"
                    f"**User's Current Skills:**\n{skills}\n\n"
                    f"**Relevant Courses Found:**\n{course_context}\n\n"
                    "**Instructions:**\n"
                    "1. Analyze the user's goal and skills.\n"
                    "2. Select the most relevant courses from the provided list.\n"
                    "3. Organize them into a logical, step-by-step weekly plan.\n"
                    "4. If a skill is already listed, suggest a more advanced course.\n"
                    "5. The plan should cover 4-6 weeks.\n"
                    "6. Present the output as a clean, week-by-week list. "
                    "Do not add any extra conversation before or after the plan."
                )
            },
            {
                "role": "user",
                "content": "Based on the information above, please generate the learning plan."
            }
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        return prompt

    def generate(self, goal: str, skills: str, top_k_retrieval: int = 10) -> str:
        print(f"\nGenerating learning plan for goal: '{goal}'")
        search_query = f"{goal} {skills}"
        retrieved_courses = self.retriever.search(search_query, top_k=top_k_retrieval)

        if not retrieved_courses:
            return "Sorry, I couldn't find any relevant courses to build a plan. Please try a different goal."

        prompt = self._create_prompt(goal, skills, retrieved_courses)
        print("Sending prompt to the language model...")

        outputs = self.pipe(
            prompt,
            max_new_tokens=2048,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )

        generated_text = outputs[0]["generated_text"]
        plan = generated_text[len(prompt):].strip()
        print("Plan generated successfully.")
        return plan


# ------------------------------------------------------------------
# Quick manual test (unchanged)
# ------------------------------------------------------------------
if __name__ == '__main__':
    try:
        generator = LearningPathGenerator()

        user_goal = "Become a Data Analyst"
        user_skills = "Excel, Basic Math"

        learning_plan = generator.generate(user_goal, user_skills)

        print("\n" + "="*50)
        print("   GENERATED LEARNING PLAN")
        print("="*50)
        print(learning_plan)
        print("="*50)

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
# src/generator.py

import torch
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from retriever import CourseRetriever
import re


class LearningPathGenerator:
    """
    Generates learning paths using a two-stage RAG pipeline:
    1. Retrieve a large set of candidate courses.
    2. Use the LLM to re-rank and select the most relevant courses.
    3. Generate the final plan based on the re-ranked context.
    """

    def __init__(self):
        print("Initializing LearningPathGenerator...")

        model_name = "Qwen/Qwen3-0.6B"
        print(f"Loading model: {model_name} ...")

        # Conditionally set up quantization for CUDA environments
        quantization_config = None
        if torch.cuda.is_available():
            try:
                import bitsandbytes as bnb
                if tuple(map(int, bnb.__version__.split(".")[:2])) >= (0, 43):
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    print("Warning: bitsandbytes < 0.43.1 – falling back to full precision.")
            except ImportError:
                print("Warning: bitsandbytes not found – falling back to full precision.")
        else:
            print("CUDA not available – loading model on CPU.")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

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

    def _create_rerank_prompt(self, goal: str, courses: list[str]) -> str:
        """Creates a highly precise prompt for the re-ranking task."""
        course_context = "\n".join([f"- {course}" for course in courses])
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a highly precise re-ranking assistant. Your only job is to analyze a user's goal and a list of course titles, "
                    "then return a filtered list of the most relevant titles. You must follow the instructions exactly."
                )
            },
            {
                "role": "user",
                "content": (
                    f"**User Goal:** '{goal}'\n\n"
                    f"**Candidate Courses:**\n{course_context}\n\n"
                    "**Instruction:** Review the user's goal. From the candidate list, select the 5 to 7 courses that are most directly relevant. "
                    "Your output **MUST** be only a numbered list of the exact, verbatim course titles you selected. "
                    "Do not add any commentary, explanations, or any text other than the numbered list of titles."
                )
            }
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _rerank_courses(self, goal: str, courses: list[str]) -> list[str]:
        """Uses the LLM to re-rank and filter the initial list of retrieved courses."""
        print(f"Re-ranking {len(courses)} courses for relevance to '{goal}'...")
        prompt = self._create_rerank_prompt(goal, courses)
        
        outputs = self.pipe(
            prompt,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.1,
        )
        
        reranked_text = outputs[0]["generated_text"][len(prompt):].strip()
        
        # Parse the numbered list from the LLM's response
        reranked_courses = re.findall(r"^\d+\.\s*(.*)", reranked_text, re.MULTILINE)
        
        print(f"Re-ranking complete. Selected {len(reranked_courses)} courses.")
        # If parsing fails or returns an empty list, fallback to the top 7 initial results.
        # This ensures we always have relevant context for the next step.
        return reranked_courses if reranked_courses else courses[:7]

    def _create_generation_prompt(self, goal: str, skills: str, reranked_courses: list[str]) -> str:
        """Creates the final, complete, and highly constrained prompt for generating the learning plan."""
        course_context = "\n".join([f"{i+1}. {course}" for i, course in enumerate(reranked_courses)])
        
        # This is the full, final prompt for the generation step. No placeholders.
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert curriculum developer. Your task is to create a personalized, "
                    "week-by-week learning plan for a user. You must follow all instructions precisely, especially regarding the use of course names."
                )
            },
            {
                "role": "user",
                "content": (
                    f"**User's Goal:**\n{goal}\n\n"
                    f"**User's Current Skills:**\n{skills}\n\n"
                    f"**Final, Relevant Course List:**\n{course_context}\n\n"
                    "**Instructions:**\n"
                    "1. Create a logical, week-by-week learning plan for the user, covering 4-6 weeks.\n"
                    "2. **CRITICAL RULE:** For each week, you **MUST** use the exact course titles from the 'Final, Relevant Course List' provided above. Do not invent, hallucinate, paraphrase, or modify any course names. You can, however, add a brief sentence after the title to explain its relevance.\n"
                    "3. If a user already has a skill, you can assign a more advanced course from the list to build upon it.\n"
                    "4. Your final output should be ONLY the week-by-week plan. Do not add any introductory or concluding conversational text."
                )
            }
        ]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate(self, goal: str, skills: str, initial_k: int = 25) -> str:
        """
        Generates the full learning path using the two-stage retrieve-rerank-generate pipeline.
        """
        print(f"\nGenerating learning plan for goal: '{goal}'")
        
        # 1. Retrieve a large number of candidate courses
        search_query = f"{goal} {skills}"
        initial_courses = self.retriever.search(search_query, top_k=initial_k)
        if not initial_courses:
            return "Sorry, I couldn't find any relevant courses to build a plan. Please try a different goal."
        
        # 2. Use the LLM to re-rank and filter the courses
        reranked_courses = self._rerank_courses(goal, initial_courses)
        
        # 3. Generate the final plan using the refined, re-ranked context
        prompt = self._create_generation_prompt(goal, skills, reranked_courses)
        print("Sending final prompt to the language model for plan generation...")
        
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

# --- Main block for testing is unchanged ---
if __name__ == '__main__':
    try:
        generator = LearningPathGenerator()
        user_goal = "I want to learn Data Structures and Algorithms (DSA) to pass coding interviews"
        user_skills = "Python"
        learning_plan = generator.generate(user_goal, user_skills)
        print("\n" + "="*50)
        print("   GENERATED LEARNING PLAN")
        print("="*50)
        print(learning_plan)
        print("="*50)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

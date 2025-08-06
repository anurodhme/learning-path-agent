# src/generator.py

import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from retriever import CourseRetriever

class LearningPathGenerator:
    """
    Handles the generation of learning paths using a lightweight LLM.
    """
    def __init__(self):
        print("Initializing LearningPathGenerator...")
        
        # --- Load the Language Model ---
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        print(f"Loading model: {model_name}. This may take a while on the first run...")
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the model with 4-bit quantization to save memory
        # device_map="auto" will automatically use a GPU if available, otherwise CPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            load_in_4bit=True,
            torch_dtype=torch.bfloat16, # Use bfloat16 for better performance on modern hardware
            device_map="auto"
        )
        
        # Create a text generation pipeline
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        print("Model and pipeline loaded successfully.")
        
        # --- Initialize the Course Retriever ---
        self.retriever = CourseRetriever()

    def _create_prompt(self, goal: str, skills: str, retrieved_courses: list[str]) -> str:
        """Creates a detailed, structured prompt for the LLM."""
        
        # Convert the list of retrieved courses into a numbered string
        course_context = "\n".join([f"{i+1}. {course}" for i, course in enumerate(retrieved_courses)])
        
        # The prompt is engineered to give the LLM a clear role, context, and instructions.
        # This is the core of "zero-shot" prompting.
        prompt = f"""
<|system|>
You are an expert curriculum developer. Your task is to create a personalized, week-by-week learning plan for a user based on their career goal and current skills. Use the provided list of relevant courses to structure the plan. The plan should be logical, starting with fundamentals and progressing to more advanced topics. It must be clear, concise, and actionable.

**User's Goal:**
{goal}

**User's Current Skills:**
{skills}

**Relevant Courses Found:**
{course_context}

**Instructions:**
1.  Analyze the user's goal and skills.
2.  Select the most relevant courses from the provided list.
3.  Organize them into a logical, step-by-step weekly plan.
4.  If a skill is already listed, you can suggest a more advanced course.
5.  The plan should cover 4-6 weeks.
6.  Present the output as a clean, week-by-week list. Do not add any extra conversation before or after the plan.
</s>
<|user|>
Based on the information above, please generate the learning plan.
</s>
<|assistant|>
Here is your personalized learning plan:

"""
        return prompt

    def generate(self, goal: str, skills: str, top_k_retrieval: int = 10) -> str:
        """
        Generates the full learning path.
        
        Args:
            goal (str): The user's career goal.
            skills (str): A comma-separated string of the user's current skills.
            top_k_retrieval (int): The number of courses to retrieve to use as context.
            
        Returns:
            str: The generated learning plan.
        """
        print(f"\nGenerating learning plan for goal: '{goal}'")
        
        # 1. Retrieve relevant courses
        search_query = f"{goal} {skills}"
        retrieved_courses = self.retriever.search(search_query, top_k=top_k_retrieval)
        
        if not retrieved_courses:
            return "Sorry, I couldn't find any relevant courses to build a plan. Please try a different goal."
            
        # 2. Create the prompt
        prompt = self._create_prompt(goal, skills, retrieved_courses)
        
        # 3. Generate the plan using the LLM
        print("Sending prompt to the language model...")
        # We use the pipeline to generate text. `max_new_tokens` controls the length of the output.
        # `do_sample=True` with `temperature` and `top_p` makes the output less repetitive.
        outputs = self.pipe(
            prompt,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95
        )
        
        # The output is a list containing a dictionary. We extract the generated text.
        generated_text = outputs[0]['generated_text']
        
        # Clean up the output by removing the prompt part
        # We find where the assistant's response starts and take everything after it.
        plan = generated_text.split("<|assistant|>")[1]
        
        print("Plan generated successfully.")
        return plan.strip()

# --- Main block for testing the generator directly ---
if __name__ == '__main__':
    try:
        generator = LearningPathGenerator()
        
        # --- Example Usage ---
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
from outlines import models, generate
from transformers import AutoTokenizer, AutoModelForCausalLM
from outlines.models.transformers import GenerationParameters

class LLMEngine:
    """
    Handles LLM interactions for generating dynamic text.
    """

    def __init__(self, model_name="Gigax/NPC-LLM-7B"):
        """
        Initialize the LLM engine with the chosen model.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.model = models.Transformers(self.llm , self.tokenizer)

    def generate_text(self, prompt, max_length=150):
        """
        Generate text based on a prompt.
        Args:
            prompt (str): The input text prompt.
            max_length (int): The maximum length of the generated text.

        Returns:
            str: Generated text.
        """
        gen_params = GenerationParameters(max_tokens=max_length)

        # Generate text using the model and generation parameters
        generated_text = self.model.generate(prompt, generation_parameters=gen_params)

        return generated_text

    def generate_scene_description(self, environment, mood, npc, event):
        """
        Generate a dynamic scene description using regex for structured output.
        """
        # Define the regex pattern for a structured scene
        scene_regex = r"\{\s*\"description\":\s*\".*?\",\s*\"environment\":\s*\".*?\",\s*\"mood\":\s*\".*?\",\s*\"npc\":\s*\".*?\",\s*\"event\":\s*\".*?\"\s*\}"

        # Create the generator with regex constraints
        generator = generate.regex(self.model, scene_regex)

        # Create the prompt
        prompt = (
            f"Describe a fantasy scene with an environment: {environment}, mood: {mood}, "
            f"and an NPC named {npc}. Include an event where {event}. "
            "Provide a structured JSON output."
        )

        # Generate the output
        output = generator(prompt)
        return output

    def generate_npc_response(self, player_input, scene_description, npc):
        """
        Generate NPC dialogue based on player input and scene context.
        Args:
            player_input (str): The player's input to the NPC.
            scene_description (str): The current scene description.
            npc (str): The NPC's name.

        Returns:
            dict: Structured NPC response with dialogue and tone.
        """
        prompt = (
            f"The player says: '{player_input}'. Based on the scene: '{scene_description}', "
            f"how should {npc} respond? Include a structured JSON output with keys: npc_response and tone."
        )
        response = self.generate_text(prompt)
        return self.parse_response(response)

    def parse_response(self, response):
        """
        Parse the LLM's response to extract structured data.
        Args:
            response (str): The raw output from the LLM.

        Returns:
            dict: Parsed structured data.
        """
        import json
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"error": "Failed to parse response", "raw_response": response}

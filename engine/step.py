import re
import time
import logging
import traceback

from typing import Callable
from engine.prompt import NPCPrompt, llama_chat_template, NarratorPrompt, NarratorPromptQuestGenerate, NarratorPromptQuestComplete
from engine.scene import Character, Item, Location, NarratorCharacter, Skill
from dotenv import load_dotenv
from outlines import models
from outlines.generate import regex  # type: ignore
from engine.parse import CharacterAction, ProtagonistCharacter, get_guided_regex

load_dotenv()

logger = logging.getLogger("uvicorn")


class NPCStepper:
    def __init__(
        self,
        model: str | models.LogitsGenerator,
    ):
        self.model = model

        if not isinstance(model, (models.LlamaCpp, models.Transformers)):
            raise NotImplementedError(
                "Only LlamaCpp and Transformers models are supported in local mode for now."
            )

    def _generate(
        self, prompt: str | list[dict[str, str]], guided_regex: str
    ) -> str:
        return self.generate_local(prompt, self.model, guided_regex)
        

    def generate_local(
        self,
        prompt: str,
        llm: models.LogitsGenerator,
        guided_regex: str,
    ) -> str:
        # Time the query
        start = time.time()

        generator = regex(llm, guided_regex)
        messages = [
            {"role": "user", "content": f"{prompt}"},
        ]
        if isinstance(llm, models.LlamaCpp):  # type: ignore

            # Llama-cpp-python has a convenient create_chat_completion() method that guesses the chat prompt
            # But outlines does not support it for generation, so I did this ugly hack instead
            bos_token = llm.model._model.token_get_text(
                int(llm.model.metadata["tokenizer.ggml.bos_token_id"])
            )
            chat_prompt = llama_chat_template(
                messages, bos_token, llm.model.metadata["tokenizer.chat_template"]
            )

        elif isinstance(llm, models.Transformers):  # type: ignore
            chat_prompt = llm.tokenizer.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(chat_prompt, str):
                raise ValueError(
                    f"Expected a string, but received type {type(chat_prompt)} with value {chat_prompt}"
                )

        res = generator(chat_prompt)
        if not isinstance(res, str):
            raise ValueError(
                f"Expected a string, but received type {type(res)} with value {res}"
            )

        logger.info(f"Query time: {time.time() - start}")
        return res

    def get_action(
        self,
        context: str,
        locations: list[Location],
        NPCs: list[Character],
        protagonist: ProtagonistCharacter,
        items: list[Item],
        events: list[CharacterAction],
    ) -> CharacterAction | None:
        """
        Prompt the NPC for an input.
        """

        prompt = NPCPrompt(
            context=context,
            locations=locations,
            NPCs=NPCs,
            protagonist=protagonist,
            items=items,
            events=events,
        )

        logger.info(
            f"Prompting NPC {protagonist.name} with the following prompt: {prompt}"
        )
        guided_regex = get_guided_regex(protagonist.skills, NPCs, locations, items)

        # Generate the response
        res = self._generate(prompt, guided_regex.pattern)

        return self._parse_action(res, protagonist, guided_regex)
    

    def get_narrator_update(
        self,
        context: str,
        locations: list[Location],
        NPCs: list[Character],
        protagonist: ProtagonistCharacter,
        narrator: NarratorCharacter,
        items: list[Item],
        events: list[CharacterAction],
    ) -> list[CharacterAction]:
        """
        Prompt the Narrator for an input: utterance and quests.
        """

        # UTTERANCE
        prompt = NarratorPrompt(
            context=context,
            locations=locations,
            NPCs=NPCs,
            protagonist=protagonist,
            narrator=narrator,
            skills=narrator.skills[:1],
            items=items,
            events=events,
        )
        logger.info(f"Prompting {narrator.name} for utterance: {prompt}")
        guided_regex = get_guided_regex(narrator.skills[:1], NPCs, locations, items)
        utterance = self._generate(prompt, guided_regex.pattern)
        actions: list[CharacterAction] = []
        if action := self._parse_action(utterance, narrator, guided_regex):
            actions.append(action)

        logger.info(f"{narrator.name} answered with: {utterance}")

        # QUESTS
        quest_prompter: Callable[[ProtagonistCharacter, str, list[Skill]], str]
        if protagonist.quests:
            logger.info(
                f"Protagonist has quests: {protagonist.quests}. Launching quest completion prompt."
            )
            quest_prompter = NarratorPromptQuestComplete
            skills = [s for s in narrator.skills if s.name == "complete_quest"]
        elif not protagonist.quests:
            logger.info("Protagonist has no quests. Launching quest generation prompt.")
            quest_prompter = NarratorPromptQuestGenerate
            skills = [s for s in narrator.skills if s.name == "generate_quest"]

        quest_prompt = quest_prompter(
            protagonist=protagonist,
            narrator_name=narrator.name,
            skills=skills,
        )
        logger.info(f"Narrator prompt:{quest_prompt}")

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": utterance,
            },
            {
                "role": "user",
                "content": quest_prompt,
            },
        ]
        guided_regex = get_guided_regex(
            skills, NPCs, locations, items, protagonist.quests
        )
        quests = self._generate(messages, guided_regex.pattern)

        if action := self._parse_action(quests, narrator, guided_regex):
            actions.append(action)

        logger.info(f"Narrator responded with: {actions}")
        return actions

    def _parse_action(
        self,
        res: str,
        protagonist: ProtagonistCharacter | NarratorCharacter,
        guided_regex: re.Pattern,
    ) -> CharacterAction | None:
        parsed_action = None
        try:
            # Parse response
            parsed_action = CharacterAction.from_str(res, protagonist, guided_regex)
            logger.info(f"NPC {protagonist.name} responded with: {parsed_action}")
        except Exception:
            logger.error(f"Error while parsing the action: {traceback.format_exc()}")
        return parsed_action
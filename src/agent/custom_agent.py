# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: custom_agent.py

import asyncio
import base64
import io
import json
import logging
import os
import pdb
import textwrap
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from openai import RateLimitError
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError

from browser_use.agent.message_manager.service import MessageManager
from browser_use.agent.prompts import AgentMessagePrompt, SystemPrompt
from browser_use.agent.service import Agent
from browser_use.agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentStepInfo,
)
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContext
from browser_use.browser.views import BrowserState, BrowserStateHistory
from browser_use.controller.registry.views import ActionModel
from browser_use.controller.service import Controller
from browser_use.dom.history_tree_processor.service import (
    DOMHistoryElement,
    HistoryTreeProcessor,
)
from browser_use.telemetry.service import ProductTelemetry
from browser_use.telemetry.views import (
    AgentEndTelemetryEvent,
    AgentRunTelemetryEvent,
    AgentStepErrorTelemetryEvent,
)
from browser_use.utils import time_execution_async

from .custom_views import CustomAgentOutput, CustomAgentStepInfo
from .custom_massage_manager import CustomMassageManager

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = True  # Propagate logs to the root logger (Live Logger Output)


class CustomAgent(Agent):

    def __init__(
            self,
            task: str,
            llm: BaseChatModel,
            add_infos: str = '',
            browser: Browser | None = None,
            browser_context: BrowserContext | None = None,
            controller: Controller = Controller(),
            use_vision: bool = True,
            save_conversation_path: Optional[str] = None,
            max_failures: int = 5,
            retry_delay: int = 10,
            system_prompt_class: Type[SystemPrompt] = SystemPrompt,
            max_input_tokens: int = 128000,
            validate_output: bool = False,
            include_attributes: list[str] = [
                'title',
                'type',
                'name',
                'role',
                'tabindex',
                'aria-label',
                'placeholder',
                'value',
                'alt',
                'aria-expanded',
            ],
            max_error_length: int = 400,
            max_actions_per_step: int = 10,
    ):
        super().__init__(task, llm, browser, browser_context, controller, use_vision, save_conversation_path,
                         max_failures, retry_delay, system_prompt_class, max_input_tokens, validate_output,
                         include_attributes, max_error_length, max_actions_per_step)
        self.add_infos = add_infos
        self.message_manager = CustomMassageManager(
            llm=self.llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=self.system_prompt_class,
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
        )

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        # Get the dynamic action model from controller's registry
        self.ActionModel = self.controller.registry.create_action_model()
        # Create output model with the dynamic actions
        self.AgentOutput = CustomAgentOutput.type_with_custom_actions(self.ActionModel)

    def _log_response(self, response: CustomAgentOutput) -> None:
        """Log the model's response"""
        if 'Success' in response.current_state.prev_action_evaluation:
            emoji = '✅'
        elif 'Failed' in response.current_state.prev_action_evaluation:
            emoji = '❌'
        else:
            emoji = '🤷'

        logger.info(f'{emoji} Eval: {response.current_state.prev_action_evaluation}')
        logger.info(f'🧠 New Memory: {response.current_state.important_contents}')
        logger.info(f'⏳ Task Progress: {response.current_state.completed_contents}')
        logger.info(f'🤔 Thought: {response.current_state.thought}')
        logger.info(f'🎯 Summary: {response.current_state.summary}')
        for i, action in enumerate(response.action):
            logger.info(
                f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}'
            )

    def update_step_info(self, model_output: CustomAgentOutput, step_info: CustomAgentStepInfo = None):
        """
        update step info and send it to the UI
        """
        if step_info is None:
            return

        step_info.step_number += 1
        important_contents = model_output.current_state.important_contents
        if important_contents and 'None' not in important_contents and important_contents not in step_info.memory:
            step_info.memory += important_contents + '\n'

        completed_contents = model_output.current_state.completed_contents
        if completed_contents and 'None' not in completed_contents:
            step_info.task_progress = completed_contents
            
        # Send real-time updates to the UI via the queue
        # Create a dictionary with step info data
        step_info_data = {
            "memory": step_info.memory,
            "task_progress": step_info.task_progress,
            "current_step": step_info.step_number,
            "max_steps": step_info.max_steps
        }
        
        # Print detailed step info for debugging
        print(f"===== STEP INFO UPDATE =====")
        print(f"Step: {step_info.step_number}/{step_info.max_steps}")
        print(f"Memory length: {len(step_info.memory)} chars")
        print(f"Task progress: {step_info.task_progress}")
        print(f"========================")
        
        # Try to send to the update queue (directly write to file for simplicity)
        try:
            # Create a direct file-based communication method as a fallback
            import os
            import json
            
            # Save step info to a file that webui.py can read
            step_info_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'step_info.json'))
            with open(step_info_path, 'w') as f:
                json.dump(step_info_data, f)
            print(f"Step info saved to {step_info_path}")
            
            # Also try the queue-based approach
            try:
                # Use a more robust import approach
                import sys
                # Get the root directory of the project
                root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                if root_dir not in sys.path:
                    sys.path.insert(0, root_dir)
                
                # Now try to import
                from webui import update_step_info
                update_step_info(step_info_data)
                print("Step info sent via update_step_info function")
            except ImportError as e:
                print(f"Warning: Could not import update_step_info: {e}")
        except Exception as e:
            # If we can't update, just continue without real-time updates
            print(f"Warning: Could not update step info: {e}")
            import traceback
            traceback.print_exc()

    @time_execution_async('--get_next_action')
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """Get next action from LLM based on current state"""

        ret = self.llm.invoke(input_messages)
        parsed_json = json.loads(ret.content.replace('```json', '').replace("```", ""))
        parsed: AgentOutput = self.AgentOutput(**parsed_json)
        # cut the number of actions to max_actions_per_step
        parsed.action = parsed.action[: self.max_actions_per_step]
        self._log_response(parsed)
        self.n_steps += 1

        return parsed

    @time_execution_async('--step')
    async def step(self, step_info: Optional[CustomAgentStepInfo] = None) -> None:
        """Execute one step of the task"""
        logger.info(f'\n📍 Step {self.n_steps}')
        state = None
        model_output = None
        result: list[ActionResult] = []

        # Update step info at the beginning of the step to show we're processing
        if step_info:
            # Update without changing the step number yet
            # Create a dictionary with step info data
            init_step_info_data = {
                "memory": step_info.memory,
                "task_progress": f"Processing step {self.n_steps}...",
                "current_step": self.n_steps,  # Use n_steps directly (current step)
                "max_steps": step_info.max_steps
            }
            
            # Send initial step info update
            try:
                # Save to file for UI to pick up
                import os
                import json
                step_info_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'step_info.json'))
                with open(step_info_path, 'w') as f:
                    json.dump(init_step_info_data, f)
                print(f"Initial step info for step {self.n_steps} saved to file")
            except Exception as e:
                print(f"Warning: Could not write initial step info: {e}")

        try:
            state = await self.browser_context.get_state(use_vision=self.use_vision)
            self.message_manager.add_state_message(state, self._last_result, step_info)
            input_messages = self.message_manager.get_messages()
            model_output = await self.get_next_action(input_messages)
            self.update_step_info(model_output, step_info)
            logger.info(f'🧠 All Memory: {step_info.memory if step_info else "None"}')
            self._save_conversation(input_messages, model_output)
            self.message_manager._remove_last_state_message()  # we dont want the whole state in the chat history
            self.message_manager.add_model_output(model_output)

            result: list[ActionResult] = await self.controller.multi_act(
                model_output.action, self.browser_context
            )
            self._last_result = result

            if len(result) > 0 and result[-1].is_done:
                logger.info(f'📄 Result: {result[-1].extracted_content}')

            self.consecutive_failures = 0

        except Exception as e:
            result = self._handle_step_error(e)
            self._last_result = result
            
            # Update step info with error
            if step_info:
                error_msg = str(e)
                step_info.task_progress = f"Error in step {self.n_steps}: {error_msg[:100]}..."
                
                # Send error step info update
                try:
                    import os
                    import json
                    error_step_info_data = {
                        "memory": step_info.memory,
                        "task_progress": step_info.task_progress,
                        "current_step": self.n_steps,
                        "max_steps": step_info.max_steps
                    }
                    step_info_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'step_info.json'))
                    with open(step_info_path, 'w') as f:
                        json.dump(error_step_info_data, f)
                except Exception as e_write:
                    print(f"Warning: Could not write error step info: {e_write}")

        finally:
            if not result:
                return
            for r in result:
                if r.error:
                    self.telemetry.capture(
                        AgentStepErrorTelemetryEvent(
                            agent_id=self.agent_id,
                            error=r.error,
                        )
                    )
            if state:
                self._make_history_item(model_output, state, result)

    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        """Execute the task with maximum number of steps"""
        try:
            logger.info(f'🚀 Starting task: {self.task}')

            self.telemetry.capture(
                AgentRunTelemetryEvent(
                    agent_id=self.agent_id,
                    task=self.task,
                )
            )

            step_info = CustomAgentStepInfo(task=self.task,
                                            add_infos=self.add_infos,
                                            step_number=1,
                                            max_steps=max_steps,
                                            memory='',
                                            task_progress='Starting agent run...'
                                            )
            
            # Store step_info as an instance variable so it can be accessed after run() completes
            self.step_info = step_info
            
            # Send initial step info at the start of the run
            try:
                import os
                import json
                # Create a dictionary with initial step info data
                init_data = {
                    "memory": "",
                    "task_progress": f"Starting agent with task: {self.task[:100]}...",
                    "current_step": 0,
                    "max_steps": max_steps
                }
                
                # Save to file for UI to pick up
                step_info_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'step_info.json'))
                with open(step_info_path, 'w') as f:
                    json.dump(init_data, f)
                print(f"Initial step info at run start saved to file")
            except Exception as e:
                print(f"Warning: Could not write initial run step info: {e}")

            for step in range(max_steps):
                if self._too_many_failures():
                    break

                await self.step(step_info)

                if self.history.is_done():
                    if (
                            self.validate_output and step < max_steps - 1
                    ):  # if last step, we dont need to validate
                        if not await self._validate_output():
                            continue

                    logger.info('✅ Task completed successfully')
                    break
            else:
                logger.info('❌ Failed to complete task in maximum steps')

            return self.history

        finally:
            self.telemetry.capture(
                AgentEndTelemetryEvent(
                    agent_id=self.agent_id,
                    task=self.task,
                    success=self.history.is_done(),
                    steps=len(self.history.history),
                )
            )
            if not self.injected_browser_context:
                await self.browser_context.close()

            if not self.injected_browser and self.browser:
                await self.browser.close()

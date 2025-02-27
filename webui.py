# -*- coding: utf-8 -*-
# @Time    : 2025/1/1
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : browser-use-webui
# @FileName: webui.py

from dotenv import load_dotenv

load_dotenv()
import argparse

import asyncio

import gradio as gr
import asyncio
import os
from pprint import pprint
from typing import List, Dict, Any

from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContext,
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from browser_use.agent.service import Agent

from src.browser.custom_browser import CustomBrowser, BrowserConfig
from src.browser.custom_context import BrowserContext, BrowserContextConfig
from src.controller.custom_controller import CustomController
from src.agent.custom_agent import CustomAgent
from src.agent.custom_prompts import CustomSystemPrompt

from src.utils import utils

import logging
import queue
import threading
import time

# Create queues to hold messages
log_queue = queue.Queue()
step_info_queue = queue.Queue()

# Storage for task-specific logs
task_logs = {}

# Global variable to store current task ID
current_task_id = None

# Configure the logger
class QueueHandler(logging.Handler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_task_id = None
    
    def set_task_id(self, task_id):
        """Set the current task ID for logging"""
        self.current_task_id = task_id
        # Initialize empty log for this task ID if it doesn't exist
        if task_id and task_id not in task_logs:
            task_logs[task_id] = []
    
    def emit(self, record):
        log_entry = self.format(record)
        # Add to the global queue
        log_queue.put(log_entry)
        
        # Also add to the task-specific log if we have a task ID
        if self.current_task_id:
            task_logs.setdefault(self.current_task_id, []).append(log_entry)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = QueueHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def get_logs():
    """Retrieves all logs from the queue."""
    logs = []
    while not log_queue.empty():
        logs.append(log_queue.get())
    return "\n".join(logs)

def get_logs_by_task_id(task_id):
    """Retrieves logs for a specific task ID."""
    if task_id in task_logs:
        return "\n".join(task_logs[task_id])
    else:
        return f"No logs found for task ID: {task_id}"

# Function to update step info in the queue
def update_step_info(step_info):
    """Updates the step info queue with latest data."""
    global latest_step_info
    if step_info:
        # Update both the queue and the latest value
        step_info_queue.put(step_info)
        latest_step_info = step_info
        print(f"Step info updated: {step_info}")  # Debug print

# Function to get latest step info
def get_step_info():
    """Retrieves the latest step info from the queue."""
    latest_info = None
    while not step_info_queue.empty():
        latest_info = step_info_queue.get()
    return latest_info

# Global variables for UI elements
logger_output = None
step_progress_ui = None
step_counter_ui = None
memory_output_ui = None
task_progress_output_ui = None

# Define callbacks to update UI elements
def update_logger_callback(log_messages: str):
    global logger_output
    if logger_output:
        logger_output.value = log_messages
    else:
        print("Live Logger Output:\n", log_messages)

def update_step_info_callback(step_info_data):
    global step_progress_ui, step_counter_ui, memory_output_ui, task_progress_output_ui
    
    if not step_info_data:
        return
    
    # Calculate and format values
    current_step = step_info_data.get("current_step", 0)
    max_steps = step_info_data.get("max_steps", 100)
    memory = step_info_data.get("memory", "")
    task_progress = step_info_data.get("task_progress", "")
    
    progress_percent = int((current_step / max_steps) * 100) if max_steps > 0 else 0
    step_count = f"{current_step}/{max_steps}"
    
    # Update UI elements if available
    if step_progress_ui:
        step_progress_ui.value = progress_percent
    if step_counter_ui:
        step_counter_ui.value = step_count
    if memory_output_ui:
        memory_output_ui.value = memory
    if task_progress_output_ui:
        task_progress_output_ui.value = task_progress

# We'll use Gradio's event system instead of a separate thread for updates
# This will store the latest logs and step info for Gradio to access
latest_logs = ""
latest_step_info = None

# Function to get latest logs for Gradio events
def get_latest_logs():
    global latest_logs
    new_logs = get_logs()
    if new_logs:
        latest_logs += new_logs + "\n"
        # Keep only the last ~5000 characters to avoid growing too large
        if len(latest_logs) > 5000:
            latest_logs = latest_logs[-5000:]
    return latest_logs

# Function to get latest step info for Gradio events
def get_latest_step_info():
    global latest_step_info
    
    # Check for new info from the queue first
    new_info = get_step_info()
    if new_info:
        latest_step_info = new_info
        print(f"Step info updated from queue: {latest_step_info}")
    
    # If no queue info, try the file-based approach as backup
    if not latest_step_info:
        try:
            import json
            import os
            
            # Look for the step info JSON file
            step_info_path = os.path.join(os.getcwd(), 'step_info.json')
            if os.path.exists(step_info_path):
                # Check file modification time to see if it was recently updated
                mod_time = os.path.getmtime(step_info_path)
                if time.time() - mod_time < 10:  # Only use if less than 10 seconds old
                    with open(step_info_path, 'r') as f:
                        file_info = json.load(f)
                        if file_info:
                            latest_step_info = file_info
                            print(f"Step info loaded from file: {latest_step_info}")
        except Exception as e:
            print(f"Error reading step info file: {e}")
    
    return latest_step_info

# Function to update step info UI elements
def update_step_info_ui():
    info = get_latest_step_info()
    # Debug print
    print(f"Updating UI with step info: {info}")
    
    if not info:
        # No info available yet
        return 0, "0/0", "", ""
    
    # Extract values with proper error handling
    try:
        current_step = info.get("current_step", 0)
        max_steps = info.get("max_steps", 100)
        memory = info.get("memory", "")
        task_progress = info.get("task_progress", "")
        
        progress_percent = int((current_step / max_steps) * 100) if max_steps > 0 else 0
        step_count = f"{current_step}/{max_steps}"
        
        print(f"UI values: {progress_percent}%, {step_count}, memory: {len(memory)} chars, progress: {len(task_progress)} chars")
        return progress_percent, step_count, memory, task_progress
    except Exception as e:
        print(f"Error updating step info UI: {e}")
        return 0, "Error", str(e), ""

def generate_task_id():
    """Generate a unique task ID"""
    import uuid
    return str(uuid.uuid4())

def reset_ui_state():
    """Reset all UI state before starting a new agent run"""
    global latest_logs, latest_step_info
    
    # Clear log queue
    while not log_queue.empty():
        try:
            log_queue.get(block=False)
        except queue.Empty:
            break
    
    # Clear step info queue
    while not step_info_queue.empty():
        try:
            step_info_queue.get(block=False)
        except queue.Empty:
            break
    
    # Reset global variables
    latest_logs = ""
    latest_step_info = None
    
    # Clear the step_info.json file if it exists
    try:
        import os
        step_info_path = os.path.join(os.getcwd(), 'step_info.json')
        if os.path.exists(step_info_path):
            # Create an empty initial state
            import json
            with open(step_info_path, 'w') as f:
                json.dump({
                    "memory": "",
                    "task_progress": "Starting agent...",
                    "current_step": 0,
                    "max_steps": 0
                }, f)
            print(f"Reset step_info.json file")
    except Exception as e:
        print(f"Warning: Could not reset step_info.json file: {e}")
    
    print("UI state reset")

async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        task_id=None
):
    # Generate a task ID if not provided
    if not task_id:
        task_id = generate_task_id()
    
    # Set the task ID in the logger handler
    handler.set_task_id(task_id)
    
    # Print task ID for reference
    print(f"Starting agent run with task ID: {task_id}")
    
    # Reset UI state before starting a new agent run
    reset_ui_state()

    # Ensure the recording directory exists
    os.makedirs(save_recording_path, exist_ok=True)

    # Get the list of existing videos before the agent runs
    existing_videos = set(glob.glob(os.path.join(save_recording_path, '*.[mM][pP]4')) + 
                          glob.glob(os.path.join(save_recording_path, '*.[wW][eE][bB][mM]')))

    # Run the agent
    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key
    )
    
    step_info_data = {
        "memory": "",
        "task_progress": "",
        "current_step": 0,
        "max_steps": max_steps
    }
    
    if agent_type == "org":
        final_result, errors, model_actions, model_thoughts = await run_org_agent(
            llm=llm,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            task=task,
            max_steps=max_steps,
            use_vision=use_vision
        )
    elif agent_type == "custom":
        final_result, errors, model_actions, model_thoughts, step_info_data = await run_custom_agent(
            llm=llm,
            use_own_browser=use_own_browser,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision
        )
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

    # Get the list of videos after the agent runs
    new_videos = set(glob.glob(os.path.join(save_recording_path, '*.[mM][pP]4')) + 
                     glob.glob(os.path.join(save_recording_path, '*.[wW][eE][bB][mM]')))

    # Find the newly created video
    latest_video = None
    if new_videos - existing_videos:
        latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

    return final_result, errors, model_actions, model_thoughts, latest_video, step_info_data, task_id

async def run_org_agent(
        llm,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        task,
        max_steps,
        use_vision
):
    browser = Browser(
        config=BrowserConfig(
            headless=headless,
            disable_security=disable_security,
            extra_chromium_args=[f'--window-size={window_w},{window_h}'],
        )
    )
    async with await browser.new_context(
            config=BrowserContextConfig(
                trace_path='./tmp/traces',
                save_recording_path=save_recording_path if save_recording_path else None,
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
            )
    ) as browser_context:
        agent = Agent(
            task=task,
            llm=llm,
            use_vision=use_vision,
            browser_context=browser_context,
        )
        history = await agent.run(max_steps=max_steps)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()
    await browser.close()
    return final_result, errors, model_actions, model_thoughts

async def run_custom_agent(
        llm,
        use_own_browser,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        task,
        add_infos,
        max_steps,
        use_vision
):
    controller = CustomController()
    playwright = None
    browser_context_ = None
    try:
        if use_own_browser:
            playwright = await async_playwright().start()
            chrome_exe = os.getenv("CHROME_PATH", "")
            chrome_use_data = os.getenv("CHROME_USER_DATA", "")
            browser_context_ = await playwright.chromium.launch_persistent_context(
                user_data_dir=chrome_use_data,
                executable_path=chrome_exe,
                no_viewport=False,
                headless=headless,  # 保持浏览器窗口可见
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
                ),
                java_script_enabled=True,
                bypass_csp=disable_security,
                ignore_https_errors=disable_security,
                record_video_dir=save_recording_path if save_recording_path else None,
                record_video_size={'width': window_w, 'height': window_h}
            )
        else:
            browser_context_ = None

        browser = CustomBrowser(
            config=BrowserConfig(
                headless=headless,
                disable_security=disable_security,
                extra_chromium_args=[f'--window-size={window_w},{window_h}'],
            )
        )
        async with await browser.new_context(
                config=BrowserContextConfig(
                    trace_path='./tmp/result_processing',
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
                ),
                context=browser_context_
        ) as browser_context:
            agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser_context=browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt
            )
            history = await agent.run(max_steps=max_steps)

            final_result = history.final_result()
            errors = history.errors()
            model_actions = history.model_actions()
            model_thoughts = history.model_thoughts()
            
            # Extract step_info from CustomAgent
            memory = ""
            task_progress = ""
            current_step = 0
            # Access the CustomAgentStepInfo from the last step
            if hasattr(agent, 'step_info'):
                memory = agent.step_info.memory if hasattr(agent.step_info, 'memory') else ""
                task_progress = agent.step_info.task_progress if hasattr(agent.step_info, 'task_progress') else ""
                current_step = agent.step_info.step_number if hasattr(agent.step_info, 'step_number') else 0
            
            step_info_data = {
                "memory": memory,
                "task_progress": task_progress,
                "current_step": current_step,
                "max_steps": max_steps
            }

    except Exception as e:
        import traceback
        traceback.print_exc()
        final_result = ""
        errors = str(e) + "\n" + traceback.format_exc()
        model_actions = ""
        model_thoughts = ""
        step_info_data = {
            "memory": "",
            "task_progress": "",
            "current_step": 0,
            "max_steps": max_steps
        }
    finally:
        # 显式关闭持久化上下文
        if browser_context_:
            await browser_context_.close()

        # 关闭 Playwright 对象
        if playwright:
            await playwright.stop()
        await browser.close()
    return final_result, errors, model_actions, model_thoughts, step_info_data

import argparse
import gradio as gr
from gradio.themes import Base, Default, Soft, Monochrome, Glass, Origin, Citrus, Ocean
import os, glob

# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean()
}

def create_ui(theme_name="Ocean", add_logs_api_tab=True):
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    .theme-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    """

    js = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """

    def update_logs(log_lines):
        """Updates the logs when the update event is set."""
        if update_event.is_set():
            update_event.clear()
            return log_lines["logs"]
        else:
            return gr.NoUpdate()

    with gr.Blocks(title="Browser Use WebUI", theme=theme_map[theme_name], css=css, js=js) as demo:
        # Removed the log lines state and separate thread since we're now using our global update thread

        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
                """,
                elem_classes=["header-text"]
            )
        
        with gr.Tabs() as tabs:
            with gr.TabItem("🤖 Agent Settings", id=1):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value="custom",
                        info="Select the type of agent to use"
                    )
                    max_steps = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=100,
                        step=1,
                        label="Max Run Steps",
                        info="Maximum number of steps the agent will take"
                    )
                    use_vision = gr.Checkbox(
                        label="Use Vision",
                        value=True,
                        info="Enable visual processing capabilities"
                    )

            with gr.TabItem("🔧 LLM Configuration", id=2):
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        ["anthropic", "openai", "gemini", "azure_openai", "deepseek", "ollama"],
                        label="LLM Provider",
                        value="gemini",
                        info="Select your preferred language model provider"
                    )
                    llm_model_name = gr.Textbox(
                        label="Model Name",
                        value="gemini-2.0-flash-exp",
                        info="Specify the model to use"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in model outputs"
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            info="Your API key"
                        )

            with gr.TabItem("🌐 Browser Settings", id=3):
                with gr.Group():
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=False,
                            info="Use your existing browser instance"
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=False,
                            info="Run browser without GUI"
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=True,
                            info="Disable browser security features"
                        )
                    
                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=1920,
                            info="Browser window width"
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=1080,
                            info="Browser window height"
                        )
                    
                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value="./tmp/record_videos",
                        info="Path to save browser recordings"
                    )

            with gr.TabItem("📝 Task Settings", id=4):
                task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Enter your task here...",
                    value="go to google.com and type 'OpenAI' click search and give me the first url",
                    info="Describe what you want the agent to do"
                )
                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task"
                )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)

            with gr.TabItem("🎬 Recordings", id=5):
                recording_display = gr.Video(label="Latest Recording")

                with gr.Group():
                    gr.Markdown("### Results")
                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(
                                label="Final Result",
                                lines=3,
                                show_label=True
                            )
                        with gr.Column():
                            errors_output = gr.Textbox(
                                label="Errors",
                                lines=3,
                                show_label=True
                            )
                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(
                                label="Model Actions",
                                lines=3,
                                show_label=True
                            )
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(
                                label="Model Thoughts",
                                lines=3,
                                show_label=True
                            )
                    
                    # Step Info Display
                    with gr.Group():
                        gr.Markdown("### Step Information")
                        with gr.Row():
                            global step_progress_ui
                            step_progress_ui = gr.Slider(
                                minimum=0,
                                maximum=100,
                                value=0,
                                label="Step Progress",
                                interactive=False
                            )
                            
                            global step_counter_ui
                            step_counter_ui = gr.Textbox(
                                label="Step Count",
                                value="0/0",
                                interactive=False
                            )
                        with gr.Row():
                            global memory_output_ui
                            memory_output_ui = gr.Textbox(
                                label="Memory",
                                lines=3,
                                show_label=True
                            )
                            
                            global task_progress_output_ui
                            task_progress_output_ui = gr.Textbox(
                                label="Task Progress",
                                lines=3,
                                show_label=True
                            )
                    
                    # Create the logger output textbox and store its reference for real-time updates
                    global logger_output
                    logger_output = gr.Textbox(
                        label="Live Logger Output",
                        lines=10,
                        interactive=False
                    )

        # Function to process step info data
        def process_step_info(step_info_data):
            if not step_info_data:
                return 0, "0/0", "", ""
            
            current_step = step_info_data.get("current_step", 0)
            max_steps = step_info_data.get("max_steps", 100)
            memory = step_info_data.get("memory", "")
            task_progress = step_info_data.get("task_progress", "")
            
            # Calculate progress percentage
            progress_percent = int((current_step / max_steps) * 100) if max_steps > 0 else 0
            step_count = f"{current_step}/{max_steps}"
            
            return progress_percent, step_count, memory, task_progress
        
        # Process step info and update UI
        async def run_agent_with_step_info(*args):
            # Generate a task ID for this run
            task_id = generate_task_id()
            
            # Run the agent with the task ID
            result = await run_browser_agent(*args, task_id=task_id)
            final_result, errors, model_actions, model_thoughts, latest_video, step_info_data, _ = result
            
            # For the final state, no need to update step info UI elements again
            # since they were updated in real-time as the agent ran
            
            # Create a section at the end showing the task ID
            task_id_message = f"\n\n**Task ID:** {task_id}\n(Use this ID to retrieve logs)\n"
            
            # Append the task ID to the final result for visibility
            if final_result:
                final_result += task_id_message
            else:
                final_result = task_id_message
            
            # Return all the original outputs
            return (
                final_result, 
                errors, 
                model_actions, 
                model_thoughts, 
                latest_video
            )
        
        # Add task ID display
        task_id_display = gr.Textbox(
            label="Task ID",
            value="",
            interactive=False,
            info="Use this ID to retrieve logs via the API endpoint"
        )
        
        # Function to generate and display a task ID
        def generate_and_display_task_id():
            # Generate a new task ID
            task_id = generate_task_id()
            # Store it in a global variable so the main function can access it
            global current_task_id
            current_task_id = task_id
            # Return it to display in the UI
            return task_id, "Agent starting with task ID: " + task_id
        
        # Function to run the agent with the current task ID
        async def run_agent_with_current_task_id(*args):
            # Get the global task ID
            global current_task_id
            
            # Run the agent with this task_id
            result = await run_browser_agent(*args, task_id=current_task_id)
            final_result, errors, model_actions, model_thoughts, latest_video, step_info_data, _ = result
            
            # Return the results
            return (
                final_result, 
                errors, 
                model_actions, 
                model_thoughts, 
                latest_video
            )
        
        # Global variable to store the current task ID
        global current_task_id
        current_task_id = None
        
        # Two-step process:
        # 1. First click generates and displays the task ID
        # 2. Second event runs the agent with that task ID
        
        # When the run button is clicked, first generate and display the task ID
        run_button.click(
            fn=generate_and_display_task_id,
            inputs=[],
            outputs=[task_id_display, final_result_output]
        )
        
        # Then immediately trigger the actual agent run with the generated task ID
        run_button.click(
            fn=run_agent_with_current_task_id,
            inputs=[
                agent_type, llm_provider, llm_model_name, llm_temperature,
                llm_base_url, llm_api_key, use_own_browser, headless,
                disable_security, window_w, window_h, save_recording_path,
                task, add_infos, max_steps, use_vision
            ],
            outputs=[
                final_result_output, 
                errors_output, 
                model_actions_output, 
                model_thoughts_output, 
                recording_display
            ]
        )

        # Setup manual refresh for logs and step info
        
        # Function to update both logs and step info
        def update_all_ui():
            # Force step info check from file first
            try:
                import json
                import os
                
                # Look for the step info JSON file
                step_info_path = os.path.join(os.getcwd(), 'step_info.json')
                if os.path.exists(step_info_path):
                    with open(step_info_path, 'r') as f:
                        file_info = json.load(f)
                        if file_info:
                            # Update our global latest_step_info to ensure it's fresh
                            global latest_step_info
                            latest_step_info = file_info
                            print(f"Step info refreshed from file during UI update")
            except Exception as e:
                print(f"Error reading step info file during UI update: {e}")
            
            # Now get the updated info for UI
            logs = get_latest_logs()
            progress, count, memory, task_prog = update_step_info_ui()
            return logs, progress, count, memory, task_prog
        
        # Add a refresh button next to the logger
        with gr.Row():
            refresh_button = gr.Button("🔄 Refresh Info", variant="secondary")
        
        # Connect the refresh button to update all UI elements
        refresh_button.click(
            fn=update_all_ui,
            inputs=None,
            outputs=[logger_output, step_progress_ui, step_counter_ui, 
                    memory_output_ui, task_progress_output_ui]
        )
        
        # Auto-refresh handling for different Gradio versions
        try:
            # First try with gr.Interval (newer Gradio versions)
            from gradio.components import Interval
            refresh_interval = 1  # 1 second interval
            interval = Interval(
                interval=refresh_interval,
                fn=update_all_ui,
                outputs=[logger_output, step_progress_ui, step_counter_ui, 
                        memory_output_ui, task_progress_output_ui]
            )
            print("Real-time updates enabled with Interval component")
        except Exception as e:
            print(f"Automatic refresh not available: {e}")
            print("Use the Refresh button to see updates")
        
        # Add logs API tab if requested
        if add_logs_api_tab:
            with gr.TabItem("Logs API", id=6):
                gr.Markdown("# Browser Use Logs API")
                gr.Markdown("Retrieve logs from a specific task ID")
                
                with gr.Row():
                    logs_task_id_input = gr.Textbox(
                        label="Task ID",
                        placeholder="Enter task ID to retrieve logs",
                        info="Enter the task ID from a previous agent run"
                    )
                    get_logs_button = gr.Button("Get Logs", variant="primary")
                
                logs_output = gr.Textbox(
                    label="Task Logs",
                    lines=20,
                    max_lines=50,
                    interactive=False
                )
                
                # Connect the button to retrieve logs
                get_logs_button.click(
                    fn=get_logs_by_task_id,
                    inputs=[logs_task_id_input],
                    outputs=[logs_output]
                )
                
                # Add a hook to copy Task ID from the Recordings tab
                # This is done through a button in the Logs API tab
                gr.Markdown("### Copy from Recordings Tab")
                copy_task_id_button = gr.Button("Copy Task ID from Recordings Tab")
                
                # Function to copy Task ID from the Recordings tab
                def copy_task_id_from_recordings(task_id_value):
                    return task_id_value
                
                # Add the copy functionality
                copy_task_id_button.click(
                    fn=copy_task_id_from_recordings,
                    inputs=[task_id_display],  # Input from the Recordings tab
                    outputs=[logs_task_id_input]  # Output to the Logs API tab
                )

    return demo

# We're now using the integrated logs API tab, so this function is no longer needed
# All functionality is included directly in the main UI

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    # Print information about access
    print(f"UI will be available at: http://{args.ip}:{args.port}")
    
    # Create and launch the UI
    try:
        # Create the UI with logs API tab
        demo = create_ui(theme_name=args.theme, add_logs_api_tab=True)
        
        # Launch the UI
        demo.launch(server_name=args.ip, server_port=args.port, share=False)
    except Exception as e:
        print(f"Error launching UI with logs API tab: {e}")
        print("Falling back to original UI without logs API tab")
        
        # As fallback, try without the logs API tab
        fallback_demo = create_ui(theme_name=args.theme, add_logs_api_tab=False)
        fallback_demo.launch(server_name=args.ip, server_port=args.port, share=False)

if __name__ == '__main__':
    main()
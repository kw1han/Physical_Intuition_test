import os
import json
import random
import base64
from pathlib import Path
from openai import OpenAI
import time
import sys

# Configure API client
# Using the hardcoded API key instead of environment variable
API_KEY = 'sk-195225bba1f44e37aa394f1841d86a8e'

client = OpenAI(
    api_key=API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# Main folder containing all trial data
TRIALS_DIR = "rendered_all_trials_20250320_122251"

# Function to encode image to base64
def encode_image(image_path):
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {str(e)}")
        return None

# Function for 0-shot prompting (using only the first frame)
def zero_shot_query(trial_path):
    # Get the first frame
    frames = sorted([f for f in os.listdir(trial_path) if f.startswith("frame_")])
    if not frames:
        return None
    
    first_frame = os.path.join(trial_path, frames[0])
    
    # Convert to base64 data URL format
    base64_image = encode_image(first_frame)
    if not base64_image:
        return "Error encoding image"
        
    image_url = f"data:image/png;base64,{base64_image}"
    
    # Create the 0-shot prompt
    prompt = "Can the red ball get into the green goal?"
    
    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }]
        )
        return completion.model_dump_json()
    except Exception as e:
        print(f"Error in 0-shot query: {str(e)}")
        return str(e)

# Function for few-shot prompting (using multiple frames)
def few_shot_query(trial_path, num_frames=5):
    # Get all frames
    frames = sorted([f for f in os.listdir(trial_path) if f.startswith("frame_")])
    if not frames:
        return None
    
    # Select frames (first, some middle frames, and last)
    if len(frames) <= num_frames:
        selected_frames = frames
    else:
        # Always include first and last frame
        step = max(1, (len(frames) - 1) // (num_frames - 1))
        indices = [0] + [min(i * step, len(frames) - 1) for i in range(1, num_frames - 1)] + [len(frames) - 1]
        selected_frames = [frames[i] for i in indices]
    
    # Create content array with all frames
    content = [{"type": "text", "text": "Here are several frames from a physics simulation. Can the red ball get into the green goal?"}]
    
    # Add all selected frames to the content
    for frame in selected_frames:
        frame_path = os.path.join(trial_path, frame)
        base64_image = encode_image(frame_path)
        if not base64_image:
            continue  # Skip this frame if encoding failed
        image_url = f"data:image/png;base64,{base64_image}"
        content.append({"type": "image_url", "image_url": {"url": image_url}})
    
    # If no images were successfully encoded, return error
    if len(content) <= 1:
        return "Error: No images could be encoded"
    
    try:
        completion = client.chat.completions.create(
            model="qwen-vl-plus",
            messages=[{"role": "user", "content": content}]
        )
        return completion.model_dump_json()
    except Exception as e:
        print(f"Error in few-shot query: {str(e)}")
        return str(e)

# Function to process a single trial
def process_trial(trial_path, output_dir):
    # Extract subject and attempt info from the path
    path_parts = Path(trial_path).parts
    subject = path_parts[-2]  # e.g., "Subj_1"
    attempt = path_parts[-1]  # e.g., "Falling_A_attempt_0_obj3_True"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a unique filename for results
    result_file = os.path.join(output_dir, f"{subject}_{attempt}.json")
    
    # Skip if already processed
    if os.path.exists(result_file):
        print(f"Skipping {result_file} (already exists)")
        return
    
    print(f"Processing {subject}/{attempt}...")
    
    # Run both 0-shot and few-shot queries
    zero_shot_result = zero_shot_query(trial_path)
    time.sleep(1)  # Brief pause to avoid API rate limits
    few_shot_result = few_shot_query(trial_path)
    
    # Save results
    results = {
        "subject": subject,
        "attempt": attempt,
        "zero_shot": zero_shot_result,
        "few_shot": few_shot_result
    }
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {result_file}")

# Main function to process all trials
def main():
    # Create output directory
    output_dir = "qwen_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if the trials directory exists
    if not os.path.exists(TRIALS_DIR):
        print(f"Error: Trials directory '{TRIALS_DIR}' not found.")
        sys.exit(1)
    
    # Get all subject directories
    trials_path = Path(TRIALS_DIR)
    subject_dirs = [d for d in trials_path.iterdir() if d.is_dir()]
    
    if not subject_dirs:
        print(f"Error: No subject directories found in '{TRIALS_DIR}'.")
        sys.exit(1)
    
    # Optional: Limit the number of subjects for testing
    #subject_dirs = subject_dirs[:2]
    
    # Process each subject
    for subject_dir in subject_dirs:
        print(f"Processing subject: {subject_dir.name}")
        
        # Get all attempt directories for this subject
        attempt_dirs = [d for d in subject_dir.iterdir() if d.is_dir()]
        
        # Optional: Limit the number of attempts per subject for testing
        #attempt_dirs = attempt_dirs[:3]
        
        # Process each attempt
        for attempt_dir in attempt_dirs:
            process_trial(attempt_dir, output_dir)

if __name__ == "__main__":
    # No need to check for environment variable since we're using a hardcoded key
    main()

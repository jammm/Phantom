import gradio as gr
import subprocess
import glob
import os
import threading
import time
from gradio_log import Log

log_file = "generation_log.txt"
rocm_smi_log = "rocm_smi.txt"

def get_example_images():
    """Get all PNG files from the examples folder"""
    example_dir = "examples"
    if os.path.exists(example_dir):
        png_files = glob.glob(os.path.join(example_dir, "*.png"))
        return sorted(png_files)
    return []

def start_rocm_smi_monitoring():
    """Start rocm-smi monitoring in the background"""
    def run_rocm_smi():
        try:
            # Create/truncate the rocm-smi log file
            with open(rocm_smi_log, "w") as f:
                pass
            
            # Continuously run rocm-smi with 1 second intervals
            while True:
                try:
                    # Run rocm-smi command
                    result = subprocess.run(
                        ["rocm-smi"],
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    
                    # Write output to file
                    with open(rocm_smi_log, "w") as f:
                        f.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                        f.write(result.stdout)
                        if result.stderr:
                            f.write(f"STDERR: {result.stderr}")
                        f.write("\n")
                        f.flush()
                    
                    # Sleep for 1 second before next run
                    time.sleep(1)
                    
                except subprocess.TimeoutExpired:
                    with open(rocm_smi_log, "a") as f:
                        f.write(f"rocm-smi command timed out at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                        f.flush()
                    time.sleep(1)
                    
        except Exception as e:
            with open(rocm_smi_log, "a") as f:
                f.write(f"Error running rocm-smi monitoring: {e}\n")
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=run_rocm_smi, daemon=True)
    monitor_thread.start()

def generate_video(prompt, width, height, selected_examples, uploaded_images, frames, fps, ckpt_dir, phantom_ckpt, ulysses_size, ring_size, num_gpus):
    # Validate inputs
    if not prompt.strip():
        return "❌ Please enter a prompt!", None
    
    # Collect all selected images
    ref_image_paths = []
    
    # Add selected example images
    if selected_examples:
        ref_image_paths.extend(selected_examples)
    
    # Add uploaded images
    if uploaded_images:
        for uploaded_file in uploaded_images:
            ref_image_paths.append(uploaded_file.name)
    
    if not ref_image_paths:
        return "❌ Please select at least one reference image!", None
    
    if width <= 0 or height <= 0:
        return "❌ Width and height must be positive!", None
    if frames <= 0 or fps <= 0:
        return "❌ Frames and FPS must be positive!", None

    # Convert image paths to comma-separated string
    ref_images_str = ",".join(ref_image_paths)

    # Build the command for video generation
    command = [
        "torchrun", f"--nproc_per_node={num_gpus}", "generate.py",
        "--task", "s2v-14B",
        "--size", f"{width}*{height}",
        "--frame_num", str(frames),
        "--sample_fps", str(fps),
        "--ckpt_dir", ckpt_dir,
        "--phantom_ckpt", phantom_ckpt,
        "--ref_image", ref_images_str,
        "--dit_fsdp",
        "--t5_fsdp",
        "--ulysses_size", str(ulysses_size),
        "--ring_size", str(ring_size),
        "--prompt", prompt
    ]
    
    try:
        with open(log_file, 'a') as f:
            f.write(f"Starting video generation...\n")
            f.write(f"Command: {' '.join(command)}\n")
            f.flush()
        
        # Run the command and redirect output to log file
        with open(log_file, 'a') as f:
            process = subprocess.Popen(command, cwd=".", stdout=f, stderr=subprocess.STDOUT, text=True)
            process.wait()
        
        if process.returncode != 0:
            with open(log_file, 'a') as f:
                f.write(f"\nVideo generation failed with return code: {process.returncode}\n")
            return f"❌ Video generation failed with return code: {process.returncode}", None
        else:
            with open(log_file, 'a') as f:
                f.write("\nVideo generation completed!\n")
            
            # Look for generated video files
            video_patterns = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.webm"]
            video_files = []
            for pattern in video_patterns:
                video_files.extend(glob.glob(pattern))
            # Also check common output directories
            output_dirs = ["./outputs", "./results", "./generated", "."]
            for output_dir in output_dirs:
                if os.path.exists(output_dir):
                    for pattern in video_patterns:
                        video_files.extend(glob.glob(os.path.join(output_dir, pattern)))
            # Remove duplicates
            video_files = list(set(video_files))
            if video_files:
                video_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                latest_video = video_files[0]
                with open(log_file, 'a') as f:
                    f.write(f"Generated video: {latest_video}\n")
                return f"✅ Video generation completed successfully!\n📹 Generated video: {latest_video}", latest_video
            else:
                with open(log_file, 'a') as f:
                    f.write("No video files found. Please check the output directory manually.\n")
                return "⚠️ No video files found. Please check the output directory manually.", None
    except Exception as e:
        with open(log_file, 'a') as f:
            f.write(f"Error running command: {e}\n")
        return f"❌ Error running command: {e}", None

# Start ROCm-SMI monitoring
start_rocm_smi_monitoring()

with gr.Blocks() as demo:
    gr.Markdown("## 🎬 Phantom Video Generation")

    with gr.Row():
        prompt = gr.Textbox(label="Prompt", placeholder="Enter your video generation prompt here...", lines=4,
                             value="A cartoon old grandfather wearing a yellow hat, a yellow top and brown suspenders is holding a blue steaming coffee cup in a fresh cartoon-style cafe decorated with pink and blue tables and chairs, colorful chandeliers and colorful balls. The picture style is cartoony and fresh.")
    
    with gr.Row():
        width = gr.Number(label="Width", value=832)
        height = gr.Number(label="Height", value=480)
    
    gr.Markdown("### Reference Images")
    
    # Get example images
    example_images = get_example_images()
    
    # Pre-selected images
    preselected_images = ["examples/ref14.png", "examples/ref15.png", "examples/ref16.png"]
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Select Example Images (click to select/deselect):**")
            selected_examples = gr.Gallery(
                value=preselected_images,
                label="Selected Example Images",
                show_label=True,
                elem_id="selected_gallery",
                columns=4,
                rows=1,
                object_fit="contain",
                height="auto",
                selected_index=None,
                allow_preview=False,
                interactive=True
            )
            
            # Display all available example images
            if example_images:
                example_gallery = gr.Gallery(
                    value=example_images,
                    label="Available Example Images (click to add to selection)",
                    show_label=True,
                    elem_id="example_gallery",
                    columns=4,
                    rows=2,
                    object_fit="contain",
                    height="auto",
                    selected_index=None,
                    allow_preview=False,
                    interactive=True
                )
        
        with gr.Column():
            gr.Markdown("**Or Upload Your Own Images:**")
            uploaded_images = gr.File(
                label="Upload Images",
                file_count="multiple",
                file_types=["image"],
                interactive=True
            )
    
    # Hidden component to store selected example paths
    selected_example_paths = gr.State(value=preselected_images)
    
    def update_selection(evt: gr.SelectData, current_paths):
        """Handle clicking on example gallery to add/remove from selection"""
        clicked_image = example_images[evt.index]
        
        if clicked_image in current_paths:
            # Remove from selection
            current_paths.remove(clicked_image)
        else:
            # Add to selection
            current_paths.append(clicked_image)
        
        return current_paths, current_paths
    
    def remove_from_selection(evt: gr.SelectData, current_paths):
        """Handle clicking on selected gallery to remove from selection"""
        if evt.index < len(current_paths):
            # Remove the clicked image from selection
            current_paths.pop(evt.index)
        
        return current_paths, current_paths
    
    def handle_uploaded_images(uploaded_files, current_paths):
        """Handle uploaded images by adding them to the selection"""
        if uploaded_files:
            # Add uploaded file paths to current selection
            for file in uploaded_files:
                if file.name not in current_paths:
                    current_paths.append(file.name)
        
        return current_paths, current_paths
    
    # Connect gallery click to selection update
    if example_images:
        example_gallery.select(
            fn=update_selection,
            inputs=[selected_example_paths],
            outputs=[selected_example_paths, selected_examples]
        )
    
    # Connect selected gallery click to removal
    selected_examples.select(
        fn=remove_from_selection,
        inputs=[selected_example_paths],
        outputs=[selected_example_paths, selected_examples]
    )
    
    # Connect uploaded images to selection
    uploaded_images.change(
        fn=handle_uploaded_images,
        inputs=[uploaded_images, selected_example_paths],
        outputs=[selected_example_paths, selected_examples]
    )
    
    with gr.Row():
        frames = gr.Number(label="Frames", value=121)
        fps = gr.Number(label="FPS", value=24)
    
    gr.Markdown("### Advanced Settings")
    with gr.Row():
        ckpt_dir = gr.Textbox(label="Checkpoint Dir", value="./Wan2.1-T2V-1.3B")
        phantom_ckpt = gr.Textbox(label="Phantom Ckpt", value="./Phantom-Wan-Models")
    
    with gr.Row():
        ulysses_size = gr.Number(label="Ulysses Size", value=8)
        ring_size = gr.Number(label="Ring Size", value=1)
        num_gpus = gr.Number(label="Num GPUs", value=8)
    
    generate_button = gr.Button("🎬 Generate Video", variant="primary")
    output_log = gr.Textbox(label="Output Log", interactive=False)
    
    # Create and truncate log files
    with open(log_file, "w") as f:
        pass
    
    # Add gradio-log component for real-time logging
    log_component = Log(log_file, dark=True, xterm_font_size=10, show_label=False,height=300)
    
    # Add gradio-log component for ROCm-SMI monitoring
    gr.Markdown("### GPU Monitoring (ROCm-SMI)")
    rocm_smi_component = Log(rocm_smi_log, dark=True, xterm_font_size=10, show_label=False, height=260, reload_every_time=True)
    
    video_display = gr.Video(label="Generated Video", format="mp4")
    
    generate_button.click(
        fn=generate_video,
        inputs=[prompt, width, height, selected_example_paths, uploaded_images, frames, fps, ckpt_dir, phantom_ckpt, ulysses_size, ring_size, num_gpus],
        outputs=[output_log, video_display]
    )

demo.launch(share=True)

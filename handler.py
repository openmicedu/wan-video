import os
import time
import uuid
import tempfile
import requests
import logging
import asyncio
import json
import sys
from PIL import Image
import torch
import runpod

# Import Wan 2.1 components
import wan
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.utils.utils import cache_video

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model cache
model_cache = {}

def download_image(url, timeout=30):
    """Download image from URL and return PIL Image"""
    try:
        logger.info(f"Downloading image from: {url}")
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        # Open as PIL Image
        image = Image.open(tmp_path).convert("RGB")
        os.unlink(tmp_path)  # Clean up temp file
        
        logger.info(f"Successfully downloaded image: {image.size}")
        return image
        
    except Exception as e:
        logger.error(f"Failed to download image: {str(e)}")
        raise RuntimeError(f"Failed to download image from {url}: {str(e)}")

def _upload(path):
    """Upload video to catbox.moe"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36'
    }
    
    with open(path, "rb") as f:
        r = requests.post("https://catbox.moe/user/api.php", 
                        files={"fileToUpload": f},
                        data={"reqtype": "fileupload"},
                        headers=headers,
                        timeout=60)
    r.raise_for_status()
    return r.text.strip()

def load_model():
    """Load and cache the Wan I2V model"""
    global model_cache
    
    if 'wan_i2v' not in model_cache:
        logger.info("Loading Wan 2.1 I2V model...")
        
        # Get model path from environment
        model_path = os.environ.get('MODEL_PATH', '/workspace/Wan2.1/models/Wan2.1-I2V-14B-720P')
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model path not found: {model_path}")
        
        # Load model configuration - EXACT from generate.py
        cfg = WAN_CONFIGS['i2v-14B']
        
        # Initialize model - EXACT parameters from generate.py
        model = wan.WanI2V(
            config=cfg,
            checkpoint_dir=model_path,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
        )
        
        model_cache['wan_i2v'] = model
        model_cache['config'] = cfg
        logger.info("Model loaded successfully")
    
    return model_cache['wan_i2v'], model_cache['config']

async def async_generator_handler(job):
    """RunPod async generator handler for video generation"""
    job_input = job.get("input", {})
    job_id = job.get('id', 'unknown')
    start_time = time.time()
    
    try:
        # Validate required inputs
        image_url = job_input.get("image_url")
        prompt = job_input.get("prompt")
        
        if not image_url or not prompt:
            yield {
                "error": "Missing required parameters: image_url and prompt",
                "error_code": "INVALID_INPUT",
                "status": "failed"
            }
            return
        
        logger.info(f"Starting video generation job {job_id} with prompt: '{prompt}'")
        
        # Extract optional parameters with defaults
        negative = job_input.get("negative", "")
        seed = job_input.get("seed", -1)
        resolution = job_input.get("resolution", "720p")
        
        # Map resolution to parameters - EXACT from generate.py logic
        if resolution == "720p":
            size_key = "1280*720"
            max_area = MAX_AREA_CONFIGS[size_key]  # Use exact config
            shift = 5.0
        elif resolution == "480p":
            size_key = "832*480"
            max_area = MAX_AREA_CONFIGS[size_key]  # Use exact config
            shift = 3.0
        else:
            yield {
                "error": f"Unsupported resolution: {resolution}. Use '720p' or '480p'",
                "error_code": "INVALID_INPUT",
                "status": "failed"
            }
            return
        
        # Step 1: Download input image
        yield {
            "status": "downloading",
            "progress": 10,
            "message": "Downloading input image...",
            "timestamp": time.time() - start_time
        }
        
        image = download_image(image_url)
        
        # Step 2: Load model
        yield {
            "status": "loading",
            "progress": 20,
            "message": "Loading model...",
            "timestamp": time.time() - start_time
        }
        
        model, cfg = load_model()
        
        # Step 3: Generate video
        yield {
            "status": "generating",
            "progress": 30,
            "message": "Starting video generation...",
            "timestamp": time.time() - start_time
        }
        
        logger.info("Generating video...")
        # EXACT method call from generate.py
        video = model.generate(
            prompt,                    # First positional - EXACT from generate.py
            image,                     # Second positional - EXACT from generate.py
            max_area=max_area,         # EXACT from generate.py
            frame_num=81,              # EXACT default from generate.py (4n+1 format)
            shift=shift,               # EXACT shift parameter
            sample_solver='unipc',     # EXACT default from generate.py
            sampling_steps=40,         # EXACT default for i2v tasks
            guide_scale=5.0,           # EXACT default
            n_prompt=negative if negative else cfg.sample_neg_prompt,
            seed=seed if seed >= 0 else -1,        # EXACT logic from generate.py
            offload_model=True         # EXACT from generate.py
        )
        
        # Step 4: Save video
        yield {
            "status": "saving",
            "progress": 80,
            "message": "Saving video...",
            "timestamp": time.time() - start_time
        }
        
        # Generate unique filename
        output_filename = f"wan21_i2v_{uuid.uuid4().hex[:8]}.mp4"
        output_path = f"/tmp/{output_filename}"
        
        logger.info(f"Saving video to: {output_path}")
        # EXACT cache_video call from generate.py
        cache_video(
            tensor=video[None],        # EXACT format
            save_file=output_path,
            fps=cfg.sample_fps,        # Use config fps (16 from gradio examples)
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        # Step 5: Upload to file.io
        yield {
            "status": "uploading",
            "progress": 90,
            "message": "Uploading video...",
            "timestamp": time.time() - start_time
        }
        
        video_url = _upload(output_path)
        
        # Clean up temporary file
        if os.path.exists(output_path):
            os.unlink(output_path)
        
        processing_time = time.time() - start_time
        
        # Final result
        yield {
            "video_url": video_url,
            "duration": 5.0,
            "resolution": resolution,
            "seed": seed,
            "processing_time": round(processing_time, 2),
            "status": "completed",
            "message": f"Video generation completed in {processing_time:.2f}s",
            "timestamp": processing_time
        }
        
        logger.info(f"Job {job_id} completed successfully in {processing_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Error during video generation for job {job_id}: {str(e)}")
        
        # Determine error type
        if "download" in str(e).lower():
            error_code = "DOWNLOAD_ERROR"
        elif "model" in str(e).lower():
            error_code = "MODEL_ERROR"
        elif "memory" in str(e).lower() or "cuda" in str(e).lower():
            error_code = "MEMORY_ERROR"
        elif "upload" in str(e).lower():
            error_code = "UPLOAD_ERROR"
        else:
            error_code = "GENERATION_ERROR"
        
        yield {
            "error": str(e),
            "error_code": error_code,
            "status": "failed",
            "processing_time": time.time() - start_time,
            "timestamp": time.time() - start_time
        }

async def run_test(job):
    """Test function for local development"""
    async for item in async_generator_handler(job):
        print(json.dumps(item, indent=2))

if __name__ == "__main__":
    if "--test_input" in sys.argv:
        test_input_index = sys.argv.index("--test_input")
        if test_input_index + 1 < len(sys.argv):
            test_input_json = sys.argv[test_input_index + 1]
            try:
                job = json.loads(test_input_json)
                asyncio.run(run_test(job))
            except json.JSONDecodeError:
                print("Error: Invalid JSON in test_input")
        else:
            print("Error: --test_input requires a JSON string argument")
    else:
        # Initialize RunPod serverless with async generator
        runpod.serverless.start({
            "handler": async_generator_handler,
            "return_aggregate_stream": True  # Makes results available via /run endpoint
        })
import torch
import json
import os
import logging
import random
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

# Third-party Libs
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif

# --- 1. INSTRUMENTATION ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [SMA-01 CORE] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress minor warnings for cleaner logs
warnings.filterwarnings("ignore", category=FutureWarning)

# --- 2. CONFIGURATION ---
@dataclass(frozen=True)
class EngineConfig:
    """Immutable Configuration for JEPA-Synthetic-Lab."""
    base_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"
    prompts_path: str = "prompts.json"
    output_dir: str = "renders"
    default_negative: str = "bad quality, worse quality, low resolution, watermark, text, error, jpeg artifacts"

# --- 3. HARDWARE ABSTRACTION LAYER (HAL) ---
class HardwareProfile:
    """Detects and encapsulates hardware capabilities."""
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.vram_gb = 0.0
        self.name = "CPU"
        
        if self.device == "cuda":
            props = torch.cuda.get_device_properties(0)
            self.vram_gb = props.total_memory / 1e9
            self.name = torch.cuda.get_device_name(0)
            
            # Capability 8.0+ means Ampere (A100, A6000, 3090, 4090)
            self.compute_capability = props.major + (props.minor / 10)
        else:
            self.compute_capability = 0.0

    def is_high_end(self) -> bool:
        """Returns True for A100, A6000, H100 (VRAM > 20GB)."""
        return self.vram_gb > 22.0

    def is_ampere_or_newer(self) -> bool:
        """Returns True if GPU supports modern SDPA natively."""
        return self.compute_capability >= 8.0

# --- 4. STRATEGY PATTERN ---
class OptimizationStrategy(ABC):
    @abstractmethod
    def apply(self, pipe: AnimateDiffPipeline, profile: HardwareProfile):
        pass

    @abstractmethod
    def get_resolution_constraints(self) -> Dict[str, int]:
        pass

class HighPerformanceStrategy(OptimizationStrategy):
    """
    Strategy for A100/H100/A6000.
    Prioritizes Native PyTorch SDPA (Scaled Dot Product Attention).
    Avoids xFormers on Ampere to prevent kernel deadlocks.
    """
    def apply(self, pipe: AnimateDiffPipeline, profile: HardwareProfile):
        logger.info(f"🚀 Strategy: HIGH PERFORMANCE ({profile.name}).")
        
        # Modern PyTorch (2.0+) on Ampere GPUs uses SDPA automatically.
        # It is faster and more stable than xformers for this architecture.
        # We explicitly DO NOT enable xformers here to avoid 'freeze' issues.
        logger.info("⚡ Optimization: Native PyTorch 2.0 SDPA Active.")
        
        # No CPU offload needed for 40GB+ VRAM
        
    def get_resolution_constraints(self) -> Dict[str, int]:
        return {"max_width": 1024, "max_height": 1024}

class SurvivalStrategy(OptimizationStrategy):
    """
    Strategy for T4/L4/Consumer GPUs (< 16GB VRAM).
    Prioritizes memory safety over speed.
    """
    def apply(self, pipe: AnimateDiffPipeline, profile: HardwareProfile):
        logger.info(f"🛡️ Strategy: SURVIVAL MODE ({profile.name}).")
        
        # 1. Try Memory Efficient Attention (xFormers)
        # On older cards (T4), xformers is a lifesaver.
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("⚡ xFormers: ENABLED (Memory optimization).")
        except Exception:
            logger.warning("⚠️ xFormers not found/failed. Fallback to slicing.")

        # 2. Aggressive VRAM Management
        # Only offload if absolutely necessary (T4 usually needs it for 512x768)
        if profile.vram_gb < 15.0:
            logger.info("📉 Memory: Enabling CPU Offload.")
            pipe.enable_model_cpu_offload()
        
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()

    def get_resolution_constraints(self) -> Dict[str, int]:
        return {"max_width": 512, "max_height": 512}

def get_strategy(profile: HardwareProfile) -> OptimizationStrategy:
    if profile.is_high_end():
        return HighPerformanceStrategy()
    return SurvivalStrategy()

# --- 5. CORE ENGINE ---
class LatentMotionEngine:
    def __init__(self):
        logger.info("⚙️ Initializing SMA-01 Core Engine v2.0...")
        
        self.config = EngineConfig()
        self.profile = HardwareProfile()
        self.token = self._resolve_token()
        
        # Load Data
        self.prompts_db = self._load_prompts()
        
        # Select Strategy
        self.strategy = get_strategy(self.profile)
        
        # Build Pipe
        self.pipe = self._build_pipeline()

    def _resolve_token(self) -> Optional[str]:
        token = os.getenv("HF_TOKEN")
        if not token:
            logger.warning("⚠️ HF_TOKEN not found. Public models only.")
        return token

    def _load_prompts(self) -> Dict[str, Any]:
        if not os.path.exists(self.config.prompts_path):
            logger.error(f"❌ Config {self.config.prompts_path} missing.")
            return {"scenes": {}}
        with open(self.config.prompts_path, 'r') as f:
            return json.load(f)

    def _build_pipeline(self) -> AnimateDiffPipeline:
        dtype = torch.float16 if self.profile.device == "cuda" else torch.float32
        
        logger.info("🔌 Mounting Neural Adapters...")
        adapter = MotionAdapter.from_pretrained(
            self.config.motion_adapter_id,
            torch_dtype=dtype,
            token=self.token
        )

        pipe = AnimateDiffPipeline.from_pretrained(
            self.config.base_model_id,
            motion_adapter=adapter,
            torch_dtype=dtype,
            token=self.token
        )

        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            timestep_spacing="trailing", 
            beta_schedule="linear"
        )

        # Apply Hardware Strategy
        self.strategy.apply(pipe, self.profile)
        
        return pipe

    def _enforce_limits(self, width: int, height: int) -> Tuple[int, int]:
        """Dynamic Resolution Scaling based on Hardware Constraints."""
        limits = self.strategy.get_resolution_constraints()
        
        new_w = min(width, limits["max_width"])
        new_h = min(height, limits["max_height"])
        
        if new_w != width or new_h != height:
            logger.warning(f"⚠️ Resolution Override: {width}x{height} -> {new_w}x{new_h} (VRAM Constraint)")
        
        return new_w, new_h

    def render(self, scene_name: str, forced_seed: int = -1):
        if scene_name not in self.prompts_db.get('scenes', {}):
            return

        scene_data = self.prompts_db['scenes'][scene_name]
        sys_config = self.prompts_db.get('system_settings', {})

        # 1. Resolution Safety
        width, height = self._enforce_limits(
            sys_config.get('width', 512), 
            sys_config.get('height', 512)
        )
        
        # 2. Physics & Compute
        motion_scale = scene_data.get('motion_scale', 1.0)
        base_steps = sys_config.get('base_steps', 25)
        
        # 3. Seed Integrity (CPU Generator for Cross-Platform Consistency)
        json_seed = scene_data.get('seed', -1)
        seed = forced_seed if forced_seed != -1 else (json_seed if json_seed != -1 else random.randint(0, 2**32-1))
        generator = torch.Generator("cpu").manual_seed(seed)

        logger.info(f"🎬 Action: {scene_name} | Seed: {seed} | Motion: {motion_scale}")

        # 4. Execution
        output = self.pipe(
            prompt=scene_data['positive'],
            negative_prompt=scene_data.get('negative', self.config.default_negative),
            num_frames=16,
            guidance_scale=7.5,
            num_inference_steps=base_steps,
            generator=generator,
            width=width,
            height=height,
        )

        # 5. Export
        os.makedirs(self.config.output_dir, exist_ok=True)
        filename = f"{self.config.output_dir}/{scene_name}_{seed}.gif"
        export_to_gif(output.frames[0], filename)
        logger.info(f"💾 Artifact: {filename}")

# --- 6. ENTRY POINT ---
if __name__ == "__main__":
    try:
        engine = LatentMotionEngine()
        scenes = engine.prompts_db.get('scenes', {})
        
        if not scenes:
            logger.warning("No scenes found in prompts.json")
        
        for scene_key in scenes.keys():
            engine.render(scene_key)
            
        print("\n✅ Batch Processing Complete.")
    except Exception as e:
        logger.critical(f"❌ System Failure: {e}")
        raise
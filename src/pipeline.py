import torch
import json
import os
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Union

# Сторонние библиотеки (International Standards)
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from google.colab import userdata

# --- 1. SETUP LOGGER (Бортовой самописец) ---
# Мы настраиваем формат: ВРЕМЯ - МОДУЛЬ - УРОВЕНЬ - СООБЩЕНИЕ
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [JEPA-CORE] - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURATION DATA CLASS (Паспорт) ---
@dataclass
class EngineConfig:
    """
    Неизменяемая конфигурация.
    Хранит ссылки на модели и пути.
    """
    base_model_id: str = "SG161222/Realistic_Vision_V5.1_noVAE"
    motion_adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2"
    prompts_path: str = "configs/prompts.json"
    output_dir: str = "renders"

# --- 3. STRATEGY PATTERN (Мозги управления железом) ---

class OptimizationStrategy(ABC):
    """
    Абстрактный контракт.
    Любая стратегия обязана иметь метод apply.
    """
    @abstractmethod
    def apply(self, pipe: AnimateDiffPipeline):
        pass

class HighPerformanceStrategy(OptimizationStrategy):
    """Стратегия для A100 / H100 (>24GB VRAM)."""
    def apply(self, pipe: AnimateDiffPipeline):
        logger.info("🚀 Strategy: HIGH PERFORMANCE. All systems in VRAM.")
        # Включаем только нарезку VAE, остальное держим в памяти для скорости
        pipe.enable_vae_slicing()

class SurvivalStrategy(OptimizationStrategy):
    """Стратегия для T4 / Consumer GPU (<16GB VRAM)."""
    def apply(self, pipe: AnimateDiffPipeline):
        logger.info("🛡️ Strategy: SURVIVAL MODE. Aggressive offloading enabled.")
        # Выгружаем веса модели в RAM, когда они не нужны
        pipe.enable_model_cpu_offload()
        # Режем VAE декодинг на куски (спасает от OOM)
        pipe.enable_vae_slicing()
        # Тайлинг (работа с картинкой по квадратам)
        pipe.enable_vae_tiling()

def detect_strategy() -> OptimizationStrategy:
    """Фабрика стратегий: сама щупает железо и выдает нужный алгоритм."""
    if not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not found! CPU mode is not supported efficiently.")
        return SurvivalStrategy() # Возвращаем самый легкий режим

    # Получаем память в Гигабайтах
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    device_name = torch.cuda.get_device_name(0)
    
    logger.info(f"🖥️ Hardware Detected: {device_name} ({vram_gb:.1f} GB)")

    if vram_gb > 20.0:
        return HighPerformanceStrategy()
    else:
        return SurvivalStrategy()

# --- 4. MAIN ENGINE (Основной класс) ---

class LatentMotionEngine:
    """
    SMA-01 Core Engine.
    Orchestrates the loading, optimization, and rendering process.
    """
    
    def __init__(self, hf_token: Optional[str] = None):
        logger.info("⚙️ Initializing Latent Motion Engine...")
        
        self.config = EngineConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Авторизация
        self.token = hf_token or self._fetch_token()
        
        # 1. Загрузка "Мозга" (Промпты)
        self.prompts_db = self._load_prompts()
        
        # 2. Выбор стратегии (без загрузки модели)
        self.strategy = detect_strategy()
        
        # 3. Сборка Пайплайна (Тяжелая операция)
        self.pipe = self._build_pipeline()

    def _fetch_token(self) -> Optional[str]:
        try:
            return userdata.get('HF_TOKEN')
        except Exception:
            logger.warning("⚠️ HF_TOKEN not found in Secrets. Using public access.")
            return None

    def _load_prompts(self) -> Dict:
        if not os.path.exists(self.config.prompts_path):
            # В продакшене здесь лучше кинуть ошибку, но для Colab создадим заглушку
            logger.warning(f"❌ Config {self.config.prompts_path} not found.")
            return {"scenes": {}}
            
        with open(self.config.prompts_path, 'r') as f:
            data = json.load(f)
        return data

    def _build_pipeline(self) -> AnimateDiffPipeline:
        logger.info("🔌 Loading Neural Network weights...")
        
        # Адаптер движения (Motion Module)
        adapter = MotionAdapter.from_pretrained(
            self.config.motion_adapter_id,
            torch_dtype=self.dtype,
            token=self.token
        )

        # Основной пайплайн
        pipe = AnimateDiffPipeline.from_pretrained(
            self.config.base_model_id,
            motion_adapter=adapter,
            torch_dtype=self.dtype,
            token=self.token
        )

        # Настройка планировщика (Scheduler)
        pipe.scheduler = EulerDiscreteScheduler.from_config(
            pipe.scheduler.config, 
            timestep_spacing="trailing", 
            beta_schedule="linear"
        )

        # === ПРИМЕНЕНИЕ СТРАТЕГИИ ===
        # Движок не знает деталей. Он просто говорит: "Оптимизируй себя!"
        self.strategy.apply(pipe)
        
        return pipe

    def render(self, 
               scene_name: str, 
               num_frames: int = 16, 
               seed: int = -1) -> str:
        """
        Основной метод генерации.
        Args:
            scene_name: Ключ из JSON конфига.
            seed: Число для генератора. -1 для случайного.
        """
        # Валидация
        if scene_name not in self.prompts_db.get('scenes', {}):
            raise ValueError(f"❌ Scene '{scene_name}' not found in DB.")
            
        scene_data = self.prompts_db['scenes'][scene_name]
        
        # Управление Случайностью (Reproducibility)
        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"🎲 Seed Auto-Generated: {seed}")
        else:
            logger.info(f"🔒 Using Fixed Seed: {seed}")
            
        generator = torch.Generator(self.device).manual_seed(seed)
        
        # Процесс Инференса
        logger.info(f"🎬 Rendering: {scene_data.get('description', 'Unknown')}")
        
        output = self.pipe(
            prompt=scene_data['positive'],
            negative_prompt=scene_data['negative'],
            num_frames=num_frames,
            guidance_scale=7.5,
            num_inference_steps=35,
            generator=generator
        )
        
        # Экспорт
        os.makedirs(self.config.output_dir, exist_ok=True)
        filename = f"{self.config.output_dir}/{scene_name}_{seed}.gif"
        export_to_gif(output.frames[0], filename)
        
        logger.info(f"💾 Artifact saved: {filename}")
        return filename

# --- ENTRY POINT (Точка входа) ---
if __name__ == "__main__":
    # Этот блок сработает, только если запустить файл напрямую.
    # Если импортировать его в ноутбук, этот код не выполнится.
    print("\n--- JEPA/SMA-01 PIPELINE CHECK ---")
    try:
        # Тестовая инициализация (без рендера, только проверка сборки)
        engine = LatentMotionEngine()
        print("✅ System Status: ONLINE via Strategy Pattern.")
    except Exception as e:
        print(f"❌ System Status: FAILED. Error: {e}")
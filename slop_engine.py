from __future__ import annotations

import io
import os
import random
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from diffusers import (
    AnimateDiffVideoToVideoPipeline,
    AutoPipelineForImage2Image,
    DDIMScheduler,
    MotionAdapter,
)
from PIL import Image, ImageEnhance, ImageOps


@dataclass(frozen=True)
class SlopMode:
    id: str
    name: str
    description: str
    prompt: str
    negative_prompt: str
    min_strength: float
    max_strength: float
    steps: int
    guidance_scale: float
    passes: int = 1
    use_caption: bool = False
    think_absurd: bool = False


ABSURD_PREFIXES = [
    "this cursed photo actually shows a weird",
    "in a bizarre alternate universe, this is",
    "the secret conspiracy hidden here is a giant",
    "scientists discovered this creature is made of",
    "a low quality nightmare photograph of",
]

ABSURD_TWISTS = [
    "made of melted cheese and chrome, glowing red eyes, cursed jpeg, 8k",
    "surrounded by floating bananas and soviet concrete, uncanny valley, hdr",
    "wearing tiny sunglasses, eldritch horror vibes, deep fried, plastic texture",
    "inside a microwave, surreal fever dream, distorted geometry, unreal engine 5",
    "with extra arms and mysterious glowing runes, chaotic ai hallucination",
    "merged with a pigeon and a washing machine, hyperrealistic slop, 4k",
]


SLOP_MODES: dict[str, SlopMode] = {
    "total_bs": SlopMode(
        id="total_bs",
        name="Полная хуйня (Думающий бред)",
        description="Нейронка 2 раза всматривается в фото, придумывает дичь, шакалит картинку и дорисовывает бред.",
        prompt="chaotic surreal fever dream, cursed ai hallucination, hyper detailed plastic nightmare, 8k",
        negative_prompt="",
        min_strength=0.52,
        max_strength=0.86,
        steps=4,
        guidance_scale=4.0,
        passes=2,
        use_caption=True,
        think_absurd=True,
    ),
    "plastic_hdr": SlopMode(
        id="plastic_hdr",
        name="Пластиковый HDR",
        description="Глянцевая восковая кожа, кислотный контраст и пафосный 8K-глянец.",
        prompt=(
            "hyperrealistic 8k masterpiece, plastic glossy skin, intense hdr, "
            "unreal engine 5, trending on artstation, dramatic rim lighting, "
            "overly detailed, airbrushed, stock photo"
        ),
        negative_prompt="",
        min_strength=0.35,
        max_strength=0.72,
        steps=4,
        guidance_scale=1.8,
    ),
    "cursed_2022": SlopMode(
        id="cursed_2022",
        name="Галлюцинации 2022",
        description="Поплывшие лица, лишние пальцы и нейросетевой бред с высоким CFG.",
        prompt=(
            "uncanny valley ai generated photo, distorted weird details, "
            "surreal artifacts, vivid oversaturated colors, glossy nightmare, "
            "masterpiece 4k"
        ),
        negative_prompt="",
        min_strength=0.45,
        max_strength=0.84,
        steps=4,
        guidance_scale=3.4,
    ),
    "broken_phone": SlopMode(
        id="broken_phone",
        name="Испорченный телефон",
        description="Слепая нейронка BLIP угадывает, что на фото, а диффузия перерисовывает по этому бреду.",
        prompt="masterpiece, 8k, plastic texture, uncanny ai art, dramatic lighting",
        negative_prompt="",
        min_strength=0.55,
        max_strength=0.88,
        steps=4,
        guidance_scale=2.3,
        use_caption=True,
    ),
    "deep_fry": SlopMode(
        id="deep_fry",
        name="Глубокая прожарка (×3)",
        description="Тройной прогон через нейросеть с накоплением артефактов и пережаркой.",
        prompt=(
            "deep fried 8k hdr masterpiece, overly sharpened, plastic glowing texture, "
            "ai artifacts, hyper detailed, surreal"
        ),
        negative_prompt="",
        min_strength=0.32,
        max_strength=0.56,
        steps=3,
        guidance_scale=2.4,
        passes=3,
    ),
    "anime_slop": SlopMode(
        id="anime_slop",
        name="Дешёвый аниме-слоп",
        description="Кривой фильтр с огромными глазами и пластиковыми бликами.",
        prompt=(
            "cheap anime screenshot, glossy big eyes, plastic cel shading, "
            "oversaturated neon colors, ai anime filter, masterpiece"
        ),
        negative_prompt="",
        min_strength=0.42,
        max_strength=0.80,
        steps=4,
        guidance_scale=2.6,
    ),
}


@dataclass(frozen=True)
class VideoMode:
    id: str
    name: str
    description: str
    engine: str  # "morph" or "animatediff"
    default_frames: int
    default_fps: int


VIDEO_MODES: dict[str, VideoMode] = {
    "morph_brainrot": VideoMode(
        id="morph_brainrot",
        name="Эволюция бреда (Морфинг)",
        description="Быстрый морфинг на sd-turbo: картинка мутирует, а BLIP каждые 5 кадров меняет бред.",
        engine="morph",
        default_frames=20,
        default_fps=10,
    ),
    "morph_zoom": VideoMode(
        id="morph_zoom",
        name="Бесконечный зум в слоп",
        description="Плавное падение внутрь фото с нарастающими нейрогаллюцинациями.",
        engine="morph",
        default_frames=20,
        default_fps=10,
    ),
    "morph_spin": VideoMode(
        id="morph_spin",
        name="Вращение в бездну",
        description="Зум + закручивание кадра + кислотный пережар на каждом шаге.",
        engine="morph",
        default_frames=20,
        default_fps=10,
    ),
    "animatediff": VideoMode(
        id="animatediff",
        name="AnimateDiff (Видео-диффузия)",
        description="Настоящая видео-нейросеть (при первом запуске докачает ~4.5 ГБ моделей SD 1.5 + MotionAdapter).",
        engine="animatediff",
        default_frames=16,
        default_fps=8,
    ),
}


class SlopEngine:
    """Local diffusion engine tuned to generate authentic AI slop."""

    def __init__(
        self,
        sd_model_id: str | None = None,
        caption_model_id: str | None = None,
    ) -> None:
        self.sd_model_id = sd_model_id or os.getenv("SLOP_MODEL_ID", "stabilityai/sd-turbo")
        self.caption_model_id = caption_model_id or os.getenv(
            "SLOP_CAPTION_MODEL_ID", "Salesforce/blip-image-captioning-base"
        )
        self.anim_base_model_id = os.getenv(
            "SLOP_ANIM_BASE_MODEL_ID", "stable-diffusion-v1-5/stable-diffusion-v1-5"
        )
        self.anim_adapter_id = os.getenv(
            "SLOP_ANIM_ADAPTER_ID", "guoyww/animatediff-motion-adapter-v1-5-2"
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        self._lock = threading.Lock()
        self._pipe: Any = None
        self._anim_pipe: Any = None
        self._blip_processor: Any = None
        self._blip_model: Any = None

    def get_status(self) -> dict[str, Any]:
        gpu_name = None
        vram_gb = None
        if self.device == "cuda":
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        return {
            "device": self.device,
            "gpu_name": gpu_name,
            "vram_gb": vram_gb,
            "sd_model_id": self.sd_model_id,
            "sd_loaded": self._pipe is not None,
            "blip_model_id": self.caption_model_id,
            "blip_loaded": self._blip_model is not None,
        }

    def _ensure_sd_loaded(self) -> None:
        if self._pipe is not None:
            return
        pipe = AutoPipelineForImage2Image.from_pretrained(
            self.sd_model_id,
            torch_dtype=self.dtype,
            variant="fp16" if self.dtype == torch.float16 else None,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self.device)
        pipe.set_progress_bar_config(disable=True)
        if self.device == "cuda":
            pipe.enable_attention_slicing()
        self._pipe = pipe

    def _ensure_blip_loaded(self) -> None:
        if self._blip_model is not None and self._blip_processor is not None:
            return
        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained(self.caption_model_id)
        model = BlipForConditionalGeneration.from_pretrained(
            self.caption_model_id,
            torch_dtype=self.dtype,
        ).to(self.device)
        model.eval()
        self._blip_processor = processor
        self._blip_model = model

    def _caption_image(
        self,
        image: Image.Image,
        prefix: str | None = None,
        temperature: float = 1.15,
        max_tokens: int = 24,
    ) -> str:
        self._ensure_blip_loaded()
        if prefix:
            inputs = self._blip_processor(images=image, text=prefix, return_tensors="pt").to(
                self.device, dtype=self.dtype
            )
        else:
            inputs = self._blip_processor(images=image, return_tensors="pt").to(
                self.device, dtype=self.dtype
            )
        with torch.inference_mode():
            out = self._blip_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                top_k=50,
                top_p=0.92,
                temperature=temperature,
                repetition_penalty=1.15,
            )
        caption = self._blip_processor.decode(out[0], skip_special_tokens=True).strip()
        return caption or "a weird blurry object"

    @staticmethod
    def _prepare_image(image: Image.Image, max_side: int = 576) -> Image.Image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        w, h = image.size
        scale = min(max_side / max(w, h), 1.0)
        new_w = max(256, int(round(w * scale / 8) * 8))
        new_h = max(256, int(round(h * scale / 8) * 8))
        if max(new_w, new_h) > max_side:
            if new_w >= new_h:
                new_h = max(256, int(round(new_h * (max_side / new_w) / 8) * 8))
                new_w = int(round(max_side / 8) * 8)
            else:
                new_w = max(256, int(round(new_w * (max_side / new_h) / 8) * 8))
                new_h = int(round(max_side / 8) * 8)
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    @staticmethod
    def _fry_between_passes(image: Image.Image) -> Image.Image:
        image = ImageEnhance.Contrast(image).enhance(1.16)
        image = ImageEnhance.Color(image).enhance(1.22)
        image = ImageEnhance.Sharpness(image).enhance(1.4)
        return image

    @staticmethod
    def _jpeg_fry(image: Image.Image, quality: int = 16) -> Image.Image:
        w, h = image.size
        down = image.resize((max(64, int(w * 0.65)), max(64, int(h * 0.65))), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        down.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        fried = Image.open(buf).convert("RGB")
        return fried.resize((w, h), Image.Resampling.NEAREST)

    @staticmethod
    def _glitch_channels(image: Image.Image, seed: int, shift: int = 8) -> Image.Image:
        rng = np.random.default_rng(seed)
        arr = np.array(image, dtype=np.int16)
        dx_r = int(rng.integers(-shift, shift + 1))
        dx_b = int(rng.integers(-shift, shift + 1))
        arr[:, :, 0] = np.roll(arr[:, :, 0], dx_r, axis=1)
        arr[:, :, 2] = np.roll(arr[:, :, 2], dx_b, axis=1)
        noise = rng.integers(-14, 15, size=arr.shape, dtype=np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def slopify(
        self,
        image: Image.Image,
        mode_id: str = "plastic_hdr",
        slop_level: int = 65,
        custom_prompt: str = "",
        seed: int | None = None,
        jpeg_fry: bool = False,
        glitch_shift: bool = False,
    ) -> dict[str, Any]:
        mode = SLOP_MODES.get(mode_id, SLOP_MODES["plastic_hdr"])
        clamped_level = max(1, min(100, int(slop_level)))
        t = (clamped_level - 1) / 99.0
        strength = mode.min_strength + t * (mode.max_strength - mode.min_strength)

        steps = mode.steps
        while int(steps * strength) < 1:
            steps += 1

        used_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**31 - 1)
        rng = random.Random(used_seed)
        prepared = self._prepare_image(image)

        started = time.perf_counter()
        caption = None
        thoughts: list[str] = []

        with self._lock:
            self._ensure_sd_loaded()
            if self.device == "cuda" and self._pipe.device.type != "cuda":
                torch.cuda.empty_cache()
                self._pipe.to(self.device)

            current = prepared
            if jpeg_fry:
                current = self._jpeg_fry(current, quality=max(8, 28 - clamped_level // 5))
                thoughts.append("Этап 0: исходник пережат шакалами (JPEG + pixelation)")
            if glitch_shift:
                current = self._glitch_channels(current, seed=used_seed, shift=max(4, clamped_level // 8))
                thoughts.append("Этап 0: сдвинуты RGB-каналы и добавлен цифровой шум")

            prompt = mode.prompt
            if mode.think_absurd:
                prefix_1 = rng.choice(ABSURD_PREFIXES)
                thought_1 = self._caption_image(current, prefix=prefix_1, temperature=1.42, max_tokens=28)
                twist = rng.choice(ABSURD_TWISTS)
                thoughts.append(f"Мысль #1 (взгляд на фото): «{thought_1}»")
                thoughts.append(f"Придумка #1 (добавка бреда): «{twist}»")
                caption = thought_1
                prompt = f"{thought_1}, {twist}, {mode.prompt}"
            elif mode.use_caption:
                caption = self._caption_image(current, temperature=1.22, max_tokens=22)
                thoughts.append(f"BLIP увидел: «{caption}»")
                prompt = f"{caption}, {prompt}"

            if custom_prompt.strip():
                prompt = f"{custom_prompt.strip()}, {prompt}"

            for pass_idx in range(mode.passes):
                pass_prompt = prompt
                if mode.think_absurd and pass_idx > 0:
                    current = self._fry_between_passes(current)
                    current = self._jpeg_fry(current, quality=18)
                    current = self._glitch_channels(current, seed=used_seed + pass_idx, shift=6)
                    thought_2 = self._caption_image(
                        current,
                        prefix="what nightmare creature is this",
                        temperature=1.5,
                        max_tokens=26,
                    )
                    twist_2 = rng.choice(ABSURD_TWISTS)
                    thoughts.append(f"Мысль #2 (осмысление каши): «{thought_2}» + «{twist_2}»")
                    pass_prompt = f"{thought_2}, {twist_2}, {mode.prompt}"
                    if custom_prompt.strip():
                        pass_prompt = f"{custom_prompt.strip()}, {pass_prompt}"

                generator = torch.Generator(device=self.device).manual_seed(used_seed + pass_idx)
                with torch.inference_mode():
                    result = self._pipe(
                        prompt=pass_prompt,
                        image=current,
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=mode.guidance_scale,
                        generator=generator,
                    ).images[0]
                current = result
                if pass_idx < mode.passes - 1 and not mode.think_absurd:
                    current = self._fry_between_passes(current)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "image": current,
            "mode": mode.id,
            "mode_name": mode.name,
            "slop_level": clamped_level,
            "strength": round(strength, 3),
            "steps": steps,
            "seed": used_seed,
            "caption": caption,
            "thoughts": thoughts,
            "prompt": prompt,
            "elapsed_ms": elapsed_ms,
        }

    def _ensure_animatediff_loaded(self) -> None:
        if self._anim_pipe is not None:
            return
        # Free sd-turbo VRAM if needed before loading SD 1.5 + MotionAdapter on 8GB GPU
        if self._pipe is not None and self.device == "cuda":
            self._pipe.to("cpu")
            torch.cuda.empty_cache()

        adapter = MotionAdapter.from_pretrained(self.anim_adapter_id, torch_dtype=self.dtype)
        scheduler = DDIMScheduler.from_pretrained(
            self.anim_base_model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )
        pipe = AnimateDiffVideoToVideoPipeline.from_pretrained(
            self.anim_base_model_id,
            motion_adapter=adapter,
            scheduler=scheduler,
            torch_dtype=self.dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.set_progress_bar_config(disable=True)
        if self.device == "cuda":
            pipe.vae.enable_slicing()
            pipe.enable_attention_slicing()
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cpu")
        self._anim_pipe = pipe

    @staticmethod
    def _transform_frame(
        image: Image.Image,
        zoom: float = 1.0,
        angle: float = 0.0,
        shift_x: int = 0,
        shift_y: int = 0,
    ) -> Image.Image:
        w, h = image.size
        frame = image
        if angle != 0.0:
            frame = frame.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False)
        if zoom > 1.001:
            crop_w = max(64, int(round(w / zoom)))
            crop_h = max(64, int(round(h / zoom)))
            left = max(0, min(w - crop_w, (w - crop_w) // 2 + shift_x))
            top = max(0, min(h - crop_h, (h - crop_h) // 2 + shift_y))
            frame = frame.crop((left, top, left + crop_w, top + crop_h)).resize(
                (w, h), Image.Resampling.LANCZOS
            )
        return frame

    @staticmethod
    def _encode_mp4(frames: list[Image.Image], fps: int) -> bytes:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with imageio.get_writer(
                tmp_path,
                fps=fps,
                codec="libx264",
                pixelformat="yuv420p",
                macro_block_size=1,
                quality=7,
            ) as writer:
                for frame in frames:
                    w, h = frame.size
                    even_w = w - (w % 2)
                    even_h = h - (h % 2)
                    if (even_w, even_h) != (w, h):
                        frame = frame.resize((even_w, even_h), Image.Resampling.LANCZOS)
                    writer.append_data(np.asarray(frame.convert("RGB")))
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def slopify_video(
        self,
        image: Image.Image,
        video_mode_id: str = "morph_brainrot",
        slop_level: int = 65,
        custom_prompt: str = "",
        seed: int | None = None,
        num_frames: int | None = None,
        fps: int | None = None,
        jpeg_fry: bool = False,
        glitch_shift: bool = False,
    ) -> dict[str, Any]:
        vmode = VIDEO_MODES.get(video_mode_id, VIDEO_MODES["morph_brainrot"])
        clamped_level = max(1, min(100, int(slop_level)))
        t = (clamped_level - 1) / 99.0
        frames_count = max(8, min(40, int(num_frames or vmode.default_frames)))
        out_fps = max(4, min(24, int(fps or vmode.default_fps)))

        used_seed = seed if seed is not None and seed >= 0 else random.randint(0, 2**31 - 1)
        rng = random.Random(used_seed)
        started = time.perf_counter()
        thoughts: list[str] = []

        with self._lock:
            if vmode.engine == "animatediff":
                frames_count = min(32, frames_count)
                prepared = self._prepare_image(image, max_side=448)
                if jpeg_fry:
                    prepared = self._jpeg_fry(prepared, quality=16)
                    thoughts.append("Исходник пережат шакалами перед AnimateDiff")
                if glitch_shift:
                    prepared = self._glitch_channels(prepared, seed=used_seed, shift=6)
                    thoughts.append("Добавлен глитч-сдвиг каналов")

                caption = self._caption_image(
                    prepared,
                    prefix=rng.choice(ABSURD_PREFIXES),
                    temperature=1.35,
                    max_tokens=24,
                )
                twist = rng.choice(ABSURD_TWISTS)
                thoughts.append(f"AnimateDiff идея: «{caption}» + «{twist}»")
                prompt = f"{caption}, {twist}, chaotic ai animation, hyperrealistic 8k, plastic texture"
                if custom_prompt.strip():
                    prompt = f"{custom_prompt.strip()}, {prompt}"

                self._ensure_animatediff_loaded()
                strength = round(0.48 + t * 0.34, 3)
                steps = 16
                generator = torch.Generator(device="cpu").manual_seed(used_seed)
                with torch.inference_mode():
                    output = self._anim_pipe(
                        video=[prepared] * frames_count,
                        prompt=prompt,
                        negative_prompt="blurry, static, watermark",
                        height=prepared.height,
                        width=prepared.width,
                        strength=strength,
                        num_inference_steps=steps,
                        guidance_scale=7.5,
                        decode_chunk_size=8,
                        generator=generator,
                    )
                frames = list(output.frames[0])
            else:
                self._ensure_sd_loaded()
                if self.device == "cuda" and self._pipe.device.type != "cuda":
                    self._pipe.to(self.device)

                prepared = self._prepare_image(image, max_side=512)
                current = prepared
                if jpeg_fry:
                    current = self._jpeg_fry(current, quality=16)
                    thoughts.append("Кадр 0 пережат шакалами")
                if glitch_shift:
                    current = self._glitch_channels(current, seed=used_seed, shift=6)
                    thoughts.append("Кадр 0: глитч-сдвиг каналов")

                strength = round(0.32 + t * 0.26, 3)
                steps = 3
                guidance = 3.2 if vmode.id == "morph_brainrot" else 2.2
                base_twist = rng.choice(ABSURD_TWISTS)
                caption = self._caption_image(
                    current,
                    prefix=rng.choice(ABSURD_PREFIXES),
                    temperature=1.4,
                    max_tokens=22,
                )
                thoughts.append(f"Старт (кадр 1): «{caption}» + «{base_twist}»")
                prompt = f"{caption}, {base_twist}, surreal ai slop, 8k, glossy"
                if custom_prompt.strip():
                    prompt = f"{custom_prompt.strip()}, {prompt}"

                frames = [current.copy()]
                for idx in range(1, frames_count):
                    if vmode.id == "morph_spin":
                        current = self._transform_frame(current, zoom=1.05, angle=3.8)
                        current = ImageEnhance.Color(current).enhance(1.06)
                    elif vmode.id == "morph_zoom":
                        current = self._transform_frame(current, zoom=1.055, angle=0.4)
                    else:  # morph_brainrot
                        dx = rng.randint(-6, 6)
                        dy = rng.randint(-6, 6)
                        current = self._transform_frame(
                            current, zoom=1.045, angle=rng.uniform(-2.2, 2.2), shift_x=dx, shift_y=dy
                        )
                        if idx % 5 == 0:
                            current = self._glitch_channels(current, seed=used_seed + idx, shift=5)
                            new_cap = self._caption_image(
                                current,
                                prefix=rng.choice(ABSURD_PREFIXES),
                                temperature=1.48,
                                max_tokens=22,
                            )
                            new_twist = rng.choice(ABSURD_TWISTS)
                            thoughts.append(f"Мутация (кадр {idx + 1}): «{new_cap}» + «{new_twist}»")
                            prompt = f"{new_cap}, {new_twist}, surreal ai slop, 8k"
                            if custom_prompt.strip():
                                prompt = f"{custom_prompt.strip()}, {prompt}"

                    generator = torch.Generator(device=self.device).manual_seed(used_seed + idx)
                    with torch.inference_mode():
                        current = self._pipe(
                            prompt=prompt,
                            image=current,
                            strength=strength,
                            num_inference_steps=steps,
                            guidance_scale=guidance,
                            generator=generator,
                        ).images[0]
                    frames.append(current)

            mp4_bytes = self._encode_mp4(frames, fps=out_fps)

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return {
            "video_bytes": mp4_bytes,
            "mode": vmode.id,
            "mode_name": vmode.name,
            "frames": len(frames),
            "fps": out_fps,
            "strength": strength,
            "steps": steps,
            "seed": used_seed,
            "thoughts": thoughts,
            "elapsed_ms": elapsed_ms,
        }

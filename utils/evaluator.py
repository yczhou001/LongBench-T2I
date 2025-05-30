import re
import pathlib
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_random_exponential
from utils.prompt import *
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

class GemniEvaluator:
    def __init__(self, api_keys, model="gemini-2.5-pro-exp-03-25", temperature=0.0, max_cycles=2):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.client = self._create_client(self.api_keys[self.current_key_index])
        self.model = model
        self.temperature = temperature
        self.generate_config = types.GenerateContentConfig(
            temperature=self.temperature
        )
        self.tried_attempts = 0
        self.max_attempts = len(api_keys) * max_cycles

    def _create_client(self, api_key):
        """
           Create a GenAI client with the specified API key.

           Args:
               api_key (str): Google Generative AI API key.

           Returns:
               genai.Client: Initialized GenAI client.
        """
        return genai.Client(api_key=api_key)

    def _switch_api_key(self):
        """
            Switch to the next API key in the list if quota is exceeded.
            Raises an exception if all keys are exhausted.

            Raises:
                Exception: When all API keys are used up after multiple cycles.
        """
        self.tried_attempts += 1
        if self.tried_attempts >= self.max_attempts:
            raise Exception(
                f"🚫 All API keys have been cycled through {self.max_attempts} times. No usable key remains.")

        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        print(
            f"🔁 Switching to API key (index: {self.current_key_index}) | Attempt {self.tried_attempts}/{self.max_attempts}")
        self.client = self._create_client(self.api_keys[self.current_key_index])

    def _handle_exception(self, exception):
        """
            Check if the exception is due to quota or retriable issue, and switch API key.

            Args:
                exception (Exception): The raised exception from a failed request.
        """
        exception_text = str(exception).lower()
        retriable_errors = ["resource_exhausted", "quota", "rate limit", "timeout", "temporarily unavailable"]
        if any(err in exception_text for err in retriable_errors):
            print("⚠️ API quota limit reached. Attempting to switch to the next API key...")
            self._switch_api_key()

    def _load_image_part(self, path: str, mime_type: str = "image/png"):
        """
            Load an image from disk as a Gemini-compatible binary part.

            Args:
                path (str): Path to the image file.
                mime_type (str): MIME type of the image.

            Returns:
                types.Part: Gemini content part containing image bytes.
        """
        return types.Part.from_bytes(
            data=pathlib.Path(path).read_bytes(),
            mime_type=mime_type
        )
    @retry(wait=wait_random_exponential(min=30, max=300), stop=stop_after_attempt(5))
    def evaluate_plan(self, prompt:str):
        """
            Evaluate a prompt using the Gemini model without an image.

            Args:
                prompt (str): Instruction or evaluation prompt.

            Returns:
                Tuple[str, Union[int, str]]: Model response and extracted score.

            Raises:
                Exception: On failure after retries and key cycling.
        """
        try:

            contents = [prompt]

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self.generate_config
            )
            result_text = response.text.strip()
            print("📋 Evaluation Output:\n", result_text)

            score = self._extract_score(result_text)
            return result_text,score

        except Exception as e:
            print(f"❌ Evaluation failed due to error: {e}")
            self._handle_exception(e)
            raise e

    @retry(wait=wait_random_exponential(min=30, max=300), stop=stop_after_attempt(5))
    def evaluate(self, image_path: str,prompt:str):
        """
            Evaluate an image-prompt pair using the Gemini model.

            Args:
                image_path (str): Path to the image file.
                prompt (str): Instructional prompt.

            Returns:
                Tuple[str, Union[int, str]]: Model response and extracted score.

            Raises:
                Exception: If all retries and key switches fail.
        """
        try:
            curr_img_part = self._load_image_part(image_path)

            contents = [curr_img_part,prompt]

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=self.generate_config
            )
            result_text = response.text.strip()
            print("📋 Evaluation Output:\n", result_text)

            score = self._extract_score(result_text)
            return result_text,score

        except Exception as e:
            print(f"❌ Evaluation failed due to error: {e}")
            self._handle_exception(e)
            raise e

    def _extract_score(self, text):
        """
            Extract a numeric score from the LLM's response text.

            Args:
                text (str): Text response from the model.

            Returns:
                Union[int, str]: Parsed score or -1 if not found.
        """
        match = re.search(r"\*\*?Score\*\*?\s*:\s*[{(]*\s*(\d+)\s*[})]*", text, re.IGNORECASE)
        if match:
            score = match.group(1).capitalize()
            return score
        return -1
    @retry(
        wait=wait_random_exponential(min=50, max=500),
        stop=stop_after_attempt(32)
    )
    def compare(self, evalution_p, previous_image_path, current_image_path):
        """
            Compare two images using a textual evaluation prompt.

            Args:
                evalution_p (str): Prompt describing evaluation criteria.
                previous_image_path (str): Path to the reference image.
                current_image_path (str): Path to the image to compare.

            Returns:
                str: Textual comparison result from the Gemini model.

            Raises:
                Exception: If evaluation repeatedly fails or API quota is exhausted.
        """
        current_image_part = self._load_image_part(path=current_image_path)
        previous_image_part = self._load_image_part(path=previous_image_path)

        if previous_image_path != current_image_path:
            contents = [evalution_p, previous_image_part, current_image_part]
        else:
            contents = [evalution_p, current_image_part]

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                # config=self.generate_content_config
            )
            try:
                response = response.text.strip()
                return response
            except Exception as e:
                return "failed","Exit"

        except Exception as e:
            print(f"❌ Gemini Evaluation failed: {e}")
            self._handle_exception(e)
            raise e



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    """
        Build a torchvision transformation pipeline for image preprocessing.

        Args:
            input_size (int): Size to resize the image to (square).

        Returns:
            torchvision.transforms.Compose: Transform pipeline.
    """
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    """
        Divide an image into multiple square tiles based on aspect ratio.

        Args:
            image (PIL.Image): The input image.
            min_num (int): Minimum number of tiles.
            max_num (int): Maximum number of tiles.
            image_size (int): Size of each square tile.
            use_thumbnail (bool): Whether to append a thumbnail tile.

        Returns:
            List[PIL.Image]: List of image tiles.
    """
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        [(w, h) for w in range(1, max_num + 1) for h in range(1, max_num + 1)
         if min_num <= w * h <= max_num],
        key=lambda x: x[0] * x[1]
    )

    def ratio_score(w, h):
        return abs((w / h) - aspect_ratio)

    best_ratio = min(target_ratios, key=lambda x: ratio_score(*x))
    cols, rows = best_ratio
    grid_w, grid_h = image_size * cols, image_size * rows

    resized_img = image.resize((grid_w, grid_h))
    tiles = []
    for i in range(rows):
        for j in range(cols):
            left = j * image_size
            upper = i * image_size
            right = left + image_size
            lower = upper + image_size
            tile = resized_img.crop((left, upper, right, lower))
            tiles.append(tile)

    if use_thumbnail and len(tiles) > 1:
        thumb = image.resize((image_size, image_size))
        tiles.append(thumb)

    return tiles

def load_image(image_path, input_size=448, max_num=12):
    """
        Load an image and convert it into a batch of preprocessed tiles.

        Args:
            image_path (str): Path to the image file.
            input_size (int): Tile size.
            max_num (int): Max number of tiles to return.

        Returns:
            torch.Tensor: Tensor of shape (N, 3, H, W), batch of image tiles.
    """
    image = Image.open(image_path).convert("RGB")
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num)
    transform = build_transform(input_size)
    tensor_tiles = [transform(tile) for tile in tiles]
    return torch.stack(tensor_tiles)

class InternVLEvaluator:
    def __init__(self, model_path="OpenGVLab/InternVL2_5-78B", max_num=12):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        self.model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        self.generation_config = dict(max_new_tokens=1024, do_sample=False)
        self.max_num = max_num  # max image tiles
        print(f"✅ InternVL2.5 model loaded from {model_path}")

    def evaluate(self, image_path: str, prompt: str):
        """
            Evaluate an image and prompt using InternVL2.5 model.

            Args:
                image_path (str): Path to the image.
                prompt (str): Instructional prompt.

            Returns:
                Tuple[str, Union[int, str]]: Model response and extracted score.
        """
        try:
            pixel_values = load_image(image_path, max_num=self.max_num)
            pixel_values = pixel_values.to(torch.bfloat16).cuda()

            full_prompt = "<image>\n" + prompt
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                full_prompt,
                generation_config=self.generation_config
            )

            print("📋 Model Output:\n", response)
            score = self._extract_score(response)
            return response, score
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return str(e), -1

    def _extract_score(self, text):
        """
            Extract a numeric score from the model response using regex.

            Args:
                text (str): Text output from the model.

            Returns:
                Union[int, str]: Score if found, else -1.
        """
        match = re.search(r"\*\*?Score\*\*?\s*:\s*[{(]*\s*(\d+)\s*[})]*", text, re.IGNORECASE)
        if match:
            score = match.group(1).capitalize()
            return score
        return -1
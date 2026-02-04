from mlx_lm import load, generate as mlx_generate
from openai import OpenAI
from loguru import logger
from time import sleep

GLOBAL_LLM = None

# Default MLX model from mlx-community (Qwen2.5-7B is a powerful model for local inference)
DEFAULT_MLX_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"


class LLM:
    def __init__(self, api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
        if api_key:
            self.llm = OpenAI(api_key=api_key, base_url=base_url)
            self.mlx_model = None
            self.mlx_tokenizer = None
        else:
            # Use MLX for local inference on Apple Silicon
            mlx_model_name = model if model else DEFAULT_MLX_MODEL
            logger.info(f"Loading MLX model: {mlx_model_name}")
            self.mlx_model, self.mlx_tokenizer = load(mlx_model_name)
            self.llm = None
        self.model = model
        self.lang = lang

    def generate(self, messages: list[dict]) -> str:
        if self.llm is not None:
            # OpenAI API path
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = self.llm.chat.completions.create(messages=messages, temperature=0, model=self.model)
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {e}")
                    if attempt == max_retries - 1:
                        raise
                    sleep(3)
            return response.choices[0].message.content
        else:
            # MLX local inference path
            prompt = self.mlx_tokenizer.apply_chat_template(
                messages, 
                add_generation_prompt=True,
                tokenize=False
            )
            response = mlx_generate(
                self.mlx_model,
                self.mlx_tokenizer,
                prompt=prompt,
                max_tokens=2048,
                verbose=False
            )
            return response

def set_global_llm(api_key: str = None, base_url: str = None, model: str = None, lang: str = "English"):
    global GLOBAL_LLM
    GLOBAL_LLM = LLM(api_key=api_key, base_url=base_url, model=model, lang=lang)

def get_llm() -> LLM:
    if GLOBAL_LLM is None:
        logger.info("No global LLM found, creating a default one. Use `set_global_llm` to set a custom one.")
        set_global_llm()
    return GLOBAL_LLM
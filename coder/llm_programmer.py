import glob
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests
from llama_cpp import Llama
# pip install llama-cpp-python
# pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

@dataclass
class TestRefinementResult:
    needs_more_refinement: bool
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ModelConfig:
    model_name: str
    quantization: str = "4bit"
    temperature: float = 0.7
    max_tokens: int = 2048


class LLMProgrammer(ABC):
    def __init__(self, workspace: str, model: str, init_req: str):
        self.workspace = Path(workspace)
        if not self.workspace.exists():
            self.workspace.mkdir(parents=True)
        self.model = model

        self.req_dir = self.workspace / "req"
        self.tests_dir = self.workspace / "tests"
        self.src_dir = self.workspace / "src"
        self.logs_dir = self.workspace / "logs"

        self._ensure_structure()

        self.state = {
            "requirements_loaded": False,
            "tests_generated": False,
            "tests_frozen": False,
            "code_generated": False,
            "tests_passed": False,
        }

    # ---------------------------------------------------------
    # Workspace setup
    # ---------------------------------------------------------
    def _ensure_structure(self):
        for d in [self.req_dir, self.tests_dir, self.src_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Git helpers
    # ---------------------------------------------------------
    def _git(self, *args):
        subprocess.run(["git", "-C", str(self.workspace), *args], check=True)

    def commit(self, message: str):
        return # need to solve the gitignore conflict first
        self._git("add", ".")
        self._git("commit", "-m", message)

    # ---------------------------------------------------------
    # LLM call placeholder
    # ---------------------------------------------------------
    @abstractmethod
    def _llm(self, prompt: str) -> str:
        """
        Replace this with your actual model call.
        Could be local (llama.cpp, vLLM) or remote (OpenAI, etc.)
        """
        pass

    # ---------------------------------------------------------
    # Stage 1: Load requirements
    # ---------------------------------------------------------
    def load_req(self, filename: Path = Path("raw.txt")) -> str:
        if filename.exists():
            req_path = filename
        elif (self.req_dir / filename).exists():
            req_path = self.req_dir / filename
        else:
            raise FileNotFoundError(f"File not found: {filename}")
        text = req_path.read_text()
        print(f"[load_req] Loaded requirements: {text}")
        return text

    # ---------------------------------------------------------
    # Stage 2: Generate tests
    # ---------------------------------------------------------
    @abstractmethod
    def generate_tests(self, req_text: str) -> str:
        pass

    # ---------------------------------------------------------
    # Stage 3: Refine tests
    # ---------------------------------------------------------
    @abstractmethod
    def refine_tests(self) -> TestRefinementResult:
        pass

    # ---------------------------------------------------------
    # Stage 4: Generate initial API / code
    # ---------------------------------------------------------
    @abstractmethod
    def generate_api(self, req_text: str) -> str:
        pass

    # ---------------------------------------------------------
    # Stage 5: Update code (patching)
    # ---------------------------------------------------------
    @abstractmethod
    def update_code(self, failing_output: Optional[str] = None) -> str:
        """
        failing_output: pytest output or error message
        """
        pass

    # ---------------------------------------------------------
    # Stage 6: Run tests
    # ---------------------------------------------------------
    @abstractmethod
    def run_tests(self) -> bool:
        """
        Returns True if tests pass, False otherwise.
        """
        pass


class Phi3MiniBackend:
    """
    Minimal llama.cpp backend for Phi-3-mini (GGUF).
    Works on CPU or GPU.
    """

    HF_REPO = "microsoft/Phi-3-mini-4k-instruct-gguf"  # official GGUF repo

    def __init__(self, model_config: ModelConfig):
        self.model_path = self._resolve_model_path(model_config.model_name)
        self.max_tokens = model_config.max_tokens
        self.temperature = model_config.temperature

        self._llm = Llama(
            model_path=str(self.model_path),  # path to GGUF file
            n_ctx=4096,  # The max sequence length to use - note that longer sequence lengths require much more resources
            n_threads=8, # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=0, # The number of layers to offload to GPU, if you have GPU acceleration available. Set to 0 if no GPU acceleration is available on your system.
        )

    # ---------------------------------------------------------
    # Automatic model path resolution
    # ---------------------------------------------------------
    def _resolve_model_path(self, model_name: str) -> Path:
        """
        Search for GGUF model in:
        - local ./models/
        - HuggingFace cache
        - llama.cpp cache
        """

        candidates = []

        # 1. Local project folder
        candidates += glob.glob(f"./models/{model_name}*.gguf")

        # 2. HuggingFace cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        candidates += glob.glob(str(hf_cache / "**" / f"{model_name}*.gguf"), recursive=True)

        # 3. llama.cpp cache
        llama_cache = Path.home() / ".cache" / "llama.cpp"
        candidates += glob.glob(str(llama_cache / f"{model_name}*.gguf"))

        for c in candidates:
            print(f"[Phi3MiniBackend] Found model candidate: {c}")
        if candidates:
            assert len(candidates) == 1, f"Multiple model candidates found for '{model_name}': {candidates}"
            return Path(candidates[0])

        # -----------------------------------------------------
        # Not found → download from HuggingFace
        # -----------------------------------------------------
        print(f"[Phi3MiniBackend] Model '{model_name}' not found locally.")
        print("[Phi3MiniBackend] Downloading from HuggingFace…")

        # Construct filename
        filename = f"{model_name}.gguf"
        url = f"https://huggingface.co/{self.HF_REPO}/resolve/main/{filename}"

        models_dir = Path("./models")
        models_dir.mkdir(exist_ok=True)
        path = models_dir / filename

        # Stream download
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"[Phi3MiniBackend] Downloaded model to: {path}")
        return path

    def generate(self, prompt: str) -> str:
        output = self._llm(
            f"<|user|>\n{prompt}<|end|>\n<|assistant|>",
            max_tokens=256,  # Generate up to 256 tokens
            stop=["<|end|>"], 
            echo=True,  # Whether to echo the prompt
        )

        first_answer = output['choices'][0]['text']
        print(f"[Phi3MiniBackend] Generated output: {first_answer}")
        return first_answer


class Phi3MiniProgrammer(LLMProgrammer):
    """
    Concrete implementation for:
    - Python code generation
    - pytest test generation
    - Phi-3-mini backend
    """

    def __init__(self, workspace: str, backend: Phi3MiniBackend, init_req: str):
        super().__init__(workspace, model=backend, init_req=init_req)
        self.backend = backend
        self.init_req = init_req

    # -------------------------
    # LLM call
    # -------------------------
    def _llm(self, prompt: str) -> str:
        return self.backend.generate(prompt)

    # -------------------------
    # Generate tests
    # -------------------------
    def generate_tests(self, req_text: str) -> str:
        prompt = f"""
Write pytest unit tests for the following requirements.
Tests must be deterministic and not depend on external resources.

Requirements:
{req_text}
"""
        tests = self._llm(prompt)

        path = self.tests_dir / "test_generated.py"
        path.write_text(tests)

        self.state["tests_generated"] = True
        self.commit("[tests] generated")
        return tests

    # -------------------------
    # Refine tests
    # -------------------------
    def refine_tests(self) -> TestRefinementResult:
        # Minimal implementation: no refinement
        return TestRefinementResult(needs_more_refinement=False, issues=[], suggestions=[])

    # -------------------------
    # Generate initial code
    # -------------------------
    def generate_code(self, req_text: str) -> str:
        prompt = f"""
Write Python code that satisfies the following requirements.
Do NOT write tests. Only write the implementation.

Requirements:
{req_text}
"""
        code = self._llm(prompt)

        path = self.src_dir / "module.py"
        path.write_text(code)

        self.state["code_generated"] = True
        self.commit("[src] initial implementation")
        return code

    # -------------------------
    # Update code after failing tests
    # -------------------------
    def update_code(self, failing_output: Optional[str] = None) -> str:
        prompt = f"""
The following pytest output indicates failing tests:

{failing_output}

Provide a corrected full version of module.py that fixes the issues.
Return ONLY the full updated file content.
"""
        new_code = self._llm(prompt)

        path = self.src_dir / "module.py"
        path.write_text(new_code)

        self.commit("[src] patch after failing tests")
        return new_code

    # -------------------------
    # Run tests
    # -------------------------
    def run_tests(self) -> bool:
        result = subprocess.run(["pytest", str(self.tests_dir)], cwd=self.workspace, capture_output=True, text=True)

        log_path = self.logs_dir / "pytest_last.log"
        log_path.write_text(result.stdout + "\n" + result.stderr)

        if result.returncode == 0:
            self.state["tests_passed"] = True
            return True

        return False

    def generate_api(self, req_text: str) -> str:
        # For this example, we skip separate API generation and do it in generate_code
        return self.generate_code(req_text)


if __name__ == "__main__":
    # Example usage
    # pip install huggingface-hub>=0.17.1
    # hf download microsoft/Phi-3-mini-4k-instruct-gguf
    workspace = "./tests/ws"
    path_init_req = Path("tests/test1_req.txt")
    assert path_init_req.exists(), f"Initial requirements file not found: {path_init_req}"
    init_req = path_init_req.read_text()


    model_config = ModelConfig(model_name="Phi-3-mini-4k-instruct-q4", quantization="4bit", temperature=0.7, max_tokens=2048)
    backend = Phi3MiniBackend(model_config)
    programmer = Phi3MiniProgrammer(workspace, backend, init_req)

    req_text = programmer.load_req(path_init_req)
    programmer.generate_tests(req_text)
    programmer.generate_code(req_text)

    while not programmer.run_tests():
        failing_output = (programmer.logs_dir / "pytest_last.log").read_text()
        programmer.update_code(failing_output)

    print("All tests passed! Code is ready.")

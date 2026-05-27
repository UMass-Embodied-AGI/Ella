import torch
import os

from .server import ModelProcess, ProcessChannel
from .client import *
from ..generator import Generator

class ModelManager:
    def __init__(self, device=None, port=8000, local=False):
        self.models = {}
        self._channel = None
        self._processes = []
        self.init(device, port, local)

    def init(self, device=None, port=8000, local=False):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        channel = None
        processes = []
        if local:
            if device.startswith("mps"):
                os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            channel = ProcessChannel()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                cuda_devices = [d for d in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if d]
            else:
                cuda_devices = list(range(torch.cuda.device_count()))
            dc = len(cuda_devices)
            if dc < 1:
                print(f"no CUDA devices available! starting model process in {device.upper()} mode.")
                process = ModelProcess(channel, cuda_devices=[], model_subset=["ram", "dino", "sam", "clip"], device=device)
                process.start()
                processes.append(process)
            else:
                # for smem models
                if dc > 1:
                    process = ModelProcess(channel, cuda_devices=[cuda_devices[1]], model_subset=["ram", "dino", "sam", "clip", "embedding"])
                else:
                    process = ModelProcess(channel, cuda_devices=[cuda_devices[0]], model_subset=["ram", "dino", "sam", "clip", "embedding"])
                process.start()
                processes.append(process)

                # for llms
                if dc >= 6:
                    process = ModelProcess(channel, cuda_devices=cuda_devices[2:6], model_subset=["completion"])
                elif dc > 2:
                    process = ModelProcess(channel, cuda_devices=cuda_devices[2:min(4,dc)], model_subset=["completion"])
                else:
                    process = ModelProcess(channel, cuda_devices=[cuda_devices[0]], model_subset=["completion"])
                process.start()
                processes.append(process)
        
        self.models = {
            "ram": RAMClient(device, port, channel),
            "dino": DINOClient(device, port, channel),
            "sam": SAMClient(device, port, channel),
            "clip": CLIPClient(device, port, channel),
            "embedding": EmbedClient(device, port, channel),
            "completion": CompletionClient(device, port, channel),
        }
        self._channel = channel
        self._processes = processes
    
    def set_channel(self, channel, device=None):
        if device is not None:
            self.device = device
        self._channel = channel
        for client in self.models.values():
            if hasattr(client, "server_channel"):
                client.server_channel = channel
                if device is not None:
                    client.device = device

    def get_model(self, model_name):
        if model_name not in self.models:
            raise ValueError(f"unknown model: {model_name}")
        return self.models[model_name]

    def get_generator(self, lm_source, lm_id, max_tokens=4096, temperature=0, top_p=1, logger=None):
        if lm_id not in self.models:
            self.models[lm_id] = Generator(lm_source, lm_id, max_tokens, temperature, top_p, logger)
        return self.models[lm_id]

    def close(self):
        # terminate any locally spawned model processes and cleanup channel
        try:
            for p in getattr(self, "_processes", []) or []:
                try:
                    if p.is_alive():
                        p.terminate()
                except Exception:
                    pass
            for p in getattr(self, "_processes", []) or []:
                try:
                    p.join(timeout=5)
                except Exception:
                    pass
        finally:
            self._processes = []
        if getattr(self, "_channel", None) is not None:
            try:
                if hasattr(self._channel, "shutdown"):
                    self._channel.shutdown()
            finally:
                self._channel = None
        self.models = {}

global_model_manager = ModelManager(local=False)

__all__ = ["global_model_manager", "ModelManager"]
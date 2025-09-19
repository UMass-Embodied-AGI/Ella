import torch
import os
import multiprocessing as mp

from .server import ModelProcess, ProcessChannel
from .client import *
from ..generator import Generator

# # Ensure CUDA-safe multiprocessing semantics
# try:
#     mp.set_start_method("spawn", force=True)
# except RuntimeError:
#     # Start method may already be set by the runtime; that's fine
#     pass

class ModelManager:
    def __init__(self, device='cuda', port=8000, local=False):
        self.models = {}
        self.init(device, port, local)

    def init(self, device='cuda', port=8000, local=False):
        self.device = device
        channel = None
        if local:
            channel = ProcessChannel()
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                cuda_devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            else:
                cuda_devices = list(range(torch.cuda.device_count()))
            dc = len(cuda_devices)
            if dc < 1:
                print("no CUDA devices available! do not start model process.")
                return
            # for smem models
            if dc > 1:
                process = ModelProcess(channel, cuda_devices=[cuda_devices[1]], model_subset=["ram", "dino", "sam", "clip", "embedding"])
            else:
                process = ModelProcess(channel, cuda_devices=[cuda_devices[0]], model_subset=["ram", "dino", "sam", "clip", "embedding"])
            process.start()
            
            # for llms
            if dc >= 6:
                process = ModelProcess(channel, cuda_devices=cuda_devices[2:6], model_subset=["completion"])
            elif dc > 2:
                process = ModelProcess(channel, cuda_devices=cuda_devices[2:min(4,dc)], model_subset=["completion"])
            else:
                process = ModelProcess(channel, cuda_devices=[cuda_devices[0]], model_subset=["completion"])
            process.start()
        
        self.models = {
            "ram": RAMClient(device, port, channel),
            "dino": DINOClient(device, port, channel),
            "sam": SAMClient(device, port, channel),
            "clip": CLIPClient(device, port, channel),
            "embedding": EmbedClient(device, port, channel),
            "completion": CompletionClient(device, port, channel),
        }
    
    def get_model(self, model_name):
        if model_name not in self.models:
                raise ValueError(f"unknown model: {model_name}")
        return self.models[model_name]

    def get_generator(self, lm_source, lm_id, max_tokens=4096, temperature=0, top_p=1, logger=None):
        if lm_id not in self.models:
            self.models[lm_id] = Generator(lm_source, lm_id, max_tokens, temperature, top_p, logger)
        return self.models[lm_id]

global_model_manager = ModelManager(local=False)

__all__ = ["global_model_manager", "ModelManager"]
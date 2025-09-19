import torch
import pickle
import requests
import random
import multiprocessing as mp

from agents.sg.builder.model import *
from .server import ProcessChannel

def all_to_device(arg, device: str):
    if isinstance(arg, torch.Tensor):
        return arg.to(device)
    if isinstance(arg, tuple):
        return tuple(all_to_device(a, device) for a in arg)
    if isinstance(arg, list):
        return [all_to_device(a, device) for a in arg]
    if isinstance(arg, dict):
        return {k: all_to_device(v, device) for k, v in arg.items()}
    return arg

class ModelClient:
    def __init__(self, device="cuda", server_port=8000, server_channel: ProcessChannel=None):
        self.device = device
        self.server_port = server_port
        self.server_channel = server_channel
    
    def post(self, url, *args, **kwargs):
        args = all_to_device(args, "cpu")
        kwargs = all_to_device(kwargs, "cpu")
        if self.server_channel is not None:
            id = random.randint(0, 1000000000)
            # print(f"putting {id}")
            self.server_channel.put_c(id, url, (args, kwargs))
            result = self.server_channel.get_c(id)
            return all_to_device(result, self.device)
        else:
            return all_to_device(self.http_post(url, *args, **kwargs), self.device)

    def http_post(self, url, *args, **kwargs):
        args = all_to_device(args, "cpu")
        kwargs = all_to_device(kwargs, "cpu")
        data = pickle.dumps((args, kwargs))
        r = None
        for i in range(5):
            try:
                r = requests.post(f"http://localhost:{self.server_port}{url}", data=data)
                if r.status_code == 200:
                    break
            except Exception as e:
                continue
        if r is None or r.status_code != 200:
            raise ValueError("failed to connect to server")
        return pickle.loads(r.content)

class RAMClient(ModelClient, RAMWrapper):
    def predict(self, rgb):
        return self.post(f"/ram", rgb)

class SAMClient(ModelClient, SAMWrapper):
    def predict(self, rgb, boxes, annotate=False):
        return self.post(f"/sam", rgb, boxes, annotate)

class DINOClient(ModelClient, DINOWrapper):
    def predict(self, rgb, text, annotate=False):
        return self.post(f"/dino", rgb, text, annotate)

class CLIPClient(ModelClient, CLIPWrapper):
    def predict_image(self, rgb, normalize=True):
        return self.post(f"/clip/image", rgb, normalize)
    
    def predict_text(self, text, normalize=True):
        return self.post(f"/clip/text", text, normalize)

class EmbedClient(ModelClient):
    def encode(self, text):
        return self.post(f"/embedding", text)

class CompletionClient(ModelClient):
    def complete(self, text, max_tokens=4096, temperature=0, top_p=1):
        from vllm import SamplingParams
        return self.post(f"/completion", text, sampling_params=SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p))
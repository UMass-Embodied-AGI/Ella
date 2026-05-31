import argparse
import time
import logging
import pickle
import numpy as np
import torch
from http.server import BaseHTTPRequestHandler, HTTPServer
import multiprocessing as mp
import os

_ctx = mp.get_context('spawn')

def auto_batched(args: list):
    def batch_one(arg):
        if isinstance(arg[0], bool):
            return arg[0]
        else:
            return arg
    
    if isinstance(args[0], tuple):
        return tuple([batch_one([arg[i] for arg in args]) for i in range(len(args[0]))])
    elif isinstance(args[0], dict):
        return {k: batch_one([arg[k] for arg in args]) for k in args[0].keys()}
    raise ValueError("args should be list or dict")

class ModelServerHandler(BaseHTTPRequestHandler):
    server: 'ModelServer'
    
    def do_POST(self):
        start = time.time()
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        args, kwargs = pickle.loads(post_data)
        result = self.server.process.process(self.path, *args, **kwargs)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(pickle.dumps(result))
        self.server.process.logger.debug(f"{self.path} elapsed: {time.time() - start}")

class ProcessChannel:
    UNBATCHED = ["/sam", "/clip/image"]
    def __init__(self):
        self._manager = _ctx.Manager()
        self.input = {
            "/ram": _ctx.Queue(),
            "/dino": _ctx.Queue(),
            "/sam": _ctx.Queue(),
            "/clip/image": _ctx.Queue(),
            "/clip/text": _ctx.Queue(),
            "/embedding": _ctx.Queue(),
            "/completion": _ctx.Queue(),
        }
        self.output = self._manager.dict()
        self.cond_c = _ctx.Condition()
        self.cond_s = _ctx.Condition()
    
    def put_s(self, vals): # vals: [(id, val), ...]
        if not isinstance(vals, list):
            vals = [vals]
        with self.cond_c:
            for id, val in vals:
                self.output[id] = val
            self.cond_c.notify_all()
    
    def get_s(self, model_subset=None): # return url, [(id, val), ...]
        import queue
        with self.cond_s:
            while True:
                for url, q in self.input.items():
                    if model_subset is not None and all([m not in url for m in model_subset]):
                        continue
                    try:
                        first = q.get_nowait()
                    except queue.Empty:
                        continue
                    if url in self.UNBATCHED:
                        return url, first
                    vals = [first]
                    for _ in range(3):
                        try:
                            vals.append(q.get_nowait())
                        except queue.Empty:
                            break
                    return url, vals
                self.cond_s.wait(timeout=1.0)

    def put_c(self, id, url, val):
        self.input[url].put((id, val))
        with self.cond_s:
            self.cond_s.notify_all()
    
    def get_c(self, id):
        with self.cond_c:
            while id not in self.output:
                self.cond_c.wait(timeout=1.0)
            return self.output.pop(id)

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_manager']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._manager = None

    def shutdown(self):
        # best-effort cleanup of queues and manager resources
        try:
            for q in self.input.values():
                try:
                    q.close()
                except Exception:
                    pass
                try:
                    q.join_thread()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if self._manager is not None:
                self._manager.shutdown()
        except Exception:
            pass

class ModelProcess(_ctx.Process):
    def __init__(self, channel: ProcessChannel, cuda_devices=[0], model_subset: list=None, device: str=None):
        super().__init__()
        self.cuda_devices = cuda_devices
        self.channel = channel
        self.model_subset = model_subset
        self.models = None
        # explicit device overrides the cuda_devices-derived default
        self._device_override = device

    def _load_model(self, model_name):
        if self.model_subset is not None and model_name not in self.model_subset:
            return None
        device = self._device_override or ("cuda" if self.cuda_devices else "cpu")
        if model_name == "ram":
            from agents.sg.builder.model import RAMWrapper
            return RAMWrapper(device=device)
        if model_name == "dino":
            from agents.sg.builder.model import DINOWrapper
            return DINOWrapper(device=device)
        if model_name == "sam":
            from agents.sg.builder.model import SAMWrapper
            return SAMWrapper(device=device)
        if model_name == "clip":
            from agents.sg.builder.model import CLIPWrapper
            return CLIPWrapper(device=device)
        if model_name == "embedding":
            from vllm import LLM
            return LLM(model="BAAI/bge-base-en-v1.5")
        if model_name == "completion":
            from vllm import LLM
            return LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B", dtype="float16", tensor_parallel_size=len(self.cuda_devices), enable_prefix_caching=False, enable_chunked_prefill=False)
        raise ValueError(f"unknown model: {model_name}")

    def init(self):
        if self._device_override and not self._device_override.startswith("cuda"):
            # MPS or CPU: don't touch CUDA_VISIBLE_DEVICES
            pass
        elif self.cuda_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in self.cuda_devices])
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.models = {}

        self.logger = logging.Logger("model_server")
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        os.makedirs("output", exist_ok=True)
        file_handler = logging.FileHandler("output/model_server.log", mode="w")
        file_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def run(self):
        while True:
            url, vals = self.channel.get_s(self.model_subset)
            if url in self.channel.UNBATCHED:
                id, (args, kwargs) = vals
                result = self.process(url, *args, **kwargs)
                self.channel.put_s((id, result))
            else:
                args = auto_batched([val[0] for id, val in vals])
                kwargs = auto_batched([val[1] for id, val in vals])
                # self.logger.info(f"{url} processing batch_size={len(vals)}, ids={[id for id, val in vals]}")
                result = self.process(url, *args, **kwargs)
                self.channel.put_s([(val[0], res) for val, res in zip(vals, result)])
    
    def process(self, url, *args, **kwargs):
        if self.models is None:
            self.init()
        start_pred = time.time()
        paths = url[1:].split("/")
        if paths[0] not in self.models:
            self.models[paths[0]] = self._load_model(paths[0])
        model = self.models[paths[0]]
        if paths[0] == "embedding":
            outputs = model.encode(*args, **kwargs)
            result = [np.array(output.outputs.embedding) for output in outputs]
        elif paths[0] == "completion":
            outputs = model.chat(*args, **kwargs)
            result = [output.outputs[0].text.split("</think>")[-1] for output in outputs]
        elif paths[0] == "clip":
            if paths[1] == "image":
                result = model.predict_image(*args, **kwargs)
            else:
                result = model.predict_text(*args, **kwargs)
        else:
            result = model.predict(*args, **kwargs)
        if self.logger:
            self.logger.debug(f"{url} predict elapsed: {time.time() - start_pred}")
        return result

class ModelServer(HTTPServer):
    def __init__(self, server_address, device):
        super().__init__(server_address, ModelServerHandler)

        self.device = device
        self.process = ModelProcess(device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--port", default=8000)
    args = parser.parse_args()
    server = ModelServer(('localhost', args.port), args.device)
    server.serve_forever()

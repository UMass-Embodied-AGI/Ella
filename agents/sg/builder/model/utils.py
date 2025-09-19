import requests
import os

def get_device_type(device: str):
    if device.startswith("cuda"):
        return "cuda"
    return "cpu"

def check_download_to(url: str, path: str):
    if os.path.exists(path):
        return
    print(f"Downloading checkpoint {url} to {path}")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    r = requests.get(url, stream=True)
    with open(path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)

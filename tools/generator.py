import json
import logging
import random
import time

import backoff
import os
import base64
from PIL import Image
from io import BytesIO
from typing import Union
import traceback
import pickle

from vico.tools.utils import atomic_save


def encode_image(img: Union[str, Image.Image]) -> str:
	if isinstance(img, str): # if it's image path, open and then encode/decode
		with open(img, "rb") as image_file:
			return base64.b64encode(image_file.read()).decode('utf-8')
	elif isinstance(img, Image.Image): # if it's image already, buffer and then encode/decode
		buffered = BytesIO()
		img.save(buffered, format="JPEG")
		return base64.b64encode(buffered.getvalue()).decode("utf-8")
	else:
		raise Exception("img can only be either str or Image.Image")

class Generator:
	def __init__(self, lm_source, lm_id, max_tokens=4096, temperature=0.7, top_p=1.0, logger=None):
		self.lm_source = lm_source
		self.lm_id = lm_id
		self.max_tokens = max_tokens
		self.temperature = temperature
		self.top_p = top_p
		self.logger = logger
		self.caller_analysis = {}
		if self.logger is None:
			self.logger = logging.getLogger(__name__)
			self.logger.setLevel(logging.DEBUG)
			self.logger.addHandler(logging.StreamHandler())
		self.max_retries = 3
		self.cost = 0 # cost in us dollars
		self.cache_path = f"cache_{lm_id}.pkl"
		if os.path.exists(self.cache_path):
			with open(self.cache_path, 'rb') as f:
				self.cache = pickle.load(f)
		else:
			self.cache = {}
		if self.lm_id == "text-embedding-3-small":
			self.embedding_dim = 1536
		elif self.lm_id == "text-embedding-3-large":
			self.embedding_dim = 3072
		else:
			self.embedding_dim = 0
		if self.lm_id == "gpt-4o":
			self.input_token_price = 2.5 * 10 ** -6
			self.output_token_price = 10 * 10 ** -6
		elif self.lm_id == "gpt-4.1":
			self.input_token_price = 2 * 10 ** -6
			self.output_token_price = 8 * 10 ** -6
		elif self.lm_id == "o3-mini" or self.lm_id == "o4-mini":
			self.input_token_price = 1.1 * 10 ** -6
			self.output_token_price = 4.4 * 10 ** -6
		elif self.lm_id == "gpt-35-turbo":
			self.input_token_price = 1 * 10 ** -6
			self.output_token_price = 2 * 10 ** -6
		else:
			self.input_token_price = -1 * 10 ** -6
			self.output_token_price = -2 * 10 ** -6
		if self.lm_source == "openai":
			from openai import OpenAI
			self.client = OpenAI(
				api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
				max_retries=self.max_retries,
			) if 'OPENAI_API_KEY' in os.environ else None
		elif self.lm_source == "azure":
			from openai import AzureOpenAI
			try:
				api_keys = json.load(open(".api_keys.json", "r"))
				if "embedding" in self.lm_id:
					api_keys = api_keys["embedding"]
				else:
					api_keys = api_keys["all"]
				api_keys = random.sample(api_keys, 1)[0]
				self.logger.info(f"Using Azure API key: {api_keys['AZURE_ENDPOINT']}")
				self.client = AzureOpenAI(
					azure_endpoint=api_keys['AZURE_ENDPOINT'],
					api_key=api_keys['OPENAI_API_KEY'],
					api_version="2024-12-01-preview",
				)
			except Exception as e:
				self.logger.error(f"Error loading .api_keys.json: {e} with traceback: {traceback.format_exc()}")
				self.client = None
		elif self.lm_source == "huggingface":
			from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
			# self.client = AutoModelForCausalLM.from_pretrained(self.lm_id)
			# self.tokenizer = AutoTokenizer.from_pretrained(self.lm_id)
			self.client = pipeline(
				"text-generation",
				model=self.lm_id,
				device_map="auto",
			)
			# lm_id: "meta-llama/Meta-Llama-3.1-8B-Instruct"
		elif self.lm_source == "llava":
			from llava.model.builder import load_pretrained_model
			from llava.mm_utils import get_model_name_from_path
			from llava.constants import (
				IMAGE_TOKEN_INDEX,
				DEFAULT_IMAGE_TOKEN,
				DEFAULT_IM_START_TOKEN,
				DEFAULT_IM_END_TOKEN,
				IMAGE_PLACEHOLDER,
			)
			from llava.conversation import conv_templates, SeparatorStyle
			import torch
			from llava.mm_utils import (
				process_images,
				tokenizer_image_token,
				get_model_name_from_path,
				KeywordsStoppingCriteria,
			)
			self.model_name = get_model_name_from_path(self.lm_id)
			if 'lora' in self.model_name and '7b' in self.model_name:
				self.lm_base = "liuhaotian/llava-v1.5-7b"
			self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
				model_path=self.lm_id, model_base=self.lm_base, model_name=self.model_name, )  # load_4bit=True)
		elif self.lm_source == "vla": # will merge to huggingface later
			from transformers import AutoModelForVision2Seq, AutoProcessor
			from peft import PeftModel
			import torch
			self.processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
			self.base_model = AutoModelForVision2Seq.from_pretrained(
				"openvla/openvla-7b",
				attn_implementation="flash_attention_2",
				torch_dtype=torch.bfloat16,
				low_cpu_mem_usage=True,
				trust_remote_code=True
			)
			self.lora_model = PeftModel.from_pretrained(
				self.base_model,
				"/home/zheyuanzhang/Documents/GitHub/VLA/adapter_tmp/openvla-7b+ella_dataset+b16+lr-0.0005+lora-r32+dropout-0.0", # will add the lora adapter path into env config later
				torch_dtype=torch.bfloat16,
				device_map="auto",
			).to("cuda:0")
		elif self.lm_source == "google":
			import google.generativeai as genai
			from google.generativeai.types import HarmCategory, HarmBlockThreshold
			from google.api_core.exceptions import ResourceExhausted
			genai.configure(api_key=os.environ["GEMINI_API_KEY"])
			self.client = genai.GenerativeModel(self.lm_id)
		elif self.lm_source == "local":
			from openai import OpenAI
			from tools.model_manager import global_model_manager
			self.client = global_model_manager.get_model("completion")
			self.embed_client = global_model_manager.get_model("embedding")
			self.input_token_price = 0
			self.output_token_price = 0
		else:
			raise NotImplementedError(f"{self.lm_source} is not supported!")

	def generate(self, prompt, max_tokens=None, temperature=None, top_p=None, img=None, json_mode=False, chat_history=None, caller="none"):
		if max_tokens is None:
			max_tokens = self.max_tokens
		if temperature is None:
			temperature = self.temperature
		if top_p is None:
			top_p = self.top_p
			
		if self.lm_source == 'openai' or self.lm_source == 'azure':
			return self.openai_generate(prompt, max_tokens, temperature, top_p, img, json_mode, chat_history, caller)
		elif self.lm_source == 'gemini':
			return self.gemini_generate(prompt)
		elif self.lm_source == 'huggingface':
			return self.huggingface_generate(prompt, max_tokens, temperature, top_p)
		elif self.lm_source == 'vla':
			return self.vla_generate(prompt, img, max_tokens)
		elif self.lm_source == 'local':
			message = [] if chat_history is None else chat_history
			message.append({ "role": "user", "content": prompt })
			return self.client.complete(message, max_tokens, temperature, top_p)
		else:
			raise ValueError(f"Invalid lm_source: {self.lm_source}")

	def openai_generate(self, prompt, max_tokens, temperature, top_p, img: Union[str, Image.Image, None, list], json_mode=False, chat_history=None, caller="none"):
		@backoff.on_exception(
			backoff.expo,  # Exponential backoff
			Exception,  # Base exception to catch and retry on
			max_tries=self.max_retries,  # Maximum number of retries
			jitter=backoff.full_jitter,  # Add full jitter to the backoff
			logger=self.logger  # Logger for retry events, which is in the level of INFO
		)
		def _generate():
			content = [{
						 "type": "text",
						 "text": prompt
					 }, ]
			if img is not None:
				if type(img) != list:
					imgs = [img]
				else:
					imgs = img
				for each_img in imgs:
					content.append({
						"type": "image_url",
						"image_url": {"url": f"data:image/png;base64,{encode_image(each_img)}"},
						# "detail": "low"
					})
			if chat_history is not None:
				messages = chat_history
			else:
				messages = []
			messages.append(
				{
					"role": "user",
					"content": content
				})
			start = time.perf_counter()
			if self.lm_id[0] == 'o':
				params = {
					"reasoning_effort": "high",
					"timeout": 400,
				}
			else:
				params = {
					"temperature": temperature,
					"top_p": top_p,
					"timeout": 40,
				}
			response = self.client.chat.completions.create(
					model=self.lm_id,
					messages=messages,
					max_completion_tokens=max_tokens,
					response_format={
						"type": "json_object" if json_mode else "text"
					},
					**params,
				)
			self.logger.debug(f"api request time: {time.perf_counter() - start}")
			with open(f"chat_raw.jsonl", 'a') as f:
				chat_entry = {
					"prompt": prompt,
					"response": response.model_dump_json(indent=4)
				}
				# Write as a single JSON object per line
				f.write(json.dumps(chat_entry))
				f.write('\n')
			usage = dict(response.usage)
			self.cost += usage['completion_tokens'] * self.output_token_price + usage['prompt_tokens'] * self.input_token_price
			if caller in self.caller_analysis:
				self.caller_analysis[caller].append(usage['total_tokens'])
			else:
				self.caller_analysis[caller] = [usage['total_tokens']]
			response = response.choices[0].message.content
			# self.logger.debug(f'======= prompt ======= \n{prompt}', )
			# self.logger.debug(f'======= response ======= \n{response}')
			# self.logger.debug(f'======= usage ======= \n{usage}')
			if self.cost > 7:
				self.logger.critical(f'COST ABOVE 7 dollars! There must be sth wrong. Stop the exp immediately!')
				raise Exception(f'COST ABOVE 7 dollars! There must be sth wrong. Stop the exp immediately!')
			self.logger.info(f'======= total cost ======= {self.cost}')
			return response
		try:
			return _generate()
		except Exception as e:
			self.logger.error(f"Error with openai_generate: {e}, the prompt was:\n {prompt}")
			return ""

	def gemini_generate(self, prompt):
		try:
			response = self.client.generate_content(prompt, safety_settings={
				HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
				HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
				HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
				HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
			})
			usage = response.usage_metadata.total_token_count
			self.cost += usage * self.input_token_price
			response = response.text
			self.logger.debug(f'======= prompt ======= \n{prompt}', )
			self.logger.debug(f'======= response ======= \n{response}')
			self.logger.debug(f'======= usage ======= \n{usage}')
			self.logger.debug(f'======= total cost ======= {self.cost}')
		except Exception as e:
			self.logger.error(f"Error generating content: {e}")
			raise e
		return response

	def huggingface_generate(self, prompt, max_tokens, temperature, top_p):
		messages = []
		messages.append(
			{
				"role": "system",
				"content": "You are a helpful assistant."
			})
		messages.append(
			{
				"role": "user",
				"content": prompt
			})
		response = self.client(
			prompt,
			do_sample = False if temperature == 0 else True,
			temperature=temperature if temperature != 0 else 1,
			top_p=top_p,
			max_new_tokens=max_tokens,)
		response = response[0]['generated_text']
		self.logger.debug(f'======= prompt ======= \n{prompt}', )
		self.logger.debug(f'======= response ======= \n{response}')
		return response
		# inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True, padding="max_length", max_length=self.max_tokens, truncation=True)
		# outputs = self.client.generate(**inputs, max_length=self.max_tokens, num_return_sequences=1, temperature=self.temperature, top_p=self.top_p)
		# response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
		# self.logger.debug(f'======= prompt ======= \n{prompt}', )
		# self.logger.debug(f'======= response ======= \n{response}')

	def vla_generate(self, prompt, img, max_tokens):
		import torch
		inputs = self.processor(prompt, Image.fromarray(img)).to("cuda:0", dtype=torch.bfloat16)
		with torch.no_grad():
			outputs = self.lora_model.generate(**inputs, max_length=max_tokens)
		return self.processor.decode(outputs[0], skip_special_tokens=True)

	def get_embedding(self, text, caller="none"):
		@backoff.on_exception(
			backoff.expo,  # Exponential backoff
			Exception,  # Base exception to catch and retry on
			max_tries=self.max_retries,  # Maximum number of retries
			jitter=backoff.full_jitter,  # Add full jitter to the backoff
			logger=self.logger  # Logger for retry events
		)
		def _embed() -> list:
			response = self.client.embeddings.create(
				model=self.lm_id,
				input=[text]
			)
			usage = dict(response.usage)
			if caller in self.caller_analysis:
				self.caller_analysis[caller].append(usage['total_tokens'])
			else:
				self.caller_analysis[caller] = [usage['total_tokens']]
			return response.data[0].embedding

		if text in self.cache:
			return self.cache[text]

		if self.lm_source == "local":
			embedding = self.embed_client.encode(text)
		else:
			embedding = _embed()
		self.cache[text] = embedding
		if len(self.cache) % 10 == 0:
			atomic_save(self.cache_path, pickle.dumps(self.cache))
		return embedding


if __name__ == "__main__":
	generator = Generator(
		lm_source='azure',
		lm_id='o4-mini',
		max_tokens=4096,
		temperature=0.7,
		top_p=1.0,
		logger=None
	)
	prompt1 = "What is the meaning of life?"
	prompt2 = "How many images did I sent you?"
	# print(generator.generate(prompt1))
	print(generator.generate(prompt2, img=["ViCo/assets/imgs/avatars/Abraham Lincoln.png", "ViCo/assets/imgs/avatars/Albert Einstein.png"]))
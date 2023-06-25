from transformers import AutoTokenizer, AutoModel, AutoFeatureExtractor
import av
import numpy as np
import torch
from datasets import Dataset, Audio


def generate_text_output_layer(tokenizer, text_model, sample_text):
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  text_inputs = tokenizer(sample_text, return_tensors="pt", padding=True, )
  text_inputs = {k:torch.tensor(v).to(device) for k,v in text_inputs.items()}

  with torch.no_grad():
      hidden_text = text_model(**text_inputs)

  cls_text = hidden_text.last_hidden_state[:,0,:].clone()

  return cls_text



def generate_speech_output_layer(speech_feature_extractor, speech_model, sample_speech):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  speech_hidden = torch.empty((0,)).to(device)

  speech_input = speech_feature_extractor(sample_speech["audio"]["array"], sampling_rate=16000, return_tensors="pt", padding=True)
  speech_input = {k:torch.tensor(v).to(device) for k,v in speech_input.items()}

  with torch.no_grad():
      output = speech_model(**speech_input)
  hidden = output.last_hidden_state[:,0,:]
  speech_hidden = torch.cat((speech_hidden, hidden), 0)

  return speech_hidden



def read_video_pyav(container, indices):
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def generate_video_output_layer(feature_extractor, video_model, video_file_path):

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  video_sample = str(video_file_path)

  video_hidden = torch.empty((0,)).to(device)

  # video clip consists of 300 frames (10 seconds at 30 FPS)
  container = av.open(video_sample)

  # sample 16 frames
  indices = sample_frame_indices(clip_len=16, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
  video = read_video_pyav(container, indices)

  inputs = feature_extractor(list(video), return_tensors="pt")
  inputs = {k:torch.tensor(v).to(device) for k,v in inputs.items()}

  with torch.no_grad():
    output = video_model(**inputs)

  hidden = output.last_hidden_state[:,0,:]
  video_hidden = torch.cat((video_hidden, hidden), 0)

  return video_hidden

  

def prepare_data(text_path, speech_path, video_path):

  text_model_output = generate_text_output_layer(tokenizer, text_model, text_path)

  audio_dataset = Dataset.from_dict({"audio": [speech_path]}).cast_column("audio", Audio())
  speech_model_output = generate_speech_output_layer(speech_feature_extractor, speech_model, audio_dataset[0])

  video_model_output = generate_video_output_layer(feature_extractor, video_model, video_path)
  concated_data = torch.cat((text_model_output, speech_model_output), 1)
  concated_data = torch.cat((concated_data, video_model_output), 1)

  return concated_data



def load_models(text_mdl_name, speech_mdl_name, video_mdl_name):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  
  global feature_extractor, video_model, speech_feature_extractor, speech_model, tokenizer, text_model

  tokenizer = AutoTokenizer.from_pretrained(text_mdl_name)
  text_model = AutoModel.from_pretrained(text_mdl_name).to(device)

  speech_feature_extractor = AutoFeatureExtractor.from_pretrained(speech_mdl_name)
  speech_model = AutoModel.from_pretrained(speech_mdl_name).to(device)

  feature_extractor = AutoFeatureExtractor.from_pretrained(video_mdl_name)
  video_model = AutoModel.from_pretrained(video_mdl_name).to(device)

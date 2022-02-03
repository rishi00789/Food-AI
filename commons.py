import io
from utils.output_utils import prepare_output
from args import get_parser
from model import get_model
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import pickle
import os
import torchvision.transforms as transforms

def g_model():
	data_dir = 'data/'
	ingrs_vocab = pickle.load(open('ingr_vocab.pkl','rb'))
	vocab = pickle.load(open('instr_vocab.pkl','rb'))
	ingr_vocab_size = len(ingrs_vocab)
	instrs_vocab_size = len(vocab)
	output_dim = instrs_vocab_size


	import sys; sys.argv=['']; del sys
	args = get_parser()
	args.maxseqlen = 15
	args.ingrs_only=False
	model = get_model(args, ingr_vocab_size, instrs_vocab_size)
	# Load the trained model parameters
	model_path = 'modelbest.ckpt'
	model.load_state_dict(torch.load(model_path, map_location='cpu'))
	model.to('cpu')
	model.eval()
	model.ingrs_only = False
	model.recipe_only = False
	return model

def get_tensor(image_bytes):
	my_transforms = transforms.Compose([transforms.Resize(256),
        				    transforms.CenterCrop(224),
        				    transforms.ToTensor(),
        				    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             					  std=[0.229, 0.224, 0.225])])
	image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0).to('cpu')

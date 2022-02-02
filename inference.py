
from commons import g_model, get_tensor
import torch
import pickle
import os
from utils.output_utils import prepare_output
model = g_model()

def get_recipe(image_bytes):
	a=[]
	b=[]
	c=[]
	len1=[]
	len2=[]
	tensor = get_tensor(image_bytes)
	greedy = [True, False, False, False]
	beam = [-1, -1, -1, -1]
	temperature = 1.0
	numgens = len(greedy)
	for i in range(numgens):
		with torch.no_grad():
			outputs = model.sample(tensor, greedy=greedy[i],temperature=temperature, beam=beam[i], true_ingrs=None)
		ingr_ids = outputs['ingr_ids'].cpu().numpy()
		recipe_ids = outputs['recipe_ids'].cpu().numpy()
		data_dir = 'data/'
		ingrs_vocab = pickle.load(open('ingr_vocab.pkl','rb'))
		vocab = pickle.load(open('instr_vocab.pkl','rb'))
		outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
		len1.append(len(outs['ingrs']))
		len2.append(len(outs['recipe']))
		a.append(outs['title'])
		b.append(outs['ingrs'])
		c.append(outs['recipe'])

	return a,b,c,len1,len2

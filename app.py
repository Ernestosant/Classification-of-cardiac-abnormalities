from fastai.learner import load_learner
import numpy as np
import pandas as pd
from tsai.all import*
from fastai.tabular.all import *
from sklearn.preprocessing import MinMaxScaler
import gradio as gr

model = load_learner('model_1.pkl')
# inference fucntion
def inference(model, x_test):  
  min_max_scaler = MinMaxScaler()
  x_test = min_max_scaler.fit_transform(x_test)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1],1))
  x_test = np.transpose(x_test, (0, 2, 1))
  id = np.linspace(1, x_test.shape[0], x_test.shape[0]).astype('int')
  dls = model.dls.test_dl(itemify(x_test, id))
  preds, _ =  model.get_preds(dl=dls)
  final_preds = preds.argmax(dim=1)+1
  results = pd.DataFrame()
  results['ID']=id
  results['Results']=final_preds

  return results

def load_csv(file_p):   
    df = pd.read_csv(file_p.name)    
    return df

file_comp = gr.components.File(label="Load csv file")

def get_predictions(file_p):
  x_test = load_csv(file_p)
  results = inference(model, x_test)
  return results
dataframe = gr.components.DataFrame()

demo = gr.Interface(fn=get_predictions, inputs=file_comp, outputs=[dataframe])

demo.launch()

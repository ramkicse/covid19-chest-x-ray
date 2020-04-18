#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, render_template, request
import torch
import torch.nn.functional as F

import torchvision
from PIL import Image
from torchvision import transforms, models
from torch import nn
from flask import send_from_directory
import re

import os


# In[5]:


# from captum.attr import IntegratedGradients
# from captum.attr import GradientShap
# from captum.attr import Occlusion
# from captum.attr import NoiseTunnel
# from captum.attr import visualization as viz
import numpy as np
from gradcam.utils import visualize_cam
from gradcam import GradCAM, GradCAMpp
from torchvision.utils import make_grid, save_image

# In[6]:


checkpoint = 'ckpt-900.pth.tar'


# In[7]:


model = models.resnet50(pretrained=False)
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 3),
                                nn.Softmax(1))


# In[8]:


checkpoint  = torch.load(checkpoint)


# In[9]:


state_dict = checkpoint['state_dict']


# In[10]:


from collections import OrderedDict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k[7:] # remove `module.`
    new_state_dict[name] = v


# In[11]:


model.load_state_dict(new_state_dict)
print('Model loaded')
model.eval()


if torch.cuda.is_available():
    model = model.cuda()
# In[12]:

gradcam = GradCAM(model, model.layer4)


configs = [
    dict(model_type='resnet', arch=model, layer_name='layer4'),
]
for config in configs:
    if torch.cuda.is_available():
        config['arch'].cuda().eval()
    else:
        config['arch'].eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]

gradcam, gradcam_pp = cams[0]

transform = transforms.Compose([
                                      transforms.Resize((224,224)),
                                      transforms.ToTensor()
                                     
                                      ])

normalize =  transforms.Normalize(mean=[0.5], std=[0.5])


classes = ['normal', 'COVID-19', 'other_pneumonia']
class_dict ={classes[i]: i for i in range(3)}

# In[13]:


#integrated_gradients = IntegratedGradients(model)


# In[17]:


app = Flask(__name__, static_folder='.')


# In[19]:

root_dir=os.getcwd()
print(root_dir)

@app.route('/assets/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.join(root_dir,'assets/'),filename)

@app.route('/result')
def serve_result():
    return send_from_directory(os.path.join(root_dir),'result.png')

@app.route('/')
def index_view():
    return render_template('index.html')



@app.route('/predict',methods=['GET','POST'])
def predict():
    print(request.method)
    f = request.files['image']
    
    file_path = os.path.join(root_dir, 'uploads', f.filename)
    print(file_path)
    f.save(file_path)
    #file_path='/home/ramkik/covid19/static/uploads/E63574A7-4188-4C8D-8D17-9D67A18A1AFA.jpeg'
    image = Image.open(file_path).convert('RGB')

    image = transform(image)

    normalize_image = normalize(image)
    normalize_image = normalize_image.unsqueeze(0)
    if torch.cuda.is_available():
        normalize_image = normalize_image.cuda()
        
    #print(normalize_image.shape)
    output = model(normalize_image)
    print(output)

#     output = F.softmax(output)
#     print(output)
    prediction_score, pred_label_idx = torch.topk(output, 1)
    pred_label_idx.squeeze_()

    print(normalize_image.shape)
    mask, _ = gradcam(normalize_image)
    heatmap, result = visualize_cam(mask, image)
    mask_pp, _ = gradcam_pp(normalize_image)
    heatmap_pp, result_pp = visualize_cam(mask_pp, image, alpha=1.0)

    save_image(result, 'result.png', nrow=1)

#         grad = transforms.ToPILImage()(result_pp)
#         plt.imshow(grad)




    response = {}
    print('output : ',output)
    print(pred_label_idx)
    print(prediction_score)

    response['class'] = classes[pred_label_idx.item()]
    response['score'] = str(prediction_score.item())

    return response

if __name__ == '__main__':
    app.run(debug=True, port=8000)


# In[ ]:





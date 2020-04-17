#!/usr/bin/env python
# coding: utf-8

# In[4]:


from flask import Flask, render_template, request
import torch
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


checkpoint = 'ckpt-2999.pth.tar'


# In[7]:


model = models.resnet50(pretrained=False)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))


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
model = model.cuda()
# In[12]:

gradcam = GradCAM(model, model.layer4)


configs = [
    dict(model_type='resnet', arch=model, layer_name='layer4'),
]
for config in configs:
    config['arch'].cuda().eval()

cams = [
    [cls.from_config(**config) for cls in (GradCAM, GradCAMpp)]
    for config in configs
]

gradcam, gradcam_pp = cams[0]

transform = transforms.Compose([transforms.Grayscale(1),
                                      transforms.Resize((224,224)),
                                      transforms.ToTensor()
                                     
                                      ])

normalize =  transforms.Normalize(mean=[0.5], std=[0.5])


# In[13]:


#integrated_gradients = IntegratedGradients(model)


# In[17]:


app = Flask(__name__, static_folder='.')


# In[19]:

root_dir='/home/ramkik/covid19/static/'

@app.route('/assets/<path:filename>')
def serve_static(filename):
    #root_dir = os.path.dirname(os.getcwd())
    #print(os.path.join(root_dir,'assets/'), filename)
    return send_from_directory(os.path.join(root_dir,'assets/'),filename)

@app.route('/result')
def serve_result():
    #root_dir = os.path.dirname(os.getcwd())
    #print(os.path.join(root_dir,'assets/'), filename)
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
    image = Image.open(file_path)
    image = transform(image)

    normalize_image = normalize(image)
    normalize_image = normalize_image.unsqueeze(0)
    normalize_image = normalize_image.cuda()
    output = model(normalize_image)
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

    if pred_label_idx.item() == 0:
        response['class'] = 'COVID-19'
    else:
        response['class'] = 'NORMAL'
    response['score'] = str(pow(10.0,prediction_score.item()))

    return response

if __name__ == '__main__':
    app.run(debug=True, port=8000)


# In[ ]:





# Text Recognition Machine Learning Model
This is a basic machine learning model project that detects text from given images, which was part of my final year project in [CityU](http://dspace.cityu.edu.hk/handle/2031/9511).

## Development Environment
- <b>Machine Learning Platform: </b>TensorFlow + Keras API   
- <b>Development Language: </b> Python   
- <b>Datasets: </b>[EMNIST Balanced Datasets](https://www.kaggle.com/datasets/crawford/emnist)   
- <b>ML Model Algorithm: </b>Convolutional Neural Network (CNN)     
- <b>Model Designed as below</b>
<table class="tg">
<thead>
  <tr>
    <th class="tg-0pky" colspan="4">Model: "sequential"</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-0pky">Layer</td>
    <td class="tg-0pky">Type</td>
    <td class="tg-0pky">Output Shape</td>
    <td class="tg-0pky">Param #</td>
  </tr>
  <tr>
    <td class="tg-0pky">1</td>
    <td class="tg-0pky">Conv2D</td>
    <td class="tg-0pky">(None, 26, 26, 32)</td>
    <td class="tg-0pky">320</td>
  </tr>
  <tr>
    <td class="tg-0pky">2</td>
    <td class="tg-0pky">MaxPooling2D</td>
    <td class="tg-0pky">(None, 13, 13, 32)</td>
    <td class="tg-0pky">0</td>
  </tr>
  <tr>
    <td class="tg-0pky">3</td>
    <td class="tg-0pky">Flatten</td>
    <td class="tg-0pky">(None, 5408)</td>
    <td class="tg-0pky">0</td>
  </tr>
  <tr>
    <td class="tg-0pky">4</td>
    <td class="tg-0pky">Dense</td>
    <td class="tg-0pky">(None, 512)</td>
    <td class="tg-0pky">2769408</td>
  </tr>
  <tr>
    <td class="tg-0pky">5</td>
    <td class="tg-0pky">Dense</td>
    <td class="tg-0pky">(None, 128)</td>
    <td class="tg-0pky">65664</td>
  </tr>
  <tr>
    <td class="tg-0pky">6</td>
    <td class="tg-0pky">Dense</td>
    <td class="tg-0pky">(None, 47)</td>
    <td class="tg-0pky">6063</td>
  </tr>
  <tr>
    <td class="tg-0pky" colspan="4">Total params: 2,841,455</td>
  </tr>
  <tr>
    <td class="tg-0pky" colspan="4">Trainable params: 2,841,455</td>
  </tr>
  <tr>
    <td class="tg-0pky" colspan="4">Non-trainable params: 0</td>
  </tr>
</tbody>
</table>       
        
        
## Model Training 
Model training stops when val_accuracy does not improve in three epoches.   
- Model Training with Training Datasets    
![image](https://user-images.githubusercontent.com/12388329/162203470-61e762db-3df4-4e97-8ccf-011c2a40e2a6.png)    
- Model Evaluating with Testing Datasets    
![image](https://user-images.githubusercontent.com/12388329/162204147-87702a81-1a87-4386-85f5-f39003302b84.png)
- <b>Accuracy: 85.30%</b>     
     
     
## Model Prediction   
- Detecting Text Area (OpenCV)    
![image](https://user-images.githubusercontent.com/12388329/162204916-65df22ce-5e70-4af1-a684-e6baaf33bd66.png)    
- Prediction with Saved Model (Model.json / Model.h5)  
![image](https://user-images.githubusercontent.com/12388329/162205070-0411e6a6-2758-4166-86ff-62caef3aa1da.png)



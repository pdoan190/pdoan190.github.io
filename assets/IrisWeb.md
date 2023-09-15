```python
import numpy as np 
from sklearn import datasets

#Import function to create training and test data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import norm
import sympy as sy
from itertools import combinations



```


```python
iris = datasets.load_iris()

```


```python
#Decision/Prediction Variables of classifier model: length, width, plength, pwidth

#Flowers: setosa, versicolor, virginica


#Extract Data of prediction variables 
def ExtractParameterData(flower = str):
    
    flower_length = []
    flower_width = []
    flower_plength = []
    flower_pwidth = []
    
    if flower == 'setosa':
        i = 0
        j = 50
    elif flower == 'versicolor':
        i = 50
        j = 100
    elif flower == 'virginica':
        i = 100
        j = 150

    for element in iris['data'][i:j]:
        #0-3: length, width, plength, pwidth
        flower_length.append(element[0])
        flower_width.append(element[1])
        flower_plength.append(element[2])
        flower_pwidth.append(element[3])
        
    if flower == 'setosa':
        setosa_length = flower_length 
        setosa_width = flower_width
        setosa_plength = flower_plength
        setosa_pwidth = flower_pwidth
        return setosa_length,setosa_width,setosa_plength,setosa_pwidth
        
        
    elif flower == 'versicolor':
        versicolor_length = flower_length 
        versicolor_width = flower_width
        versicolor_plength = flower_plength
        versicolor_pwidth = flower_pwidth
        return versicolor_length,versicolor_width,versicolor_plength,versicolor_pwidth
        
    elif flower == 'virginica':
        virginica_length = flower_length 
        virginica_width = flower_width
        virginica_plength = flower_plength
        virginica_pwidth = flower_pwidth
        return virginica_length,virginica_width,virginica_plength,virginica_pwidth
    
setosa_length,setosa_width,setosa_plength,setosa_pwidth = ExtractParameterData('setosa')

versicolor_length,versicolor_width,versicolor_plength,versicolor_pwidth = ExtractParameterData('versicolor')

virginica_length,virginica_width,virginica_plength,virginica_pwidth = ExtractParameterData('virginica')



```


```python
#Training and test datasets 

x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.4)
```


```python
#Use softmax function as objective function
#Softmax function returns normalized probability distribution. 
def SoftMax(vector):
    return (np.exp(vector)/np.sum(np.exp(vector)))
```


```python
def Deviation(ys_predict, y_given):
        return(np.sum(abs(np.subtract(ys_predict, y_given))))
    

```


```python
#Optimization routine to train model: vary functions AKA model has to change as a result
#Model is function, adjust weights to optimize while training 


#Error function: Use error to train model on the go  
def ErrorFunc(prediction, ):
    #use mean squared error 
    return(np.sum((prediction - ground_truth)**2)/len(prediction))



    
    
```


```python
iris
```




    {'data': array([[5.1, 3.5, 1.4, 0.2],
            [4.9, 3. , 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5. , 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5. , 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [5.4, 3.7, 1.5, 0.2],
            [4.8, 3.4, 1.6, 0.2],
            [4.8, 3. , 1.4, 0.1],
            [4.3, 3. , 1.1, 0.1],
            [5.8, 4. , 1.2, 0.2],
            [5.7, 4.4, 1.5, 0.4],
            [5.4, 3.9, 1.3, 0.4],
            [5.1, 3.5, 1.4, 0.3],
            [5.7, 3.8, 1.7, 0.3],
            [5.1, 3.8, 1.5, 0.3],
            [5.4, 3.4, 1.7, 0.2],
            [5.1, 3.7, 1.5, 0.4],
            [4.6, 3.6, 1. , 0.2],
            [5.1, 3.3, 1.7, 0.5],
            [4.8, 3.4, 1.9, 0.2],
            [5. , 3. , 1.6, 0.2],
            [5. , 3.4, 1.6, 0.4],
            [5.2, 3.5, 1.5, 0.2],
            [5.2, 3.4, 1.4, 0.2],
            [4.7, 3.2, 1.6, 0.2],
            [4.8, 3.1, 1.6, 0.2],
            [5.4, 3.4, 1.5, 0.4],
            [5.2, 4.1, 1.5, 0.1],
            [5.5, 4.2, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.2],
            [5. , 3.2, 1.2, 0.2],
            [5.5, 3.5, 1.3, 0.2],
            [4.9, 3.6, 1.4, 0.1],
            [4.4, 3. , 1.3, 0.2],
            [5.1, 3.4, 1.5, 0.2],
            [5. , 3.5, 1.3, 0.3],
            [4.5, 2.3, 1.3, 0.3],
            [4.4, 3.2, 1.3, 0.2],
            [5. , 3.5, 1.6, 0.6],
            [5.1, 3.8, 1.9, 0.4],
            [4.8, 3. , 1.4, 0.3],
            [5.1, 3.8, 1.6, 0.2],
            [4.6, 3.2, 1.4, 0.2],
            [5.3, 3.7, 1.5, 0.2],
            [5. , 3.3, 1.4, 0.2],
            [7. , 3.2, 4.7, 1.4],
            [6.4, 3.2, 4.5, 1.5],
            [6.9, 3.1, 4.9, 1.5],
            [5.5, 2.3, 4. , 1.3],
            [6.5, 2.8, 4.6, 1.5],
            [5.7, 2.8, 4.5, 1.3],
            [6.3, 3.3, 4.7, 1.6],
            [4.9, 2.4, 3.3, 1. ],
            [6.6, 2.9, 4.6, 1.3],
            [5.2, 2.7, 3.9, 1.4],
            [5. , 2. , 3.5, 1. ],
            [5.9, 3. , 4.2, 1.5],
            [6. , 2.2, 4. , 1. ],
            [6.1, 2.9, 4.7, 1.4],
            [5.6, 2.9, 3.6, 1.3],
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 3. , 4.5, 1.5],
            [5.8, 2.7, 4.1, 1. ],
            [6.2, 2.2, 4.5, 1.5],
            [5.6, 2.5, 3.9, 1.1],
            [5.9, 3.2, 4.8, 1.8],
            [6.1, 2.8, 4. , 1.3],
            [6.3, 2.5, 4.9, 1.5],
            [6.1, 2.8, 4.7, 1.2],
            [6.4, 2.9, 4.3, 1.3],
            [6.6, 3. , 4.4, 1.4],
            [6.8, 2.8, 4.8, 1.4],
            [6.7, 3. , 5. , 1.7],
            [6. , 2.9, 4.5, 1.5],
            [5.7, 2.6, 3.5, 1. ],
            [5.5, 2.4, 3.8, 1.1],
            [5.5, 2.4, 3.7, 1. ],
            [5.8, 2.7, 3.9, 1.2],
            [6. , 2.7, 5.1, 1.6],
            [5.4, 3. , 4.5, 1.5],
            [6. , 3.4, 4.5, 1.6],
            [6.7, 3.1, 4.7, 1.5],
            [6.3, 2.3, 4.4, 1.3],
            [5.6, 3. , 4.1, 1.3],
            [5.5, 2.5, 4. , 1.3],
            [5.5, 2.6, 4.4, 1.2],
            [6.1, 3. , 4.6, 1.4],
            [5.8, 2.6, 4. , 1.2],
            [5. , 2.3, 3.3, 1. ],
            [5.6, 2.7, 4.2, 1.3],
            [5.7, 3. , 4.2, 1.2],
            [5.7, 2.9, 4.2, 1.3],
            [6.2, 2.9, 4.3, 1.3],
            [5.1, 2.5, 3. , 1.1],
            [5.7, 2.8, 4.1, 1.3],
            [6.3, 3.3, 6. , 2.5],
            [5.8, 2.7, 5.1, 1.9],
            [7.1, 3. , 5.9, 2.1],
            [6.3, 2.9, 5.6, 1.8],
            [6.5, 3. , 5.8, 2.2],
            [7.6, 3. , 6.6, 2.1],
            [4.9, 2.5, 4.5, 1.7],
            [7.3, 2.9, 6.3, 1.8],
            [6.7, 2.5, 5.8, 1.8],
            [7.2, 3.6, 6.1, 2.5],
            [6.5, 3.2, 5.1, 2. ],
            [6.4, 2.7, 5.3, 1.9],
            [6.8, 3. , 5.5, 2.1],
            [5.7, 2.5, 5. , 2. ],
            [5.8, 2.8, 5.1, 2.4],
            [6.4, 3.2, 5.3, 2.3],
            [6.5, 3. , 5.5, 1.8],
            [7.7, 3.8, 6.7, 2.2],
            [7.7, 2.6, 6.9, 2.3],
            [6. , 2.2, 5. , 1.5],
            [6.9, 3.2, 5.7, 2.3],
            [5.6, 2.8, 4.9, 2. ],
            [7.7, 2.8, 6.7, 2. ],
            [6.3, 2.7, 4.9, 1.8],
            [6.7, 3.3, 5.7, 2.1],
            [7.2, 3.2, 6. , 1.8],
            [6.2, 2.8, 4.8, 1.8],
            [6.1, 3. , 4.9, 1.8],
            [6.4, 2.8, 5.6, 2.1],
            [7.2, 3. , 5.8, 1.6],
            [7.4, 2.8, 6.1, 1.9],
            [7.9, 3.8, 6.4, 2. ],
            [6.4, 2.8, 5.6, 2.2],
            [6.3, 2.8, 5.1, 1.5],
            [6.1, 2.6, 5.6, 1.4],
            [7.7, 3. , 6.1, 2.3],
            [6.3, 3.4, 5.6, 2.4],
            [6.4, 3.1, 5.5, 1.8],
            [6. , 3. , 4.8, 1.8],
            [6.9, 3.1, 5.4, 2.1],
            [6.7, 3.1, 5.6, 2.4],
            [6.9, 3.1, 5.1, 2.3],
            [5.8, 2.7, 5.1, 1.9],
            [6.8, 3.2, 5.9, 2.3],
            [6.7, 3.3, 5.7, 2.5],
            [6.7, 3. , 5.2, 2.3],
            [6.3, 2.5, 5. , 1.9],
            [6.5, 3. , 5.2, 2. ],
            [6.2, 3.4, 5.4, 2.3],
            [5.9, 3. , 5.1, 1.8]]),
     'target': array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
            2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
     'frame': None,
     'target_names': array(['setosa', 'versicolor', 'virginica'], dtype='<U10'),
     'DESCR': '.. _iris_dataset:\n\nIris plants dataset\n--------------------\n\n**Data Set Characteristics:**\n\n    :Number of Instances: 150 (50 in each of three classes)\n    :Number of Attributes: 4 numeric, predictive attributes and the class\n    :Attribute Information:\n        - sepal length in cm\n        - sepal width in cm\n        - petal length in cm\n        - petal width in cm\n        - class:\n                - Iris-Setosa\n                - Iris-Versicolour\n                - Iris-Virginica\n                \n    :Summary Statistics:\n\n    ============== ==== ==== ======= ===== ====================\n                    Min  Max   Mean    SD   Class Correlation\n    ============== ==== ==== ======= ===== ====================\n    sepal length:   4.3  7.9   5.84   0.83    0.7826\n    sepal width:    2.0  4.4   3.05   0.43   -0.4194\n    petal length:   1.0  6.9   3.76   1.76    0.9490  (high!)\n    petal width:    0.1  2.5   1.20   0.76    0.9565  (high!)\n    ============== ==== ==== ======= ===== ====================\n\n    :Missing Attribute Values: None\n    :Class Distribution: 33.3% for each of 3 classes.\n    :Creator: R.A. Fisher\n    :Donor: Michael Marshall (MARSHALL%PLU@io.arc.nasa.gov)\n    :Date: July, 1988\n\nThe famous Iris database, first used by Sir R.A. Fisher. The dataset is taken\nfrom Fisher\'s paper. Note that it\'s the same as in R, but not as in the UCI\nMachine Learning Repository, which has two wrong data points.\n\nThis is perhaps the best known database to be found in the\npattern recognition literature.  Fisher\'s paper is a classic in the field and\nis referenced frequently to this day.  (See Duda & Hart, for example.)  The\ndata set contains 3 classes of 50 instances each, where each class refers to a\ntype of iris plant.  One class is linearly separable from the other 2; the\nlatter are NOT linearly separable from each other.\n\n.. topic:: References\n\n   - Fisher, R.A. "The use of multiple measurements in taxonomic problems"\n     Annual Eugenics, 7, Part II, 179-188 (1936); also in "Contributions to\n     Mathematical Statistics" (John Wiley, NY, 1950).\n   - Duda, R.O., & Hart, P.E. (1973) Pattern Classification and Scene Analysis.\n     (Q327.D83) John Wiley & Sons.  ISBN 0-471-22361-1.  See page 218.\n   - Dasarathy, B.V. (1980) "Nosing Around the Neighborhood: A New System\n     Structure and Classification Rule for Recognition in Partially Exposed\n     Environments".  IEEE Transactions on Pattern Analysis and Machine\n     Intelligence, Vol. PAMI-2, No. 1, 67-71.\n   - Gates, G.W. (1972) "The Reduced Nearest Neighbor Rule".  IEEE Transactions\n     on Information Theory, May 1972, 431-433.\n   - See also: 1988 MLC Proceedings, 54-64.  Cheeseman et al"s AUTOCLASS II\n     conceptual clustering system finds 3 classes in the data.\n   - Many, many more ...',
     'feature_names': ['sepal length (cm)',
      'sepal width (cm)',
      'petal length (cm)',
      'petal width (cm)'],
     'filename': 'iris.csv',
     'data_module': 'sklearn.datasets.data'}




```python

x = [x for x in range(50)]



#Plot data

#Length

y1_length = np.sort(setosa_length)
y2_length = np.sort(versicolor_length)
y3_length = np.sort(virginica_length)
length = np.array([y1_length,y2_length,y3_length])

plt.plot(x, y1_length, label = "setosa_length")
plt.plot(x, y2_length, label = "versicolor_length")
plt.plot(x, y3_length, label = "virginica_length")

plt.legend()
plt.show()

#Width
y1_width = np.sort(setosa_width)
y2_width = np.sort(versicolor_width)
y3_width = np.sort(virginica_width)
width = np.array([y1_width,y2_width,y3_width])


plt.plot(x, y1_width, label = "setosa_width")
plt.plot(x, y2_width, label = "versicolor_width")
plt.plot(x, y3_width, label = "virginica_width")

plt.legend()
plt.show()

#Plength
y1_plength = np.sort(setosa_plength)
y2_plength = np.sort(versicolor_plength)
y3_plength = np.sort(virginica_plength)
plength = np.array([y1_plength,y2_plength,y3_plength])


plt.plot(x, y1_plength, label = "setosa_plength")
plt.plot(x, y2_plength, label = "versicolor_plength")
plt.plot(x, y3_plength, label = "virginica_plength")

plt.legend()
plt.show()

#Pwidth
y1_pwidth = np.sort(setosa_pwidth)
y2_pwidth = np.sort(versicolor_pwidth)
y3_pwidth = np.sort(virginica_pwidth)
pwidth = np.array([y1_pwidth,y2_pwidth,y3_pwidth])



plt.plot(x, y1_pwidth, label = "setosa_pwidth")
plt.plot(x, y2_pwidth, label = "versicolor_pwidth")
plt.plot(x, y3_pwidth, label = "virginica_pwidth")

plt.legend()
plt.show()








```


    
![png](output_8_0.png)
    



    
![png](output_8_1.png)
    



    
![png](output_8_2.png)
    



    
![png](output_8_3.png)
    



```python
#create polynomial from data basically

#depends on training size
x_coords = np.linspace(0,49,50)

#plot 99th order polyfit 
#where FeatureDataofaFlower is training data
def PolyHundredSum(x_coords,FeatureDataofaFlower,degree = 99):
    #feature e.g. y1_length
    PolyHundredCoefficients = np.polyfit(x_coords,FeatureDataofaFlower,degree)
    j = np.flip(np.linspace(0,degree,100))
    y_coords = np.zeros((50,))
    for i in range(degree + 1):
        y_coords += np.multiply(PolyHundredCoefficients[i],np.power(x_coords,j[i])) 
    return y_coords


#Function to determine deviation of given value and 

def Deviation(ys_predict, y_given):
    
    return(np.sum(abs(np.subtract(ys_predict, y_given))))


    
    

    

features = ['length','width','plength','pwidth'] 
labels = ['setosa','versicolor','virginica']

#features of flowers in order of labels
FeatureData = np.array([length,width,plength,pwidth])


#FeatureData[i][j] where i is feature(l,w,pl,pw) and j is flower
#Plot data (sorted lowest to highest) compared to polynomial function generated


#Extract predicted polynomial values for each combination (feature & flower)
PolynomialyValues = dict()
i = 0

for feature in FeatureData:
    
    if i > 3:
        i = 0
    
    print(f'Category: {features[i]}','\n')
    
  
    j = 0
    for flower in feature:
        
        PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
        plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
        plt.plot(x_coords, flower, label = f"{labels[j] + ' ' + features[i]}",linewidth = 2, color = 'c')
        plt.legend()
        plt.show()
        j +=1
        
    i += 1





```

    Category: length 
    
    

    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_2.png)
    


    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_4.png)
    


    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_6.png)
    


    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    

    Category: width 
    
    


    
![png](output_9_9.png)
    


    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_11.png)
    


    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_13.png)
    


    Category: plength 
    
    

    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_16.png)
    


    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_18.png)
    


    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_20.png)
    


    Category: pwidth 
    
    

    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_23.png)
    


    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_25.png)
    


    C:\Users\doanp\anaconda3\lib\site-packages\numpy\lib\polynomial.py:658: RuntimeWarning: overflow encountered in multiply
      scale = NX.sqrt((lhs*lhs).sum(axis=0))
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:56: RankWarning: Polyfit may be poorly conditioned
      PolynomialyValues[f'{features[i] + " " + labels[j]}'] = PolyHundredSum(x_coords, flower)
    C:\Users\doanp\AppData\Local\Temp\ipykernel_28008\641372883.py:57: RankWarning: Polyfit may be poorly conditioned
      plt.plot(x_coords,PolyHundredSum(x_coords,flower), label = f'{features[i]} of {labels[j]} as 99th polynomial',alpha=0.7,linewidth = 5, color = 'k')
    


    
![png](output_9_27.png)
    


# Final Model


```python
def devTodec(dev, totaldev):
    return(dev/totaldev) 


def IrisMLModel(length,width,plength,pwidth, sampling_size = 50):
    

    
    #Create 'horizontal lines' representing each feaure so it can be compared to the polynomial functions
    h_length = np.full(sampling_size, length)
    h_width = np.full(sampling_size, width)
    h_plength = np.full(sampling_size, plength)
    h_pwidth = np.full(sampling_size, pwidth)
    
    horizontal_lines = [h_length,h_width,h_plength,h_pwidth]
    
    
    feature_horizontal = {features[i]: horizontal_lines[i] for i in range(len(horizontal_lines))}
    
    extended_labels = [(' setosa',' versicolor',' virginica') for x in range(len(features))]
   

 ###################################################################################################################
    
#CATEGORY 1: Classify Based on magnitude of input features

    #use deviation for this, lower deviation means more likely to be that label
     
        
    #Determine Deviation of Horizontal lines from polynomial functions    
    
    for feature, label in zip(features, extended_labels):
        

        set_dev = Deviation(PolynomialyValues[f'{feature + label[0]}'], feature_horizontal[feature]) 

        ver_dev = Deviation(PolynomialyValues[f'{feature + label[1]}'], feature_horizontal[feature])

        vir_dev = Deviation(PolynomialyValues[f'{feature + label[2]}'], feature_horizontal[feature])
        

        total_dev = set_dev + ver_dev + vir_dev
        
        convictions = []
    


        convictions.append({f'{feature + label[0]}': devTodec(set_dev,total_dev),f'{feature + label[1]}': devTodec(ver_dev,total_dev),f'{feature + label[2]}': devTodec(vir_dev,total_dev)})

        
    #Add deviation (between horizontal and polynomial) of each category for a given flower
    setosa_feature_rating, versicolor_feature_rating, virginica_feature_rating = 0,0,0
    
    for conviction in convictions:
        for feature_flower in conviction.keys():

            if 'setosa' in feature_flower:
                setosa_feature_rating += conviction[feature_flower]

            elif 'versicolor' in feature_flower:
                versicolor_feature_rating += conviction[feature_flower]

            elif 'virginica' in feature_flower:
                virginica_feature_rating += conviction[feature_flower]
                
    #LOWER RATING means more likely to be in that label: WE ARE CALCULATING DEVIATION 
                
    feature_ratings = {'setosa': setosa_feature_rating, 'versicolor': versicolor_feature_rating, 'virginica':virginica_feature_rating}

    #Normalize deviation feature ratings, they currently add up to 4
    
    feature_ratings_norm = dict()
    
    for flower,deviation in zip(feature_ratings.keys(),feature_ratings.values()):
    
        feature_ratings_norm[flower] = (deviation/4)
    
    
    #Determine standard deviation of normalized feature deviation ratings 
    std_ratings = np.std(list(feature_ratings_norm.values()))

    #Convert ratings to scores, using probability density function: score is probability that given sample is that flower
    setosa_pdf, versicolor_pdf, virginica_pdf = norm.pdf(feature_ratings_norm['setosa'],scale = std_ratings),norm.pdf(feature_ratings_norm['versicolor'],scale = std_ratings),norm.pdf(feature_ratings_norm['virginica'],scale = std_ratings)

    pdfs = [setosa_pdf, versicolor_pdf, virginica_pdf]


    #Normalize probability densities 

    norm_pdfs = [(1/sum(pdfs)) * pdf for pdf in pdfs]
    
    
#Final Classification 1: Category 1: Based solely off magnitude of features 
    
    Probability_BasedOn_Features = {labels[i]: norm_pdfs[i] for i in range(len(labels))}
    
    

 ################################################################################################################################
    
#CATEGORY 2: classify based on relationship between features within the same flower

    #use covariance for this, higher covariance means they relate more, negative means they are inversely related


    #calculate covariance between feature combinations of same flower
    
    #Compare this calculated covariance with covariance of input features data

    feature_combinations = list(combinations(features, 2))
    

    cov_matrices = []

    for flower in labels:

        for element in feature_combinations:

            cov_matrices.append(np.cov(PolynomialyValues[f'{element[0] + " " + flower}'], PolynomialyValues[f'{element[1] + " " + flower}']))


    #set them equal sizes
    label_cov = [label  for label in labels for _ in range(6)]

    feat_comb_cov = feature_combinations * int((len(cov_matrices) / 6))

    #Organize data to respective grouping
    #This variable represents covariance matrix for each combination of features for each flower, using polynomial values
    label_combo_cov = list(zip(label_cov, feat_comb_cov, cov_matrices))
    
    #[c1,c2] Consider cov_matrix to left: c1 represents variance of first feature/element in respective feature combination tuple
    #[c3,c4] c4 represents var of 2nd feature in feat_comb_cov element, c2 & c3 are the covariance values between the features
    
    

    #Now, to compare the input data and the covariance matrices, I have to generate data from the covariance matrices using a normal dist. 
    #Pretty much going to generate data samples for each covariance matrix and use that distribution 
    #to determine the probability that the input features given lies in each of the datasamples(which represent two features of a given flower)
    
    
   
    
   
    flowers_means = {'setosa':[np.mean(setosa_length),np.mean(setosa_width),np.mean(setosa_plength),np.mean(setosa_pwidth)],'versicolor':[np.mean(versicolor_length),np.mean(versicolor_width),np.mean(versicolor_plength),np.mean(versicolor_pwidth)],'virginica':[np.mean(virginica_length),np.mean(virginica_width),np.mean(virginica_plength),np.mean(virginica_pwidth)]}

    comb_mean_set = list(combinations(flowers_means['setosa'],2))

    comb_mean_ver = list(combinations(flowers_means['versicolor'],2))

    comb_mean_vir = list(combinations(flowers_means['virginica'],2))


    comb_means_set = [(np.array(list(pair)).reshape(2,1)) for pair in comb_mean_set]
    comb_means_ver = [(np.array(list(pair))).reshape(2,1) for pair in comb_mean_ver]
    comb_means_vir = [(np.array(list(pair))).reshape(2,1) for pair in comb_mean_vir]

    comb_means = comb_means_set + comb_means_ver + comb_means_vir

    len(comb_means)

    feature_data_generated = []

    
    ############################################
    dist_samples = np.random.normal(loc = 0, scale = 1, size = 100).reshape(2,50)

    #Generate data values for each set of feature combination based on its covariance matrix
    for comb_mean, cov_matrix in zip(comb_means, cov_matrices):

        #Cholesky Decomposition used to generate data samples from covariance matrix
        L = np.linalg.cholesky(cov_matrix)

        feature_data_generated.append(comb_mean + np.dot(L,dist_samples))



    #Organize generated feature data with corresponding features
    generated_data = zip(label_cov, feat_comb_cov, feature_data_generated)

    set_gendata, ver_gendata, vir_gendata = list(generated_data)[0:6], list(generated_data)[6:12], list(generated_data)[12:18]
    
#calculate mean and standard deviation of generated data


                            
    #Get the mean value pairs for the feature pairs
    #Create function to extract 
    two_feature_means = lambda i, j, k: np.array([np.mean(feature_data_generated[i:j][k][0]),np.mean(feature_data_generated[i:j][k][1])])
    two_feature_stdss = lambda i, j, k: np.array([np.std(feature_data_generated[i:j][k][0]),np.std(feature_data_generated[i:j][k][1])])
                            
    set_gendata_meanpairs = [two_feature_means(0,6,x) for x in range(6)]
    ver_gendata_meanpairs = [two_feature_means(6,12,x) for x in range(6)]
    vir_gendata_meanpairs = [two_feature_means(12,18,x) for x in range(6)]
                            
    set_gendata_stdpairs = [two_feature_stdss(0,6,x) for x in range(6)]
    ver_gendata_stdpairs = [two_feature_stdss(6,12,x) for x in range(6)]
    vir_gendata_stdpairs =  [two_feature_stdss(12,18,x) for x in range(6)]
                            
        
                            
    #use this data to calculate probability density function 
    #create dictionary for input values
    input_data = [length,width,plength,pwidth]
    comb_input_data = np.array(list(combinations(input_data, 2)))
                 
    setosa_mean_std = list(zip(feature_combinations, set_gendata_meanpairs,set_gendata_stdpairs))
    
    ver_mean_std = list(zip(feature_combinations, ver_gendata_meanpairs,ver_gendata_stdpairs))
                 
    vir_mean_std = list(zip(feature_combinations, vir_gendata_meanpairs,vir_gendata_stdpairs))
                            
    
                 
    #with the data organized to their respective counterparts, I can now calculate the probability density function 2 dimensionally

    pdfofinput_to_generated = lambda meanpairs, stdpairs: [norm.pdf(input_feat_combo, loc = mean ,scale = std) for input_feat_combo,mean,std in zip(comb_input_data, meanpairs,stdpairs)]

    
    setosa_features_pdf = pdfofinput_to_generated(set_gendata_meanpairs,set_gendata_stdpairs)
    versicolor_features_pdf = pdfofinput_to_generated(ver_gendata_meanpairs,ver_gendata_stdpairs)
    virginica_features_pdf = pdfofinput_to_generated(vir_gendata_meanpairs,vir_gendata_stdpairs)
                            
    #Higher probability density means higher probability that input flower matches that classification
    
    gen_pdf = {'setosa': np.sum(setosa_features_pdf), 'versicolor': np.sum(versicolor_features_pdf), 'virginica': np.sum(virginica_features_pdf)}

    #This gives the probability that input is a certain flower
                            
    #Final Classification 2.2:
    Category_Two_Probability_One  = {k: 1/sum(gen_pdf.values()) * v for k, v in gen_pdf.items()}
                
                            
                         
                            
     
 ##################################################################################################                            
                            
    #Calculate the mean deviation as well
    #def Inputs_Distance_From_Mean(meanpair): sqrt((x2-x2)^2+(y2-y1))
    #Here Lower value is better as opposed to the pdf function
   
                            
    distancefrommean = lambda flower_gendata_meanpairs: [np.sqrt((comb_input_data[i][0]- flower_gendata_meanpairs[i][0])**2+(comb_input_data[i][1]-flower_gendata_meanpairs[i][1])**2) for i in range(len(comb_input_data))]

#########################

    set_feature_distance = np.sum(distancefrommean(set_gendata_meanpairs))
    ver_feature_distance = np.sum(distancefrommean(ver_gendata_meanpairs))
    vir_feature_distance = np.sum(distancefrommean(vir_gendata_meanpairs))

    feature_mean_distances = [set_feature_distance, ver_feature_distance, vir_feature_distance]
    totalsum = set_feature_distance + ver_feature_distance + vir_feature_distance

    cov_mean_score = {flower: (gendata/totalsum) for flower, gendata in zip(labels, feature_mean_distances)}


    gen_std = np.std(list(cov_mean_score.values()))

    #convert deviation to probability that input is flower
    gen_set_pdf = norm.pdf(cov_mean_score['setosa'],scale = gen_std)
    gen_ver_pdf = norm.pdf(cov_mean_score['versicolor'],scale = gen_std)
    gen_vir_pdf = norm.pdf(cov_mean_score['virginica'],scale = gen_std)

    category_two_pdfs = [gen_set_pdf, gen_ver_pdf, gen_vir_pdf]


    #Normalize probability densities 

    category_two_norm_pdfs = [(1/sum(category_two_pdfs)) * pdf for pdf in category_two_pdfs]


    #Final Classification 2.2: 

    Category_Two_Probability_Two = {labels[i]: category_two_norm_pdfs[i] for i in range(len(labels))}
    
    totalscore = {'setosa':(Probability_BasedOn_Features['setosa'] + Category_Two_Probability_One['setosa'] + Category_Two_Probability_Two['setosa'])/3 ,'versicolor': (Probability_BasedOn_Features['versicolor'] + Category_Two_Probability_One['versicolor'] + Category_Two_Probability_Two['versicolor'])/3,'virginica':(Probability_BasedOn_Features['virginica'] + Category_Two_Probability_One['virginica'] + Category_Two_Probability_Two['virginica'])/3}
    
    return max(totalscore, key = totalscore.get)

                            
'''
    Comment this part out when determining false positives and negatives 
                            
    print(f'Based off feature magnitude deviation, the probability is {Probability_BasedOn_Features}')
    print('\n')
    print(f'Based off covariance feature data relationships, the probability is {Category_Two_Probability_One}')
    print('\n')
    print(f'Based off covariance data mean deviation, the probability is {Category_Two_Probability_Two}')
                            
    print('\n')
                            
    totalscore = {'setosa':(Probability_BasedOn_Features['setosa'] + Category_Two_Probability_One['setosa'] + Category_Two_Probability_Two['setosa'])/3 ,'versicolor': (Probability_BasedOn_Features['versicolor'] + Category_Two_Probability_One['versicolor'] + Category_Two_Probability_Two['versicolor'])/3,'virginica':(Probability_BasedOn_Features['virginica'] + Category_Two_Probability_One['virginica'] + Category_Two_Probability_Two['virginica'])/3}
                            
    print(f'Considering these factors normalized together, {totalscore}')        
'''
                            
    
    



```

    The train results are [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    
    The test results are [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    
    The precisions and recalls of the classifier for the train and test datasets are  0.9444444444444444 and  0.9833333333333333 respectively
    

# TESTING BELOW:


```python


#Determine Results, false negative and false positive 

x_train_results = []

train_results = []





#For training data set
#Converting string output to numbers to compare
for data_set in x_train:
    
    prediction_x_train = (IrisMLModel(data_set[0],data_set[1],data_set[2],data_set[3])).lower().strip()
    
    if prediction_x_train == 'setosa':
        i = 0
        
    elif prediction_x_train == 'versicolor':
        i = 1
        
    else:
        i = 2
        
    x_train_results.append(i)
    
    
#Compare Numbers with given y_train    
for i in range(len(x_train_results)):
    
    if x_train_results[i] != y_train[i]:
        #0 means wrong 
        train_results.append(0)
               
    else:
        #1 means correct
        train_results.append(1)
        
x_test_results = []

test_results = []
 


#For test data set
#Converting string output to numbers to compare
for data_set in x_test:
    
    prediction_x_test = (IrisMLModel(data_set[0],data_set[1],data_set[2],data_set[3])).lower().strip()
    
    if prediction_x_test == 'setosa':
        i = 0
        
    elif prediction_x_test == 'versicolor':
        i = 1
        
    else:
        i = 2
        
    x_test_results.append(i)
    
    
#Compare Numbers with given y_train    
for i in range(len(x_test_results)):
    
    if x_test_results[i] != y_test[i]:
        #0 means wrong 
        
        test_results.append(0)
               
    else:
        #1 means correct
        test_results.append(1)
    

print(f'The train results are {train_results}', end = '\n')    
print('\n')
print(f'The test results are {test_results}', end = '\n')


    
wrongtrain = 0

wrongtest = 0

for result in train_results:
    if result == 0:
        wrongtrain += 1
        

for result in test_results:
    if result == 0:
        wrongtest += 1

precision_train  = (len(train_results) - wrongtrain) / (len(train_results))

recall_train = precision_train


precision_test  = (len(test_results) - wrongtest) / (len(test_results))

recall_test = precision_test

#Since in this case, our classifier either gets the flower wrong or right, false positives are equal to false negatives 

print('\n')
print(f'The precisions and recalls of the classifier for the train and test datasets are  {round(recall_train,2)} and  {round(recall_test,2)} respectively')
```

    The train results are [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    
    The test results are [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    
    The precisions and recalls of the classifier for the train and test datasets are  0.94 and  0.98 respectively
    

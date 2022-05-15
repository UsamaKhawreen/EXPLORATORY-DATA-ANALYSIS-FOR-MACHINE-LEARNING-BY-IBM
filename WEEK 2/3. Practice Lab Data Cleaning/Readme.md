<center>
    <img src="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/assets/logos/SN_web_lightmode.png" width="300" alt="cognitiveclass.ai logo"  />
</center>


# Data Cleaning

Estimated time needed: **45** minutes

Most of the real-world data, that the data scientist work with, are raw data, meaning that it can contain repeated, missing, and irrelevant entries of information. Hence, if this data is used in any machine learning analysis, it will result in low accuracy or incorrect prediction. For this reason, data cleaning, also known as data cleansing, is an important technique that comes prior to any model building.

In this notebook, we will take a look at some of the common data cleaning techniques that data scientists may use to prepare their data for analysis.

## Objectives

After completing this lab you will be able to:

*   Use Log function to transform the data
*   Handle the duplicates
*   Handle the missing values
*   Standardize and normalize the data
*   Handle the outliers


***


## **Setup**


For this lab, we will be using the following libraries:

*   [`pandas`](https://pandas.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) for managing the data.
*   [`numpy`](https://numpy.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) for mathematical operations.
*   [`seaborn`](https://seaborn.pydata.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) for visualizing the data.
*   [`matplotlib`](https://matplotlib.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) for visualizing the data.
*   [`sklearn`](https://scikit-learn.org/stable/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) for machine learning and machine-learning-pipeline related functions.
*   [`scipy`](https://docs.scipy.org/doc/scipy/tutorial/stats.html/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) for statistical computations.


## **Import the required libraries**


The following required modules are pre-installed in the Skills Network Labs environment. However if you run this notebook commands in a different Jupyter environment (e.g. Watson Studio or Ananconda) you will need to install these libraries by removing the `#` sign before `!mamba` in the code cell below.



```python
# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented.
# !mamba install -qy pandas==1.3.4 numpy==1.21.4 seaborn==0.9.0 matplotlib==3.5.0 scikit-learn==0.20.1
# Note: If your environment doesn't support "!mamba install", use "!pip install"
```


```python
import warnings 
warnings.filterwarnings('ignore')
```


```python
import pandas as pd
import numpy as np 

import seaborn as sns 
import matplotlib.pylab as plt
%matplotlib inline

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import norm
from scipy import stats
```

    /home/jupyterlab/conda/envs/python/lib/python3.7/site-packages/sklearn/utils/validation.py:37: DeprecationWarning: distutils Version classes are deprecated. Use packaging.version instead.
      LARGE_SPARSE_SUPPORTED = LooseVersion(scipy_version) >= '0.14.0'


## **Reading and understanding our data**


For this lab, we will be using the Ames_Housing_Data.tsv file, hosted on IBM Cloud object storage. The Ames housing dataset examines features of houses sold in Ames (a small city in the state of Iowa in the United States) during the 2006–2010 timeframe.


Let's read the data into *pandas* data frame and look at the first 5
rows using the `head()` method.



```python
housing = pd.read_csv("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/Ames_Housing_Data1.tsv", sep='\t')
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>RL</td>
      <td>93.0</td>
      <td>11160</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>



We can find more information about the features and types using the `info()`  method.



```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2931 entries, 0 to 2930
    Data columns (total 82 columns):
     #   Column           Non-Null Count  Dtype  
    ---  ------           --------------  -----  
     0   Order            2931 non-null   int64  
     1   PID              2931 non-null   int64  
     2   MS SubClass      2931 non-null   int64  
     3   MS Zoning        2931 non-null   object 
     4   Lot Frontage     2441 non-null   float64
     5   Lot Area         2931 non-null   int64  
     6   Street           2931 non-null   object 
     7   Alley            198 non-null    object 
     8   Lot Shape        2931 non-null   object 
     9   Land Contour     2931 non-null   object 
     10  Utilities        2931 non-null   object 
     11  Lot Config       2931 non-null   object 
     12  Land Slope       2931 non-null   object 
     13  Neighborhood     2931 non-null   object 
     14  Condition 1      2931 non-null   object 
     15  Condition 2      2931 non-null   object 
     16  Bldg Type        2931 non-null   object 
     17  House Style      2931 non-null   object 
     18  Overall Qual     2931 non-null   int64  
     19  Overall Cond     2931 non-null   int64  
     20  Year Built       2931 non-null   int64  
     21  Year Remod/Add   2931 non-null   int64  
     22  Roof Style       2931 non-null   object 
     23  Roof Matl        2931 non-null   object 
     24  Exterior 1st     2931 non-null   object 
     25  Exterior 2nd     2931 non-null   object 
     26  Mas Vnr Type     2908 non-null   object 
     27  Mas Vnr Area     2908 non-null   float64
     28  Exter Qual       2931 non-null   object 
     29  Exter Cond       2931 non-null   object 
     30  Foundation       2931 non-null   object 
     31  Bsmt Qual        2851 non-null   object 
     32  Bsmt Cond        2851 non-null   object 
     33  Bsmt Exposure    2848 non-null   object 
     34  BsmtFin Type 1   2851 non-null   object 
     35  BsmtFin SF 1     2930 non-null   float64
     36  BsmtFin Type 2   2850 non-null   object 
     37  BsmtFin SF 2     2930 non-null   float64
     38  Bsmt Unf SF      2930 non-null   float64
     39  Total Bsmt SF    2930 non-null   float64
     40  Heating          2931 non-null   object 
     41  Heating QC       2931 non-null   object 
     42  Central Air      2931 non-null   object 
     43  Electrical       2930 non-null   object 
     44  1st Flr SF       2931 non-null   int64  
     45  2nd Flr SF       2931 non-null   int64  
     46  Low Qual Fin SF  2931 non-null   int64  
     47  Gr Liv Area      2931 non-null   int64  
     48  Bsmt Full Bath   2929 non-null   float64
     49  Bsmt Half Bath   2929 non-null   float64
     50  Full Bath        2931 non-null   int64  
     51  Half Bath        2931 non-null   int64  
     52  Bedroom AbvGr    2931 non-null   int64  
     53  Kitchen AbvGr    2931 non-null   int64  
     54  Kitchen Qual     2931 non-null   object 
     55  TotRms AbvGrd    2931 non-null   int64  
     56  Functional       2931 non-null   object 
     57  Fireplaces       2931 non-null   int64  
     58  Fireplace Qu     1509 non-null   object 
     59  Garage Type      2774 non-null   object 
     60  Garage Yr Blt    2772 non-null   float64
     61  Garage Finish    2772 non-null   object 
     62  Garage Cars      2930 non-null   float64
     63  Garage Area      2930 non-null   float64
     64  Garage Qual      2772 non-null   object 
     65  Garage Cond      2772 non-null   object 
     66  Paved Drive      2931 non-null   object 
     67  Wood Deck SF     2931 non-null   int64  
     68  Open Porch SF    2931 non-null   int64  
     69  Enclosed Porch   2931 non-null   int64  
     70  3Ssn Porch       2931 non-null   int64  
     71  Screen Porch     2931 non-null   int64  
     72  Pool Area        2931 non-null   int64  
     73  Pool QC          13 non-null     object 
     74  Fence            572 non-null    object 
     75  Misc Feature     106 non-null    object 
     76  Misc Val         2931 non-null   int64  
     77  Mo Sold          2931 non-null   int64  
     78  Yr Sold          2931 non-null   int64  
     79  Sale Type        2931 non-null   object 
     80  Sale Condition   2931 non-null   object 
     81  SalePrice        2931 non-null   int64  
    dtypes: float64(11), int64(28), object(43)
    memory usage: 1.8+ MB


According to the output above, we have 2930 entries, 0 to 2929, as well as 81 features. The "Non-Null Count" column shows the number of non-null entries.  If the count is 2930 then there is no missing values for that particular feature. 'SalePrice' is our target or response variable and the rest of the features are our predictor variables.

We also have a mix of numerical (28 int64 and 11 float64) and object data types.


Next, let's use the `describe()` function to show the count, mean, min, max of the sale price attribute.



```python
housing["SalePrice"].describe()
```




    count      2931.000000
    mean     180807.729785
    std       79875.557267
    min       12789.000000
    25%      129500.000000
    50%      160000.000000
    75%      213500.000000
    max      755000.000000
    Name: SalePrice, dtype: float64



From the above analysis, it is important to note that the minimum value is greater than 0. Also, there is a big difference between the minimum value and the 25th percentile. It is bigger than the 75th percentile and the maximum value. This means that our data might not be normally distributed (an important assumption for linear regression analysis), so will check for normality in the Log Transform section.


## Exercise 1


The `describe()` function reveals the statistical information about the numeric attributes. To reveal some information about our categorical (object) attributes, we can use `value_counts()` function. In this exercise, describe all categories of the 'Sale Condition' attribute.



```python
housing.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Overall Qual</th>
      <th>Overall Cond</th>
      <th>Year Built</th>
      <th>Year Remod/Add</th>
      <th>Mas Vnr Area</th>
      <th>...</th>
      <th>Wood Deck SF</th>
      <th>Open Porch SF</th>
      <th>Enclosed Porch</th>
      <th>3Ssn Porch</th>
      <th>Screen Porch</th>
      <th>Pool Area</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2931.000000</td>
      <td>2.931000e+03</td>
      <td>2931.000000</td>
      <td>2441.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2908.000000</td>
      <td>...</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
      <td>2931.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1465.000341</td>
      <td>7.144003e+08</td>
      <td>57.374616</td>
      <td>69.253994</td>
      <td>10155.298874</td>
      <td>6.094848</td>
      <td>5.562948</td>
      <td>1971.352439</td>
      <td>1984.258274</td>
      <td>101.900275</td>
      <td>...</td>
      <td>93.791539</td>
      <td>47.538383</td>
      <td>23.003753</td>
      <td>2.591607</td>
      <td>15.996588</td>
      <td>2.242579</td>
      <td>50.617878</td>
      <td>6.215626</td>
      <td>2007.791198</td>
      <td>180807.729785</td>
    </tr>
    <tr>
      <th>std</th>
      <td>846.250554</td>
      <td>1.887306e+08</td>
      <td>42.636341</td>
      <td>23.405675</td>
      <td>7888.789128</td>
      <td>1.410786</td>
      <td>1.111396</td>
      <td>30.240926</td>
      <td>20.861542</td>
      <td>179.081899</td>
      <td>...</td>
      <td>126.358242</td>
      <td>67.472412</td>
      <td>64.129522</td>
      <td>25.137086</td>
      <td>56.078577</td>
      <td>35.591130</td>
      <td>566.248407</td>
      <td>2.714122</td>
      <td>1.317021</td>
      <td>79875.557267</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>5.263011e+08</td>
      <td>20.000000</td>
      <td>21.000000</td>
      <td>1300.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1872.000000</td>
      <td>1950.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>2006.000000</td>
      <td>12789.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>732.500000</td>
      <td>5.284770e+08</td>
      <td>20.000000</td>
      <td>58.000000</td>
      <td>7440.500000</td>
      <td>5.000000</td>
      <td>5.000000</td>
      <td>1954.000000</td>
      <td>1965.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>2007.000000</td>
      <td>129500.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1465.000000</td>
      <td>5.354532e+08</td>
      <td>50.000000</td>
      <td>68.000000</td>
      <td>9439.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>1973.000000</td>
      <td>1993.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>2008.000000</td>
      <td>160000.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2197.500000</td>
      <td>9.071811e+08</td>
      <td>70.000000</td>
      <td>80.000000</td>
      <td>11556.000000</td>
      <td>7.000000</td>
      <td>6.000000</td>
      <td>2001.000000</td>
      <td>2004.000000</td>
      <td>164.000000</td>
      <td>...</td>
      <td>168.000000</td>
      <td>70.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.000000</td>
      <td>2009.000000</td>
      <td>213500.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2930.000000</td>
      <td>1.007100e+09</td>
      <td>190.000000</td>
      <td>313.000000</td>
      <td>215245.000000</td>
      <td>10.000000</td>
      <td>9.000000</td>
      <td>2010.000000</td>
      <td>2010.000000</td>
      <td>1600.000000</td>
      <td>...</td>
      <td>1424.000000</td>
      <td>742.000000</td>
      <td>1012.000000</td>
      <td>508.000000</td>
      <td>576.000000</td>
      <td>800.000000</td>
      <td>17000.000000</td>
      <td>12.000000</td>
      <td>2010.000000</td>
      <td>755000.000000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 39 columns</p>
</div>




```python
housing['Sale Condition'].value_counts()
```




    Normal     2414
    Partial     245
    Abnorml     190
    Family       46
    Alloca       24
    AdjLand      12
    Name: Sale Condition, dtype: int64




```python
housing['Sale Type'].value_counts()
```




    WD       2537
    New       239
    COD        87
    ConLD      26
    CWD        12
    ConLI       9
    ConLw       8
    Oth         7
    Con         5
    VWD         1
    Name: Sale Type, dtype: int64



<details>
<summary><strong>Solution</strong> (Click Here)</summary>
    &emsp; &emsp; <code>
 housing["Sale Condition"].value_counts()
</code>
</details>


## **Looking for Correlations**


Before proceeding with the data cleaning, it is useful to establish a correlation between the response variable (in our case the sale price) and other predictor variables, as some of them might not have any major impact in determining the price of the house and will not be used in the analysis.  There are many ways to discover correlation between the target variable and the rest of the features. Building pair plots, scatter plots, heat maps, and a correlation matrixes are the most common ones. Below, we will use the `corr()` function to list the top features based on the [pearson correlation coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) (measures how closely two sequences of numbers are correlated). Correlation coefficient can only be calculated on the numerical attributes (floats and integers), therefore, only the numeric attributes will be selected.



```python
hous_num = housing.select_dtypes(include = ['float64', 'int64'])
hous_num_corr = hous_num.corr()['SalePrice'][:-1] # -1 means that the latest row is SalePrice
top_features = hous_num_corr[abs(hous_num_corr) > 0.5].sort_values(ascending=False) #displays pearsons correlation coefficient greater than 0.5
print("There is {} strongly correlated values with SalePrice:\n{}".format(len(top_features), top_features))
```

    There is 11 strongly correlated values with SalePrice:
    Overall Qual      0.799226
    Gr Liv Area       0.706791
    Garage Cars       0.647891
    Garage Area       0.640411
    Total Bsmt SF     0.632270
    1st Flr SF        0.621672
    Year Built        0.558340
    Full Bath         0.545339
    Year Remod/Add    0.532664
    Garage Yr Blt     0.526808
    Mas Vnr Area      0.508277
    Name: SalePrice, dtype: float64


Above, there are 11 features, with coefficients greater than 0.5, that are strongly correlated with the sale price.


Next, let's generate some pair plots to visually inspect the correlation between some of these features and the target variable. We will use seaborns `sns.pairplot()` function for this analysis. Also, building pair plots is one of the possible ways to spot the outliers that might be present in the data.



```python
for i in range(0, len(hous_num.columns), 5):
    sns.pairplot(data=hous_num,
                x_vars=hous_num.columns[i:i+5],
                y_vars=['SalePrice'])
```


    
![png](pics/output_31_0.png)
    



    
![png](pics/output_31_1.png)
    



    
![png](pics/output_31_2.png)
    



    
![png](pics/output_31_3.png)
    



    
![png](pics/output_31_4.png)
    



    
![png](pics/output_31_5.png)
    



    
![png](pics/output_31_6.png)
    



    
![png](pics/output_31_7.png)
    


From Pearsons Correlation Coefficients and pair plots, we can draw some conclusions about the features that are most strongly correlated to the 'SalePrice'. They are: 'Overall Qual', 'Gr Liv Area', 'Garage Cars', 'Garage Area', and others.


## **Log Transformation**


In this section, we are going to inspect whether our 'SalePrice' data are normally distributed. The assumption of the normal distribution must be met in order to perform any type of regression analysis. There are several ways to check for this assumption, however here, we will use the visual method, by plotting the 'SalePrice' distribution using the `distplot()` function from the `seaborn` library.



```python
sp_untransformed = sns.distplot(housing['SalePrice'])
```


    
![png](pics/output_35_0.png)
    


As the plot shows, our 'SalePrice' deviates from the normal distribution. It has a longer tail to the right, so we call it a positive skew. In statistics *skewness* is a measure of asymmetry of the distribution. In addition to skewness, there is also a kurtosis, parameter which refers to the pointedness of a peak in the distribution curve. Both skewness and kurtosis are frequently used together to characterize the distribution of data.


Here, we can simply use the `skew()` function to calculate our skewness level of the `SalePrice`.



```python
print("Skewness: %f" % housing['SalePrice'].skew())
```

    Skewness: 1.743222


The range of skewness for a fairly symmetrical bell curve distribution is between -0.5 and 0.5; moderate skewness is -0.5 to -1.0 and 0.5 to 1.0; and highly skewed distribution is < -1.0 and > 1.0. In our case, we have \~1.7, so it is considered  highly skewed data.

Now, we can try to transform our data, so it looks more normally distributed. We can use the `np.log()` function from the `numpy` library to perform log transform. This [documentation](https://numpy.org/doc/stable/reference/generated/numpy.log.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) contains more information about the numpy log transform.



```python
log_transformed = np.log(housing['SalePrice'])
```


```python
sp_transformed = sns.distplot(log_transformed)
```


    
![png](pics/output_41_0.png)
    



```python
print("Skewness: %f" % (log_transformed).skew())
```

    Skewness: -0.015354


As we can see, the log method transformed the 'SalePrice' distribution into a more symmetrical bell curve and the skewness level now is -0.01, well within the range.

There are other ways to correct for skewness of the data. For example, Square Root Transform (`np.sqrt`) and the Box-Cox Transform (`stats.boxcox` from the `scipy stats` library). To learn more about these two methods, please check out this [article](https://towardsdatascience.com/top-3-methods-for-handling-skewed-data-1334e0debf45?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01).


## Exercise 2


In this exercise, visually inspect the 'Lot Area' feature. If there is any skewness present, apply log transform to make it more normally distributed.



```python
la_untransformed = sns.distplot(housing['Lot Area'])
```


    
![png](pics/output_46_0.png)
    



```python
print('Skewness: %f' % housing['Lot Area'].skew())
```

    Skewness: 12.778041



```python
log_la_transform = np.log(housing['Lot Area'])

la_transformed = sns.distplot(log_la_transform)
```


    
![png](pics/output_48_0.png)
    



```python
print('Skewness: %f' % log_la_transform.skew())
```

    Skewness: -0.494639


<details>
<summary><strong>Solution</strong> (Click Here)</summary>
    &emsp; &emsp; <code>
la_plot = sns.distplot(housing['Lot Area'])
print("Skewness: %f" % housing['Lot Area'].skew())
la_log = np.log(housing['Lot Area'])
print("Skewness: %f" % la_log.skew())

</code>
</details>


## **Handling the Duplicates**


As mentioned in the video, having duplicate values can effect our analysis, so it is good to check whether there are any duplicates in our data. We will use pandas `duplicated()` function and search by the 'PID' column, which contains a unique index number for each entry.



```python
duplicate = housing[housing.duplicated(['PID'])]
duplicate
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 82 columns</p>
</div>




```python
housing.where(housing['PID'] == 526301100).dropna(how='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>526301100.0</td>
      <td>20.0</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770.0</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2010.0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>526301100.0</td>
      <td>20.0</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770.0</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2010.0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 82 columns</p>
</div>



As we can see, there is one duplicate row in this dataset. To remove it, we can use pandas `drop_duplicates()` function. By default, it removes all duplicate rows based on all the columns.



```python
dup_removed = housing.drop_duplicates()
dup_removed 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>RL</td>
      <td>93.0</td>
      <td>11160</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>527105010</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>189900</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>2926</td>
      <td>923275080</td>
      <td>80</td>
      <td>RL</td>
      <td>37.0</td>
      <td>7937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142500</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>2927</td>
      <td>923276100</td>
      <td>20</td>
      <td>RL</td>
      <td>NaN</td>
      <td>8885</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Low</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>2928</td>
      <td>923400125</td>
      <td>85</td>
      <td>RL</td>
      <td>62.0</td>
      <td>10441</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>2929</td>
      <td>924100070</td>
      <td>20</td>
      <td>RL</td>
      <td>77.0</td>
      <td>10010</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>170000</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>2930</td>
      <td>924151050</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>9627</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>188000</td>
    </tr>
  </tbody>
</table>
<p>2930 rows × 82 columns</p>
</div>



An alternative way to check if there are any duplicated Indexes in our dataset is using `index.is_unique` function.



```python
housing.index.is_unique
```




    True



## Exercise 3


In this exercise try to remove duplicates on a specific column by setting the subset equal to the column that contains the duplicate, such as 'Order'.



```python
housing[housing['Order'].duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 82 columns</p>
</div>




```python
housing.where(housing['Order'] == 1).dropna(how='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>526301100.0</td>
      <td>20.0</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770.0</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2010.0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>526301100.0</td>
      <td>20.0</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770.0</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>2010.0</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000.0</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 82 columns</p>
</div>




```python
drop_duplication = housing.drop_duplicates(subset=['Order'])
drop_duplication
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>RL</td>
      <td>93.0</td>
      <td>11160</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>527105010</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>189900</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>2926</td>
      <td>923275080</td>
      <td>80</td>
      <td>RL</td>
      <td>37.0</td>
      <td>7937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142500</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>2927</td>
      <td>923276100</td>
      <td>20</td>
      <td>RL</td>
      <td>NaN</td>
      <td>8885</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Low</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>2928</td>
      <td>923400125</td>
      <td>85</td>
      <td>RL</td>
      <td>62.0</td>
      <td>10441</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>2929</td>
      <td>924100070</td>
      <td>20</td>
      <td>RL</td>
      <td>77.0</td>
      <td>10010</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>170000</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>2930</td>
      <td>924151050</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>9627</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>188000</td>
    </tr>
  </tbody>
</table>
<p>2930 rows × 82 columns</p>
</div>



<details>
<summary><strong>Solution</strong> (Click Here)</summary>
    &emsp; &emsp; <code>
removed_sub = housing.drop_duplicates(subset=['Order'])
</code>
</details>


## **Handling the Missing Values**


### Finding the Missing Values


For easier detection of missing values, pandas provides the `isna()`, `isnull()`, and `notna()` functions. For more information on pandas missing values please check out this [documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01).


To summarize all the missing values in our dataset, we will use `isnull()` function. Then, we will add them all up, by using `sum()` function, sort them with `sort_values()` function, and plot the first 20 columns (as the majority of our missing values fall within first 20 columns), using the `bar plot` function from the `matplotlib` library.



```python
total = housing.isnull().sum().sort_values(ascending=False)
total_select = total.head(20)
total_select.plot(kind="bar", figsize = (8,6), fontsize = 10)

plt.xlabel("Columns", fontsize = 20)
plt.ylabel("Count", fontsize = 20)
plt.title("Total Missing Values", fontsize = 20)
```




    Text(0.5, 1.0, 'Total Missing Values')




    
![png](pics/output_69_1.png)
    


There are several options for dealing with missing values. We will use 'Lot Frontage' feature to analyze for missing values.


1.  We can drop the missing values, using `dropna()` method.



```python
housing.dropna(subset=["Lot Frontage"])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>RL</td>
      <td>93.0</td>
      <td>11160</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2925</th>
      <td>2925</td>
      <td>923251180</td>
      <td>20</td>
      <td>RL</td>
      <td>160.0</td>
      <td>20000</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>2926</td>
      <td>923275080</td>
      <td>80</td>
      <td>RL</td>
      <td>37.0</td>
      <td>7937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142500</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>2928</td>
      <td>923400125</td>
      <td>85</td>
      <td>RL</td>
      <td>62.0</td>
      <td>10441</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>2929</td>
      <td>924100070</td>
      <td>20</td>
      <td>RL</td>
      <td>77.0</td>
      <td>10010</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>170000</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>2930</td>
      <td>924151050</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>9627</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>188000</td>
    </tr>
  </tbody>
</table>
<p>2441 rows × 82 columns</p>
</div>



Using this method, all the rows, containing null values in 'Lot Frontage' feature, for example, will be dropped.


2.  We can drop the whole attribute (column), that contains missing values, using the `drop()` method.



```python
housing.drop("Lot Frontage", axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>RH</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>RL</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>RL</td>
      <td>11160</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>2926</td>
      <td>923275080</td>
      <td>80</td>
      <td>RL</td>
      <td>7937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142500</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>2927</td>
      <td>923276100</td>
      <td>20</td>
      <td>RL</td>
      <td>8885</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Low</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>2928</td>
      <td>923400125</td>
      <td>85</td>
      <td>RL</td>
      <td>10441</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>2929</td>
      <td>924100070</td>
      <td>20</td>
      <td>RL</td>
      <td>10010</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>170000</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>2930</td>
      <td>924151050</td>
      <td>60</td>
      <td>RL</td>
      <td>9627</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>188000</td>
    </tr>
  </tbody>
</table>
<p>2931 rows × 81 columns</p>
</div>



Using this method, the entire column containing the null values will be dropped.


3.  We can replace the missing values (zero, the mean, the median, etc.), using `fillna()` method.



```python
median = housing["Lot Frontage"].median()
median
```




    68.0




```python
housing["Lot Frontage"].fillna(median, inplace = True)
```


```python
housing.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2926</th>
      <td>2926</td>
      <td>923275080</td>
      <td>80</td>
      <td>RL</td>
      <td>37.0</td>
      <td>7937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142500</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>2927</td>
      <td>923276100</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>8885</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Low</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>2928</td>
      <td>923400125</td>
      <td>85</td>
      <td>RL</td>
      <td>62.0</td>
      <td>10441</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>2929</td>
      <td>924100070</td>
      <td>20</td>
      <td>RL</td>
      <td>77.0</td>
      <td>10010</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>170000</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>2930</td>
      <td>924151050</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>9627</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>188000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 82 columns</p>
</div>



Index# 2927, containing a missing value in the "Lot Frontage", now has been replaced with the median value.


## Exercise 4


In this exercise, let's look at 'Mas Vnr Area' feature and replace the missing values with the mean value of that column.



```python
housing['Mas Vnr Area'].isnull().sum()
```




    23




```python
mean = housing['Mas Vnr Area'].mean()
mean
```




    101.9002751031637




```python
housing['Mas Vnr Area'].fillna(mean, inplace=True)
housing['Mas Vnr Area']
```




    0       112.0
    1       112.0
    2         0.0
    3       108.0
    4         0.0
            ...  
    2926      0.0
    2927      0.0
    2928      0.0
    2929      0.0
    2930     94.0
    Name: Mas Vnr Area, Length: 2931, dtype: float64



<details>
<summary><strong>Solution</strong> (Click Here)</summary>
    &emsp; &emsp; <code>
mean = housing["Mas Vnr Area"].mean()
housing["Mas Vnr Area"].fillna(mean, inplace = True)   
</code>
</details>


## **Feature Scaling**


One of the most important transformations we need to apply to our data is feature scaling.  There are two common ways to get all attributes to have the same scale: min-max scaling and standardization.

Min-max scaling (or normalization) is the simplest: values are shifted and rescaled so they end up ranging from 0 to 1. This is done by subtracting the min value and dividing by the max minus min.

Standardization is different: first it subtracts the mean value (so standardized values always have a zero mean), and then it divides by the standard deviation, so that the resulting distribution has unit variance.


Scikit-learn library provides `MinMaxScaler` for normalization and `StandardScaler` for standardization needs. For more information on `scikit-learn` [`MinMaxScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) and [`StandardScaler`](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) please visit their respective documentation websites.


First, we will normalize our data.



```python
norm_data = MinMaxScaler().fit_transform(hous_num)
norm_data
```




    array([[0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
            3.63636364e-01, 1.00000000e+00, 2.72444089e-01],
           [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, ...,
            3.63636364e-01, 1.00000000e+00, 2.72444089e-01],
           [3.41413452e-04, 1.01788895e-04, 0.00000000e+00, ...,
            4.54545455e-01, 1.00000000e+00, 1.24238256e-01],
           ...,
           [9.99317173e-01, 8.25914814e-01, 3.82352941e-01, ...,
            5.45454545e-01, 0.00000000e+00, 1.60616051e-01],
           [9.99658587e-01, 8.27370610e-01, 0.00000000e+00, ...,
            2.72727273e-01, 0.00000000e+00, 2.11814430e-01],
           [1.00000000e+00, 8.27476641e-01, 2.35294118e-01, ...,
            9.09090909e-01, 0.00000000e+00, 2.36066294e-01]])



Note the data is now a `ndarray`


we can also standardize our data.



```python
scaled_data = StandardScaler().fit_transform(hous_num)
scaled_data
```




    array([[-1.73027969, -0.99682434, -0.87674019, ..., -0.44796566,
             1.67740664,  0.4281423 ],
           [-1.73027969, -0.99682434, -0.87674019, ..., -0.44796566,
             1.67740664,  0.4281423 ],
           [-1.72909781, -0.99656498, -0.87674019, ..., -0.07945953,
             1.67740664, -0.94923488],
           ...,
           [ 1.729097  ,  1.10758639,  0.64804102, ...,  0.2890466 ,
            -1.36026952, -0.61115139],
           [ 1.73027889,  1.11129572, -0.87674019, ..., -0.81647179,
            -1.36026952, -0.13533019],
           [ 1.73146077,  1.11156589,  0.06158671, ...,  1.76307112,
            -1.36026952,  0.09005881]])



## Exercise 5


In this exercise, use `StandardScaler()` and `fit_transform()` functions to standardize the 'SalePrice' feature only.



```python
housing['SalePrice'][:,np.newaxis]
```




    array([[215000],
           [215000],
           [105000],
           ...,
           [132000],
           [170000],
           [188000]])




```python
standard_salePrice = StandardScaler().fit_transform(housing['SalePrice'][:,np.newaxis])
standard_salePrice
```




    array([[ 0.4281423 ],
           [ 0.4281423 ],
           [-0.94923488],
           ...,
           [-0.61115139],
           [-0.13533019],
           [ 0.09005881]])



<details>
<summary><strong>Solution</strong> (Click Here)</summary>
    &emsp; &emsp; <code>
scaled_sprice = StandardScaler().fit_transform(housing['SalePrice'][:,np.newaxis]) 
scaled_sprice
</code>
</details>


## **Handling the Outliers**


### Finding the Outliers


In statistics, an outlier is an observation point that is distant from other observations. An outlier can be due to some mistakes in data collection or recording, or due to natural high variability of data points. How to treat an outlier highly depends on our data or the type of analysis to be performed. Outliers can markedly affect our models and can be a valuable source of information, providing us insights about specific behaviours.

There are many ways to discover outliers in our data. We can do Uni-variate analysis (using one variable analysis) or Multi-variate analysis (using two or more variables). One of the simplest ways to detect an outlier is to inspect the data visually, by making box plots or scatter plots.


### Uni-variate Analysis


A box plot is a method for graphically depicting groups of numerical data through their quartiles. Box plots may also have lines extending vertically from the boxes (whiskers) indicating variability outside the upper and lower quartiles. Outliers may be plotted as individual points. To learn more about box plots please click [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.boxplot.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01).


Here, we will use a box plot for the 'Lot Area' and the 'SalePrice' features.



```python
sns.boxplot(x=housing['Lot Area'])
```




    <AxesSubplot:xlabel='Lot Area'>




    
![png](pics/output_107_1.png)
    



```python
sns.boxplot(x=housing['SalePrice'])
```




    <AxesSubplot:xlabel='SalePrice'>




    
![png](pics/output_108_1.png)
    


As we can see from these two plots, we have some points that are plotted outside the box plot area and that greatly deviate from the rest of the population. Whether to remove or keep them will greatly depend on the understanding of our data and the type of analysis to be performed. In this case, the points that are outside of our box plots in the 'Lot Area' and the 'Sale Price' might be the actual true data points and do not need to be removed.


### Bi-variate Analysis


Next, we will look at the bi-variate analysis of the two features, the sale price, 'SalePrice', and the ground living area, 'GrLivArea', and plot the scatter plot of the relationship between these two parameters.



```python
price_area = housing.plot.scatter(x='Gr Liv Area',
                      y='SalePrice')
```


    
![png](pics/output_112_0.png)
    


From the above graph, there are two values above 5000 sq. ft. living area that deviate from the rest of the population and do not seem to follow the trend. It can be speculated why this is happening but for the purpose of this lab we can delete them.

The other two observations on the top are also deviating from the rest of the points but they also seem to be following the trend, so, perhaps, they can be kept.


### Deleting the Outliers


First, we will sort all of our 'Gr Liv Area' values and select only the last two.



```python
housing.sort_values(by = 'Gr Liv Area', ascending = False)[:2]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1499</th>
      <td>1499</td>
      <td>908154235</td>
      <td>60</td>
      <td>RL</td>
      <td>313.0</td>
      <td>63887</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR3</td>
      <td>Bnk</td>
      <td>...</td>
      <td>480</td>
      <td>Gd</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2008</td>
      <td>New</td>
      <td>Partial</td>
      <td>160000</td>
    </tr>
    <tr>
      <th>2181</th>
      <td>2181</td>
      <td>908154195</td>
      <td>20</td>
      <td>RL</td>
      <td>128.0</td>
      <td>39290</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Bnk</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Elev</td>
      <td>17000</td>
      <td>10</td>
      <td>2007</td>
      <td>New</td>
      <td>Partial</td>
      <td>183850</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 82 columns</p>
</div>



Now we will use the pandas `drop()` function to remove these two rows.



```python
outliers_dropped = housing.drop(housing.index[[1499,2181]])
```


```python
new_plot = outliers_dropped.plot.scatter(x='Gr Liv Area',
                                         y='SalePrice')
```


    
![png](pics/output_119_0.png)
    


As you can see, we do not have the last two points of the 'Gr Liv Area' anymore.


## Exercise 6


In this exercise, determine whether there are any outliers in the 'Lot Area' feature. You can either plot the box plot for the 'Lot Area', perform a bi-variate analysis by making a scatter plot between the 'SalePrice' and the 'Lot Area', or use the Z-score analysis. If there re any outliers, remove them from the dataset.



```python
sns.boxplot(housing['Lot Area'])
```




    <AxesSubplot:xlabel='Lot Area'>




    
![png](pics/output_123_1.png)
    



```python
sns.scatterplot(data = housing, x='Lot Area', y='SalePrice')
```




    <AxesSubplot:xlabel='Lot Area', ylabel='SalePrice'>




    
![png](pics/output_124_1.png)
    



```python
housing['Lot_Area_Stats'] = stats.zscore(housing['Lot Area'])
housing[['Lot Area','Lot_Area_Stats']].describe().round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Lot Area</th>
      <th>Lot_Area_Stats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2931.000</td>
      <td>2931.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>10155.299</td>
      <td>-0.000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7888.789</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1300.000</td>
      <td>-1.123</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>7440.500</td>
      <td>-0.344</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>9439.000</td>
      <td>-0.091</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>11556.000</td>
      <td>0.178</td>
    </tr>
    <tr>
      <th>max</th>
      <td>215245.000</td>
      <td>26.002</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.sort_values(by = 'Lot Area', ascending=True)[-4:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2072</th>
      <td>2072</td>
      <td>905301050</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>115149</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR2</td>
      <td>Low</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>302000</td>
    </tr>
    <tr>
      <th>2116</th>
      <td>2116</td>
      <td>906426060</td>
      <td>50</td>
      <td>RL</td>
      <td>68.0</td>
      <td>159000</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR2</td>
      <td>Low</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Shed</td>
      <td>500</td>
      <td>6</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>277000</td>
    </tr>
    <tr>
      <th>1571</th>
      <td>1571</td>
      <td>916125425</td>
      <td>190</td>
      <td>RL</td>
      <td>68.0</td>
      <td>164660</td>
      <td>Grvl</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Shed</td>
      <td>700</td>
      <td>8</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>228950</td>
    </tr>
    <tr>
      <th>957</th>
      <td>957</td>
      <td>916176125</td>
      <td>20</td>
      <td>RL</td>
      <td>150.0</td>
      <td>215245</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR3</td>
      <td>Low</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2009</td>
      <td>WD</td>
      <td>Normal</td>
      <td>375000</td>
    </tr>
  </tbody>
</table>
<p>4 rows × 82 columns</p>
</div>




```python
la_outliers_droped = housing.drop(housing.index[[2072, 2116, 1571, 957]])
la_outliers_droped
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Order</th>
      <th>PID</th>
      <th>MS SubClass</th>
      <th>MS Zoning</th>
      <th>Lot Frontage</th>
      <th>Lot Area</th>
      <th>Street</th>
      <th>Alley</th>
      <th>Lot Shape</th>
      <th>Land Contour</th>
      <th>...</th>
      <th>Pool Area</th>
      <th>Pool QC</th>
      <th>Fence</th>
      <th>Misc Feature</th>
      <th>Misc Val</th>
      <th>Mo Sold</th>
      <th>Yr Sold</th>
      <th>Sale Type</th>
      <th>Sale Condition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>526301100</td>
      <td>20</td>
      <td>RL</td>
      <td>141.0</td>
      <td>31770</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>215000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>526350040</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>105000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>526351010</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>172000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>526353030</td>
      <td>20</td>
      <td>RL</td>
      <td>93.0</td>
      <td>11160</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
      <td>244000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2926</th>
      <td>2926</td>
      <td>923275080</td>
      <td>80</td>
      <td>RL</td>
      <td>37.0</td>
      <td>7937</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>GdPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>142500</td>
    </tr>
    <tr>
      <th>2927</th>
      <td>2927</td>
      <td>923276100</td>
      <td>20</td>
      <td>RL</td>
      <td>68.0</td>
      <td>8885</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Low</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>131000</td>
    </tr>
    <tr>
      <th>2928</th>
      <td>2928</td>
      <td>923400125</td>
      <td>85</td>
      <td>RL</td>
      <td>62.0</td>
      <td>10441</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>Shed</td>
      <td>700</td>
      <td>7</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>132000</td>
    </tr>
    <tr>
      <th>2929</th>
      <td>2929</td>
      <td>924100070</td>
      <td>20</td>
      <td>RL</td>
      <td>77.0</td>
      <td>10010</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>4</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>170000</td>
    </tr>
    <tr>
      <th>2930</th>
      <td>2930</td>
      <td>924151050</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>9627</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>11</td>
      <td>2006</td>
      <td>WD</td>
      <td>Normal</td>
      <td>188000</td>
    </tr>
  </tbody>
</table>
<p>2927 rows × 82 columns</p>
</div>




```python
la_outliers_droped.plot.scatter(x='Lot Area', y= 'SalePrice')
```




    <AxesSubplot:xlabel='Lot Area', ylabel='SalePrice'>




    
![png](pics/output_128_1.png)
    


<details>
<summary><strong>Solution</strong> (Click Here)</summary>
    &emsp; &emsp; <code>
sns.boxplot(x=housing['Lot Area'])
price_lot = housing.plot.scatter(x='Lot Area', y='SalePrice')   
housing['Lot_Area_Stats'] = stats.zscore(housing['Lot Area'])
housing[['Lot Area','Lot_Area_Stats']].describe().round(3)
housing.sort_values(by = 'Lot Area', ascending = False)[:1]
lot_area_rem = housing.drop(housing.index[[957]])
</code>
</details>


<details>
<summary><strong>Answer</strong> (Click Here)</summary>
    &emsp; &emsp; <code>
There seems to be one outlier, the very last point in the 'Lot Area' is too far from the rest of the group. Also, according to the Z-score, the standard deviation of that point exceeds the threshhold of 3.
</code>
</details>


## Z-score Analysis


Z-score is another way to identify outliers mathematically. Z-score is the signed number of standard deviations by which the value of an observation or data point is above the mean value of what is being observed or measured. In another words, Z-score is the value that quantifies relationship between a data point and a standard deviation and mean values of a group of points. Data points which are too far from zero will be treated as the outliers. In most of the cases, a threshold of 3 or -3 is used. For example, if the Z-score value is greater than or less than 3 or -3 standard deviations respectively, that data point will be identified as a outlier.

To learn more about Z-score, please visit this [Wikipedia](https://en.wikipedia.org/wiki/Standard_score?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) site.


Below, we are using Z-score function from `scipy` library to detect the outliers in our 'Low Qual Fin SF' parameter. To learn more about `scipy.stats`, please visit this [link](https://docs.scipy.org/doc/scipy/reference/tutorial/stats.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01).



```python
housing['LQFSF_Stats'] = stats.zscore(housing['Low Qual Fin SF'])
housing[['Low Qual Fin SF','LQFSF_Stats']].describe().round(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Low Qual Fin SF</th>
      <th>LQFSF_Stats</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2931.000</td>
      <td>2931.000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>4.675</td>
      <td>-0.000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>46.303</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000</td>
      <td>-0.101</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000</td>
      <td>-0.101</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000</td>
      <td>-0.101</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000</td>
      <td>-0.101</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1064.000</td>
      <td>22.882</td>
    </tr>
  </tbody>
</table>
</div>



The scaled results show a mean of 0.000 and a standard deviation of 1.000, indicating that the transformed values fit the z-scale model. The max value of 22.882 is further proof of the presence of outliers, as it falls well above the z-score limit of +3.


# Congratulations! - You have completed the lab


## Author


[Svitlana Kramar](https://www.linkedin.com/in/svitlana-kramar?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01)


## Change Log


| Date (YYYY-MM-DD) | Version | Changed By | Change Description                   |
| ----------------- | ------- | ---------- | ------------------------------------ |
| 2021-11-30        | 0.1     | Svitlana   | Added the Log Transformation section |
| 2022-01-18        | 0.2     | Svitlana   | Added the Introduction               |


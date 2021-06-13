## 1. Import libraries

```python
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from itertools import chain
from pyspark.sql.types import FloatType,StringType

```

## 2. Load Data

Since our upladed file contains 12 months worksheets, we need to load and merge them into one single dataframe. 

```python
#read worksheet names
files=pd.ExcelFile('월별매출.xlsx').sheet_names

#load all data and merge them into one single dataframe
merged=pd.DataFrame()
for file in files:
    df=pd.read_excel('월별매출.xlsx',sheet_name=file,skiprows=range(6)).iloc[:,1:]
    merged=pd.concat([merged,df],axis=0,ignore_index=True)

```

Initialize our SparkSession and create a dataframe for pyspark. 

```python
#initialize SparkSession
spark=SparkSession\
.builder\
.master('local[*]')\
.appName('test')\
.config('spark.sql.shuffle.partitions',4)\
.getOrCreate()
```

Add prices to the exisiting dataframe

```python
#add prices to the dataframe
prices=pd.read_excel('매출가격.xlsx')
prices=prices.T.loc["제품A":'제품F',0].to_dict()
merged['제품 가격']=merged['품명'].replace(prices)
```

## 3. Laguage Translation

All data is now translated into English. 

```python
#replacing value in column by searching a dictionary
methods={'인터넷뱅킹':'online banking', '신용카드':'credit card', '휴대폰결제':'micropayment', '무통장입금':'Transfer paymenet'}
status_info={'주문완료':'order received', '배송완료':'devliered', '배송중':'deliverying'}
mapping_method=create_map([lit(x) for x in chain(*methods.items())])
mapping_status=create_map([lit(x) for x in chain(*status_info.items())])

def valueCompute(quantity,price):
    return price*quantity*1.1

computing=udf(valueCompute,FloatType())


df=df.withColumn('value',valueCompute(col('quantity'),col('price')))\
.withColumn('stamp',regexp_replace(col('stamp'),'\.','-').alias('stamp'))\
.withColumn('month',split(col('stamp'),'-').getItem(1).cast('int'))\
.select(
    'stamp',
    'month',
    substring(col('department'),3,1).alias('departmenet'),
    substring(col('products'),3,1).alias('products'),
    mapping_status[df['status']].alias('status'),
    mapping_method[df['methods']].alias('method'),
    'address',
    'quantity',
    'value',
    'userId'
)
```

## 4. Summary Statistics Part(1)

```python
monthly_menas=df.\
groupBy('month').\
avg('value').toPandas()

monthly_product_means=df.\
groupBy('month','products').\
avg('value').\
orderBy('month','products').toPandas()

```

print out the first five rows of the sumamry table

```python
monthly_means=monthly_menas.sort_values('month')
monthly_product=monthly_product_means.sort_values('month')
monthly_product.head()
```
![image](https://user-images.githubusercontent.com/53164959/121822582-d39ab380-ccda-11eb-98a3-0cffbde56033.png)

Visualize all the findings into line charts.

```python
fig,ax=plt.subplots(figsize=(12,6))

ax.set_title('Monthly Revenue Sales on Avagere ')
ax.set_xlabel('month')
ax.plot(monthly_means['month'],monthly_means['avg(value)'],label='Total Sales Value')

for product in monthly_product_means['products'].unique():
    ax.plot(monthly_product_means['month'].unique(),monthly_product_means[monthly_product_means['products']==product]['avg(value)'],label=product)

ax.legend(loc='upper right')
```
![image](https://user-images.githubusercontent.com/53164959/121822607-f036eb80-ccda-11eb-93cd-03b15a9d2fc9.png)


## 5. Summary Statistics Part(2)


```python
grouped_df_by_products=df.groupBy('products').sum('quantity').orderBy('products').toPandas()
grouped_df_by_products=grouped_df_by_products.rename(columns={'sum(quantity)':'total_quantity'})

#visualize the data 
plt.title("Total Sales by Product Type")
plt.bar(grouped_df_by_products['products'],grouped_df_by_products['total_quantity'])
```
![image](https://user-images.githubusercontent.com/53164959/121822642-28d6c500-ccdb-11eb-90c4-821886910a1c.png)


```python

pivot_tb=df.groupBy('products').pivot('method').agg(sum('value')).orderBy('products').toPandas()
pivot_tb=pivot_tb.set_index('products')


plt.rcParams['figure.figsize']=(10,5)
plt.xlabel('Paymenet Methods')
plt.ylabel('Product Types')

plt.xticks(np.arange(0.5,4),pivot_tb.columns)
plt.yticks(np.arange(0.5,6),pivot_tb.index)

plt.pcolor(pivot_tb,edgecolors='black',cmap=plt.cm.Reds)
plt.colorbar()
```

![image](https://user-images.githubusercontent.com/53164959/121822661-40ae4900-ccdb-11eb-90bd-2ec8519c8696.png)

## 5. Summary Statistics Part(3)

```python
grouped_by=df\
.groupBy('userId')\
.agg(sum('value').alias('totalExpenditure'),count('stamp').alias('frequency'))

threshold_1=grouped_by.approxQuantile('totalExpenditure',[0.9],0)
threshold_2=grouped_by.approxQuantile('frequency',[0.9],0)

grouped_by.where((col('totalExpenditure')>=threshold_1[0])&(col('frequency')>threshold_2[0])).show()
```

![image](https://user-images.githubusercontent.com/53164959/121822685-5de31780-ccdb-11eb-9e98-f4ec3204bc51.png)


# Machine Learning Foundation

## Section 1, Part a: Reading Data


### Learning Objective(s)

*   Create a SQL database connection to a sample SQL database, and read records from that database
*   Explore common input parameters

### Packages

*   [Pandas](https://pandas.pydata.org/pandas-docs/stable/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01)
*   [Pandas.read_sql](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_sql.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01)
*   [SQLite3](https://docs.python.org/3.6/library/sqlite3.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01)


## Simple data reads

Structured Query Language (SQL) is an [ANSI specification](https://docs.oracle.com/database/121/SQLRF/ap_standard_sql001.htm?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01#SQLRF55514), implemented by various databases. SQL is a powerful format for interacting with large databases efficiently, and SQL allows for a consistent experience across a large market of databases. We'll be using sqlite, a lightweight and somewhat restricted version of sql for this example. sqlite uses a slightly modified version of SQL, which may be different than what you're used to.



```python
# Imports
import sqlite3 as sq3
import pandas.io.sql as pds
import pandas as pd
```

### Database connections

Our first step will be to create a connection to our SQL database. A few common SQL databases used with Python include:

*   Microsoft SQL Server
*   Postgres
*   MySQL
*   AWS Redshift
*   AWS Aurora
*   Oracle DB
*   Terradata
*   Db2 Family
*   Many, many others

Each of these databases will require a slightly different setup, and may require credentials (username & password), tokens, or other access requirements. We'll be using `sqlite3` to connect to our database, but other connection packages include:

*   [`SQLAlchemy`](https://www.sqlalchemy.org/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01) (most common)
*   [`psycopg2`](http://initd.org/psycopg/?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01)
*   [`MySQLdb`](http://mysql-python.sourceforge.net/MySQLdb.html?utm_medium=Exinfluencer&utm_source=Exinfluencer&utm_content=000026UJ&utm_term=10006555&utm_id=NA-SkillsNetwork-Channel-SkillsNetworkCoursesIBMML0232ENSkillsNetwork30654641-2021-01-01)



```python
# Download the database
!wget -P data https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/classic_rock.db
```

    --2022-05-09 08:44:43--  https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/classic_rock.db
    Resolving cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)... 169.63.118.104
    Connecting to cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud (cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud)|169.63.118.104|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 5652480 (5.4M) [binary/octet-stream]
    Saving to: ‘data/classic_rock.db’
    
    classic_rock.db     100%[===================>]   5.39M  23.1MB/s    in 0.2s    
    
    2022-05-09 08:44:43 (23.1 MB/s) - ‘data/classic_rock.db’ saved [5652480/5652480]
    



```python
# Initialize path to SQLite databasejdbc:sqlite:/C:/__tmp/test/sqlite/jdbcTest.db
path = 'data/classic_rock.db'
con = sq3.Connection(path)

# We now have a live connection to our SQL database
```


```python
con
```




    <sqlite3.Connection at 0x7fd54916ab90>



### Reading data

Now that we've got a connection to our database, we can perform queries, and load their results in as Pandas DataFrames



```python
# Write the query
query = '''
SELECT * 
FROM rock_songs;
'''

# Execute the query
observations = pds.read_sql(query, con)

observations.head()
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
      <th>Song</th>
      <th>Artist</th>
      <th>Release_Year</th>
      <th>PlayCount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Caught Up in You</td>
      <td>.38 Special</td>
      <td>1982.0</td>
      <td>82</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Hold On Loosely</td>
      <td>.38 Special</td>
      <td>1981.0</td>
      <td>85</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Rockin' Into the Night</td>
      <td>.38 Special</td>
      <td>1980.0</td>
      <td>18</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Art For Arts Sake</td>
      <td>10cc</td>
      <td>1975.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kryptonite</td>
      <td>3 Doors Down</td>
      <td>2000.0</td>
      <td>13</td>
    </tr>
  </tbody>
</table>
</div>




```python
# We can also run any supported SQL query
# Write the query
query = '''
SELECT Artist, Release_Year, COUNT(*) AS num_songs, AVG(PlayCount) AS avg_plays  
    FROM rock_songs
    GROUP BY Artist, Release_Year
    ORDER BY num_songs desc;
'''

# Execute the query
observations = pds.read_sql(query, con)

observations.head()
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
      <th>Artist</th>
      <th>Release_Year</th>
      <th>num_songs</th>
      <th>avg_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beatles</td>
      <td>1967.0</td>
      <td>23</td>
      <td>6.565217</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Led Zeppelin</td>
      <td>1969.0</td>
      <td>18</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beatles</td>
      <td>1965.0</td>
      <td>15</td>
      <td>3.800000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beatles</td>
      <td>1968.0</td>
      <td>13</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Beatles</td>
      <td>1969.0</td>
      <td>13</td>
      <td>15.000000</td>
    </tr>
  </tbody>
</table>
</div>



## Common parameters

There are a number of common paramters that can be used to read in SQL data with formatting:

*   coerce_float: Attempt to force numbers into floats
*   parse_dates: List of columns to parse as dates
*   chunksize: Number of rows to include in each chunk

Let's have a look at using some of these parameters



```python
query='''
SELECT Artist, Release_Year, COUNT(*) AS num_songs, AVG(PlayCount) AS avg_plays  
    FROM rock_songs
    GROUP BY Artist, Release_Year
    ORDER BY num_songs desc;
'''

# Execute the query
observations_generator = pds.read_sql(query,
                            con,
                            coerce_float=True, # Doesn't efefct this dataset, because floats were correctly parsed
                            parse_dates=['Release_Year'], # Parse `Release_Year` as a date
                            chunksize=5 # Allows for streaming results as a series of shorter tables
                           )

for index, observations in enumerate(observations_generator):
    if index < 5:
        print(f'Observations index: {index}'.format(index))
        display(observations)
```

    Observations index: 0



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
      <th>Artist</th>
      <th>Release_Year</th>
      <th>num_songs</th>
      <th>avg_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>The Beatles</td>
      <td>1970-01-01 00:32:47</td>
      <td>23</td>
      <td>6.565217</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Led Zeppelin</td>
      <td>1970-01-01 00:32:49</td>
      <td>18</td>
      <td>21.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beatles</td>
      <td>1970-01-01 00:32:45</td>
      <td>15</td>
      <td>3.800000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beatles</td>
      <td>1970-01-01 00:32:48</td>
      <td>13</td>
      <td>13.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Beatles</td>
      <td>1970-01-01 00:32:49</td>
      <td>13</td>
      <td>15.000000</td>
    </tr>
  </tbody>
</table>
</div>


    Observations index: 1



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
      <th>Artist</th>
      <th>Release_Year</th>
      <th>num_songs</th>
      <th>avg_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Led Zeppelin</td>
      <td>1970-01-01 00:32:50</td>
      <td>12</td>
      <td>13.166667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Led Zeppelin</td>
      <td>1970-01-01 00:32:55</td>
      <td>12</td>
      <td>14.166667</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Pink Floyd</td>
      <td>1970-01-01 00:32:59</td>
      <td>11</td>
      <td>41.454545</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Pink Floyd</td>
      <td>1970-01-01 00:32:53</td>
      <td>10</td>
      <td>29.100000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>The Doors</td>
      <td>1970-01-01 00:32:47</td>
      <td>10</td>
      <td>28.900000</td>
    </tr>
  </tbody>
</table>
</div>


    Observations index: 2



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
      <th>Artist</th>
      <th>Release_Year</th>
      <th>num_songs</th>
      <th>avg_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Fleetwood Mac</td>
      <td>1970-01-01 00:32:57</td>
      <td>9</td>
      <td>35.666667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Jimi Hendrix</td>
      <td>1970-01-01 00:32:47</td>
      <td>9</td>
      <td>24.888889</td>
    </tr>
    <tr>
      <th>2</th>
      <td>The Beatles</td>
      <td>1970-01-01 00:32:43</td>
      <td>9</td>
      <td>2.444444</td>
    </tr>
    <tr>
      <th>3</th>
      <td>The Beatles</td>
      <td>1970-01-01 00:32:44</td>
      <td>9</td>
      <td>3.111111</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Elton John</td>
      <td>1970-01-01 00:32:53</td>
      <td>8</td>
      <td>18.500000</td>
    </tr>
  </tbody>
</table>
</div>


    Observations index: 3



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
      <th>Artist</th>
      <th>Release_Year</th>
      <th>num_songs</th>
      <th>avg_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Led Zeppelin</td>
      <td>1970-01-01 00:32:51</td>
      <td>8</td>
      <td>47.750000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Led Zeppelin</td>
      <td>1970-01-01 00:32:53</td>
      <td>8</td>
      <td>34.125000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Boston</td>
      <td>1970-01-01 00:32:56</td>
      <td>7</td>
      <td>69.285714</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Rolling Stones</td>
      <td>1970-01-01 00:32:49</td>
      <td>7</td>
      <td>36.142857</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Van Halen</td>
      <td>1970-01-01 00:32:58</td>
      <td>7</td>
      <td>51.142857</td>
    </tr>
  </tbody>
</table>
</div>


    Observations index: 4



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
      <th>Artist</th>
      <th>Release_Year</th>
      <th>num_songs</th>
      <th>avg_plays</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Bruce Springsteen</td>
      <td>1970-01-01 00:32:55</td>
      <td>6</td>
      <td>7.666667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bruce Springsteen</td>
      <td>1970-01-01 00:33:04</td>
      <td>6</td>
      <td>11.500000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Creedence Clearwater Revival</td>
      <td>1970-01-01 00:32:49</td>
      <td>6</td>
      <td>23.833333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Creedence Clearwater Revival</td>
      <td>1970-01-01 00:32:50</td>
      <td>6</td>
      <td>18.833333</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Def Leppard</td>
      <td>1970-01-01 00:33:07</td>
      <td>6</td>
      <td>32.000000</td>
    </tr>
  </tbody>
</table>
</div>


### Machine Learning Foundation (C) 2020 IBM Corporation


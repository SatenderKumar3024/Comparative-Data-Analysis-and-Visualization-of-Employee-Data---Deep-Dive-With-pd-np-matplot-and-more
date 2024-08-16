#!/usr/bin/env python
# coding: utf-8

# Merge, Concatenate, join and more

# In[1]:


import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # For statistical data visualization based on Matplotlib
import scipy  # For scientific and technical computing (including optimization, integration, and statistics)


# In[2]:


# Creating realistic data for employees
data = {
    'Employee ID': np.arange(1001, 1011),
    'Employee Name': ['Satender Kumar', 'data 1', 'Jane Smith', 'Robert Brown', 'Emily Davis', 'Michael Wilson', 'Sarah Taylor', 'David Lee', 'Laura Johnson', 'James White'],
    'Department': ['Data Analyst', 'IT', 'Finance', 'Marketing', 'Sales', 'Operations', 'R&D', 'Support', 'Admin', 'Legal'],
    'Age': [24, np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60)],
    'Location': ['London, Canada', 'Toronto', 'London', 'Sydney', 'San Francisco', 'Paris', 'Berlin', 'Tokyo', 'Dubai', 'Singapore'],
    'Salary': np.random.randint(50000, 150000, size=10),
    'Years with Company': np.random.randint(1, 15, size=10),
    'Position': ['Data Analyst', 'Developer', 'Analyst', 'Designer', 'Consultant', 'Engineer', 'Scientist', 'Support Agent', 'Admin Assistant', 'Lawyer'],
    'Performance Score': np.random.randint(1, 5, size=10),
    'Bonus': np.random.randint(1000, 10000, size=10),
    'Gender': ['Male', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male'],
    'Marital Status': ['Single', 'Single', 'Married', 'Single', 'Single', 'Married', 'Married', 'Single', 'Married', 'Single'],
    'Education': ['Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'Bachelor', 'Master'],
    'Hire Date': pd.to_datetime(['2019-06-12', '2015-07-23', '2012-09-05', '2018-11-30', '2013-05-19', '2019-02-14', '2020-08-21', '2016-06-03', '2014-01-28', '2017-03-15']),
    'Overtime Hours': np.random.randint(0, 20, size=10),
    'Sick Days Taken': np.random.randint(0, 10, size=10),
    'Vacation Days Taken': np.random.randint(5, 20, size=10),
    'Training Hours': np.random.randint(10, 50, size=10),
    'Certifications': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes'],
    'Supervisor': ['Anna Smith', 'Brian Adams', 'Clara Jones', 'Daniel Martin', 'Eva Rodriguez', 'Frank Bell', 'Grace Moore', 'Hannah Lewis', 'Ivan Scott', 'Jake Miller']
}

# Creating the DataFrame
df = pd.DataFrame(data)


# In[4]:


# Display the DataFrame
print(df)


# In[6]:


df.head()


# In[10]:


df.describe()


# In[12]:


df.isnull()


# In[14]:


df.notnull()


# In[16]:


df.count()


# In[19]:


df.head(20
       )


# In[21]:


# Creating realistic data for a second set of employees
data1 = {
    'Employee ID': np.arange(1011, 1021),
    'Employee Name': ['Satender Kumar', 'data 1', 'Chris Evans', 'Natalie Portman', 'Tom Holland', 'Emma Watson', 'Daniel Radcliffe', 'Scarlett Johansson', 'Robert Downey Jr.', 'Mark Ruffalo'],
    'Department': ['Data Analyst', 'HR', 'IT', 'Marketing', 'Finance', 'Sales', 'R&D', 'Operations', 'Legal', 'Support'],
    'Age': [24, np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60), np.random.randint(25, 60)],
    'Location': ['London, Canada', 'Los Angeles', 'New York', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 'San Antonio', 'San Diego', 'Dallas'],
    'Salary': np.random.randint(60000, 160000, size=10),
    'Years with Company': np.random.randint(1, 20, size=10),
    'Position': ['Data Analyst', 'HR Manager', 'IT Specialist', 'Marketing Coordinator', 'Financial Analyst', 'Sales Manager', 'Research Scientist', 'Operations Manager', 'Legal Advisor', 'Support Specialist'],
    'Performance Score': np.random.randint(1, 5, size=10),
    'Bonus': np.random.randint(2000, 12000, size=10),
    'Gender': ['Male', 'Male', 'Female', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Male'],
    'Marital Status': ['Single', 'Married', 'Single', 'Single', 'Married', 'Single', 'Single', 'Married', 'Single', 'Married'],
    'Education': ['Master', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD', 'Bachelor', 'Master', 'PhD'],
    'Hire Date': pd.to_datetime(['2018-07-15', '2014-03-22', '2011-10-12', '2017-04-17', '2015-09-23', '2016-11-01', '2019-05-11', '2020-07-08', '2013-08-19', '2012-01-09']),
    'Overtime Hours': np.random.randint(0, 25, size=10),
    'Sick Days Taken': np.random.randint(0, 8, size=10),
    'Vacation Days Taken': np.random.randint(7, 22, size=10),
    'Training Hours': np.random.randint(15, 55, size=10),
    'Certifications': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No'],
    'Supervisor': ['John Smith', 'Michael Johnson', 'Patricia Williams', 'Linda Brown', 'Barbara Jones', 'Elizabeth Garcia', 'Susan Martinez', 'Jessica Hernandez', 'Sarah Lopez', 'Karen Wilson']
}

# Creating the second DataFrame
df1 = pd.DataFrame(data1)


# In[23]:


# Display the DataFrame
print(df1)


# In[25]:


df1.head()


# In[27]:


df.describe()


# In[28]:


df1.notnull()


# In[29]:


df1.isnull()


# In[30]:


df1.count()


# In[ ]:


df1.sum()


# In[33]:


print(df1)


# In[36]:


import pandas as pd

# Assuming df and df1 are your DataFrames from the previous examples

# Merging the two DataFrames on the 'Employee ID' column
merged_df = pd.merge(df, df1, on='Employee ID', suffixes=('_df', '_df1'))

# Display the merged DataFrame
print(merged_df)


# In[39]:


merged_df = pd.merge(df, df1, on='Employee ID', suffixes=('_df', '_df1'), how='outer')
print(merged_df)


# In[43]:


print(df['Employee ID'])
print(df1['Employee ID'])


# In[45]:


df


# In[46]:


df1


# In[50]:


print(pd.concat([df,df1]))


# In[52]:


df.info()


# In[55]:


df.plot()


# In[57]:


df.hist(figsize=(12, 10), bins=15, grid=False)

# Display the plots
plt.show()


# In[ ]:





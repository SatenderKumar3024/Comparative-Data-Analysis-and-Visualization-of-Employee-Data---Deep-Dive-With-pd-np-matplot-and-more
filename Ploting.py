#!/usr/bin/env python
# coding: utf-8

# # Data Viz in Plotting

# In[ ]:


import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # For statistical data visualization based on Matplotlib
import scipy  # For scientific and technical computing (including optimization, integration, and statistics)


# In[4]:


import pandas as pd 
import numpy as np
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


# In[10]:


df


# In[11]:


import pandas as pd 
import numpy as np


# In[13]:


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


# In[15]:


df1


# In[17]:


merged_df = pd.merge(df, df1, on='Employee ID', suffixes=('_df', '_df1'), how='outer')
print(merged_df)


# In[19]:


df.plot()


# In[21]:


df1.plot()


# In[54]:


# Pie chart for department distribution in df1
df1['Department'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Department Distribution in df1')
plt.ylabel('')  # Remove the y-label for a cleaner look
plt.show()


# In[56]:


# Stacked bar plot showing count of employees by Department and Gender in df
df.groupby(['Department', 'Gender']).size().unstack().plot(kind='bar', stacked=True, figsize=(10, 7))
plt.title('Stacked Bar Plot for Department and Gender in df')
plt.ylabel('Number of Employees')
plt.show()


# In[57]:


# Area plot showing salary over years with company in df1
df1.plot.area(x='Years with Company', y='Salary', figsize=(10, 7), alpha=0.4)
plt.title('Area Plot for Salary Over Years with Company in df1')
plt.ylabel('Salary')
plt.show()


# In[58]:


from mpl_toolkits.mplot3d import Axes3D

# 3D scatter plot in df
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Age'], df['Salary'], df['Performance Score'], c='r', marker='o')
ax.set_xlabel('Age')
ax.set_ylabel('Salary')
ax.set_zlabel('Performance Score')
plt.title('3D Scatter Plot in df')
plt.show()


# In[59]:


# Hexbin plot for Age vs. Salary in df1
df1.plot.hexbin(x='Age', y='Salary', gridsize=20, cmap='Blues', figsize=(10, 7))
plt.title('Hexbin Plot for Age vs. Salary in df1')
plt.show()


# In[61]:


# Hexbin plot for Age vs. Salary in df
df.plot.hexbin(x='Age', y='Salary', gridsize=20, cmap='Blues', figsize=(11, 8))
plt.title('Hexbin Plot for Age vs. Salary in df')
plt.show()


# In[63]:


df.boxplot()


# In[30]:


df1.boxplot()


# In[32]:


import pandas as pd
import matplotlib.pyplot as plt

# Assuming df is your first DataFrame
df.hist(figsize=(12, 10), bins=15, grid=False)

# Display the plots for df
plt.suptitle('Histograms for df')
plt.show()


# In[35]:


# Assuming df1 is your second DataFrame
df1.hist(figsize=(12, 10), bins=15, grid=False)

# Display the plots for df1
plt.suptitle('Histograms for df1')
plt.show()


# In[36]:


# Box plots for df1
df1.boxplot(figsize=(12, 10))
plt.suptitle('Box Plots for df1')
plt.show()


# In[37]:


# Box plots for df1
df.boxplot(figsize=(12, 10))
plt.suptitle('Box Plots for df')
plt.show()


# In[44]:


# Scatter plot between 'Age' and 'Salary' in df1
df1.plot.scatter(x='Age', y='Salary', title='Age vs. Salary in df1')
plt.show()


# In[45]:


# Scatter plot between 'Age' and 'Salary' in df
df.plot.scatter(x='Age', y='Salary', title='Age vs. Salary in df')
plt.show()


# In[48]:


# Density plot for Age distribution in df1
df1['Age'].plot(kind='density', title='Density Plot for Age in df1')
plt.show()


# In[51]:


import seaborn as sns

# Violin plot for Salary distribution in df
sns.violinplot(y='Salary', data=df)
plt.title('Salary Distribution in df')
plt.show()



# In[52]:


# Pie chart for department distribution in df1
df1['Department'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Department Distribution in df1')
plt.ylabel('')  # Remove the y-label for a cleaner look
plt.show()


# In[53]:


# Pie chart for department distribution in df
df['Department'].value_counts().plot(kind='pie', autopct='%1.1f%%', title='Department Distribution in df')
plt.ylabel('')  # Remove the y-label for a cleaner look
plt.show()


# In[68]:


import numpy as np

# Radar chart for comparing different metrics for the first employee in df
categories = ['Age', 'Salary', 'Years with Company', 'Performance Score', 'Bonus']
values = df.loc[0, categories].values.flatten().tolist()

# Adding the first value to the end of the list to close the radar chart
values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, values, color='red', alpha=0.25)
ax.plot(angles, values, color='red', linewidth=2)
ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Radar Chart for First Employee in df')
plt.show()


# In[69]:


import plotly.express as px

# Sunburst plot showing hierarchy of Department and Gender in df
fig = px.sunburst(df, path=['Department', 'Gender'], values='Salary')
fig.update_layout(title='Sunburst Plot for Department and Gender in df')
fig.show()


# In[72]:


# Joint plot for Age vs. Salary in df
sns.jointplot(x='Age', y='Salary', data=df, kind='reg', height=8)
plt.suptitle('Joint Plot for Age vs. Salary in df', y=1.03)
plt.show()


# In[73]:


# Swarm plot for Performance Score across Department in df1
plt.figure(figsize=(10, 7))
sns.swarmplot(x='Department', y='Performance Score', data=df1)
plt.title('Swarm Plot for Performance Score Across Department in df1')
plt.xticks(rotation=90)
plt.show()


#!/usr/bin/env python
# coding: utf-8

# # Panda|Compare DataFrame 

# In[1]:


import pandas as pd  # For data manipulation and analysis
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For creating static, animated, and interactive visualizations
import seaborn as sns  # For statistical data visualization based on Matplotlib
import scipy  # For scientific and technical computing (including optimization, integration, and statistics)


# In[3]:


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


# In[5]:


df


# In[6]:


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


# In[8]:


df1


# In[11]:


df.head()


# In[13]:


df1.head()


# In[15]:


# Check if df and df1 are exactly the same
are_identical = df.equals(df1)
print(f"Are df and df1 identical? {are_identical}")


# In[17]:


# Find differences between df and df1
comparison_df = df.compare(df1, keep_shape=True, keep_equal=True)
print("Differences between df and df1:")
print(comparison_df)


# In[20]:


# Identify rows that differ between df and df1
differing_rows = df[df.ne(df1).any(axis=1)]
print("Rows that differ between df and df1:")
print(differing_rows)


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

# Create a heatmap to visualize differences between df and df1
diff = df.ne(df1).astype(int)  # 1 where different, 0 where the same
plt.figure(figsize=(12, 8))
sns.heatmap(diff, cmap='coolwarm', cbar=False, annot=True)
plt.title('Heatmap of Differences Between df and df1')
plt.show()


# In[24]:


# Summarize the number of differences by column
summary = df.ne(df1).sum()
print("Summary of differences by column:")
print(summary)


# In[26]:


import matplotlib.pyplot as plt

# Overlayed histograms for 'Salary' in df and df1
plt.figure(figsize=(10, 6))
plt.hist(df['Salary'], bins=15, alpha=0.5, label='df Salary', color='blue')
plt.hist(df1['Salary'], bins=15, alpha=0.5, label='df1 Salary', color='orange')
plt.title('Overlayed Histograms of Salary in df and df1')
plt.xlabel('Salary')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# In[29]:


# Combine the data into a single DataFrame for side-by-side boxplots
combined = pd.DataFrame({
    'df_Salary': df['Salary'],
    'df1_Salary': df1['Salary']
})

# Plot the boxplots
plt.figure(figsize=(10, 6))
combined.boxplot()
plt.title('Side-by-Side Boxplots of Salary in df and df1')
plt.ylabel('Salary')
plt.show()


# In[31]:


# Group by Department and count in both DataFrames
df_dept_counts = df['Department'].value_counts()
df1_dept_counts = df1['Department'].value_counts()

# Combine into a single DataFrame
dept_comparison = pd.DataFrame({'df': df_dept_counts, 'df1': df1_dept_counts})

# Plot stacked bar plot
dept_comparison.plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title('Stacked Bar Plot Comparing Department Distribution in df and df1')
plt.ylabel('Number of Employees')
plt.show()


# In[33]:


# Scatter plot comparison for 'Age' vs 'Salary' in both DataFrames
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(df['Age'], df['Salary'], color='blue', alpha=0.5)
plt.title('df: Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')

plt.subplot(1, 2, 2)
plt.scatter(df1['Age'], df1['Salary'], color='orange', alpha=0.5)
plt.title('df1: Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')

plt.show()


# In[35]:


# Density plot for 'Years with Company' in df and df1
plt.figure(figsize=(10, 6))
df['Years with Company'].plot(kind='density', label='df Years with Company', color='blue')
df1['Years with Company'].plot(kind='density', label='df1 Years with Company', color='orange')
plt.title('Density Plot of Years with Company in df and df1')
plt.xlabel('Years with Company')
plt.legend()
plt.show()


# In[37]:


# Group by Gender and count in both DataFrames
df_gender_counts = df['Gender'].value_counts()
df1_gender_counts = df1['Gender'].value_counts()

# Combine into a single DataFrame
gender_comparison = pd.DataFrame({'df': df_gender_counts, 'df1': df1_gender_counts})

# Plot clustered bar plot
gender_comparison.plot(kind='bar', width=0.8, figsize=(10, 6))
plt.title('Clustered Bar Plot Comparing Gender Distribution in df and df1')
plt.ylabel('Number of Employees')
plt.show()


# In[39]:


# Group by Gender and count in both DataFrames
df_gender_counts = df['Gender'].value_counts()
df1_gender_counts = df1['Gender'].value_counts()

# Combine into a single DataFrame
gender_comparison = pd.DataFrame({'df': df_gender_counts, 'df1': df1_gender_counts})

# Plot clustered bar plot
gender_comparison.plot(kind='bar', width=0.8, figsize=(10, 6))
plt.title('Clustered Bar Plot Comparing Gender Distribution in df and df1')
plt.ylabel('Number of Employees')
plt.show()


# In[41]:


# Bubble plot for Age vs Salary with Performance Score as bubble size
plt.figure(figsize=(10, 6))

plt.scatter(df['Age'], df['Salary'], s=df['Performance Score']*100, alpha=0.5, label='df', color='blue')
plt.scatter(df1['Age'], df1['Salary'], s=df1['Performance Score']*100, alpha=0.5, label='df1', color='orange')

plt.title('Bubble Plot Comparing Age vs Salary in df and df1')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.legend()
plt.show()


# In[43]:


# Comparing first employee in both DataFrames using a radar chart
categories = ['Age', 'Salary', 'Years with Company', 'Performance Score', 'Bonus']
df_values = df.loc[0, categories].values.flatten().tolist()
df1_values = df1.loc[0, categories].values.flatten().tolist()

# Closing the radar chart by adding the first value at the end
df_values += df_values[:1]
df1_values += df1_values[:1]
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, df_values, color='blue', alpha=0.25, label='df')
ax.plot(angles, df_values, color='blue', linewidth=2)

ax.fill(angles, df1_values, color='orange', alpha=0.25, label='df1')
ax.plot(angles, df1_values, color='orange', linewidth=2)

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
plt.title('Radar Chart Comparing First Employee in df and df1')
plt.legend()
plt.show()


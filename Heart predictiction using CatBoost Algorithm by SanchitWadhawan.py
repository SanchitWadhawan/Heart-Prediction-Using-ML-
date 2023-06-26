#!/usr/bin/env python
# coding: utf-8

# In[26]:


get_ipython().system('pip install pandas scikit-learn catboost')
get_ipython().system('pip install imbalanced-learn')


# In[24]:


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE



# In[11]:


data = pd.read_csv('heart_failure_clinical_records_dataset.csv')


# In[12]:


X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[13]:


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Step 1: Data Preparation
# Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Split features and target variable
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Model Training
# Train the CatBoost classifier
model = CatBoostClassifier()
model.fit(X_train, y_train)

# Step 3: Model Evaluation
# Predict on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))


# In[14]:


###from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.1, 0.01, 0.001],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5]
}

# Perform grid search
grid_search = GridSearchCV(estimator=CatBoostClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Train the model with the best parameters
model = CatBoostClassifier(**best_params)
model.fit(X_train, y_train)


# In[ ]:





# In[15]:


class_weights = [1, 5]  # Adjust the weights based on the class imbalance ratio
model = CatBoostClassifier(class_weights=class_weights)


# In[16]:


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# Step 1: Data Preparation
# Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Split features and target variable
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Hyperparameter Tuning
# Define the parameter grid
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.1, 0.01, 0.001],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5]
}

# Perform grid search
grid_search = GridSearchCV(estimator=CatBoostClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Step 3: Feature Scaling
# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data
X_test_scaled = scaler.transform(X_test)

# Step 4: Model Training
# Train the model with the best parameters
model = CatBoostClassifier(**best_params)
model.fit(X_train_scaled, y_train)

# Step 5: Model Evaluation
# Predict on the testing data
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[18]:


from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate precision
precision = precision_score(y_test, y_pred)

# Calculate recall
recall = recall_score(y_test, y_pred)

# Calculate F1 score
f1 = f1_score(y_test, y_pred)

# Print precision, recall, and F1 score
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))


# In[19]:


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import SMOTE

# Step 1: Data Preparation
# Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Split features and target variable
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Hyperparameter Tuning
# Define the parameter grid
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.1, 0.01, 0.001],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5]
}

# Perform grid search
grid_search = GridSearchCV(estimator=CatBoostClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Step 3: Feature Scaling
# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data
X_test_scaled = scaler.transform(X_test)

# Step 4: Data Augmentation (Optional)
# Apply oversampling using SMOTE
oversampler = SMOTE(random_state=42)
X_train_augmented, y_train_augmented = oversampler.fit_resample(X_train_scaled, y_train)

# Step 5: Model Training
# Train the model with the best parameters
model = CatBoostClassifier(**best_params)
model.fit(X_train_augmented, y_train_augmented)

# Step 6: Model Evaluation
# Predict on the testing data
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))


# In[27]:


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV 
from imblearn.over_sampling import SMOTE


# Step 1: Data Preparation
# Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Split features and target variable
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 2: Hyperparameter Tuning
# Define the parameter grid
param_grid = {
    'depth': [4, 6, 8],
    'learning_rate': [0.1, 0.01, 0.001],
    'iterations': [100, 200, 300],
    'l2_leaf_reg': [1, 3, 5]
}

# Perform grid search
grid_search = GridSearchCV(estimator=CatBoostClassifier(), param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_params = grid_search.best_params_

# Step 3: Feature Scaling
# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing data
X_test_scaled = scaler.transform(X_test)

# Step 4: Model Training with Class Weights
# Calculate class weights
class_counts = y_train.value_counts()
class_weights = dict(1 / class_counts)

# Train the model with the best parameters and class weights
model = CatBoostClassifier(**best_params, class_weights=class_weights)
model.fit(X_train_scaled, y_train)

# Step 5: Model Evaluation
# Predict on the testing data
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))


# In[28]:


from imblearn.over_sampling import SMOTE

# Apply oversampling
oversampler = SMOTE(random_state=42)
X_train_augmented, y_train_augmented = oversampler.fit_resample(X_train_scaled, y_train)


# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Step 1: Data Preparation
# Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Split features and target variable
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']

# Step 2: Feature Scaling
# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 4: Data Augmentation using SMOTE
oversampler = SMOTE(random_state=42)
X_train_augmented, y_train_augmented = oversampler.fit_resample(X_train, y_train)

# Step 5: Model Training
model = CatBoostClassifier()
model.fit(X_train_augmented, y_train_augmented)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))



# In[31]:


import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Step 1: Data Preparation
# Load the dataset
data = pd.read_csv('heart_failure_clinical_records_dataset.csv')

# Select features and target variable
features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction',
            'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'sex',
            'smoking', 'time']
target = 'DEATH_EVENT'

X = data[features]
y = data[target]

# Step 2: Feature Scaling
# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the data
X_scaled = scaler.fit_transform(X)

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Step 4: Data Augmentation using SMOTE
oversampler = SMOTE(random_state=42)
X_train_augmented, y_train_augmented = oversampler.fit_resample(X_train, y_train)

# Step 5: Model Training
model = CatBoostClassifier()
model.fit(X_train_augmented, y_train_augmented)

# Step 6: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy * 100))


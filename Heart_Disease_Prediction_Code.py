# Plotting Libraries:-
import matplotlib.pyplot as plt # Provides an implicit way of plotting
import numpy as np # Support for large, multi-dimensional arrays and matrices
import pandas as pd  # Library for working with data sets
import seaborn as sns # Provides high level API to visualize data

# Metrics for Classification Technique:-
from sklearn.metrics import accuracy_score # Accuracy classification score
from sklearn.metrics import classification_report # Build a text report showing the main classification metrics
from sklearn.metrics import confusion_matrix # Compute confusion matrix to evaluate the accuracy of a classification
from sklearn.metrics import roc_auc_score # Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores
from sklearn.metrics import roc_curve # Compute Receiver operating characteristic (ROC); This implementation is restricted to the binary classification task

# Scaler Libraries:-
from sklearn.model_selection import train_test_split # Split arrays or matrices into random train and test subsets
from sklearn.preprocessing import StandardScaler # Standardize features by removing the mean and scaling to unit variance

# Model Building Libraries:-
from sklearn.naive_bayes import GaussianNB # Gaussian Naive Bayes (GaussianNB)
from sklearn.neighbors import KNeighborsClassifier # Classifier implementing the k-nearest neighbors vote
from sklearn.tree import DecisionTreeClassifier # A decision tree classifier
from sklearn.svm import SVC # C-Support Vector Classification

import warnings
warnings.filterwarnings('ignore') # Never print matching warnings

"""
There is a unicode character '\u0332', COMBINING LOW LINE*, which acts as an underline on the character that precedes it in a string. The center() method will center align the string, using a specified character (space is default) as the fill character.
"""

def print_csv_file(df, heading):
    print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    print(df)
    
def exploratory_data_analysis(df):
    
    def plot_data_set(df):
        
        """
        The method yscale() or xscale() takes a single value as a parameter which is the type of conversion of the scale, to convert axes to logarithmic scale we pass the “log” keyword or the matplotlib.scale
        """
        
        plt.xscale('log'); plt.xlabel("Logarithmic X-Axis")
        plt.yscale('log'); plt.ylabel("Logarithmic Y-Axis")
        plt.title("Plot Data Set"); plt.plot(df)
        plt.legend(df); plt.grid(True); plt.show()

    def boxplot(df):
        plt.title("Boxplot: Graphic Display of Five Number Summary")
        sns.boxplot(data = df, orient = 'h')
        plt.show()
    
    print("\nSize of the dataset - ", df.shape)
    print("\nGeneral information of the dataset - \n") 
    df.info() # Print a concise summary of a DataFrame
    print("\nStatistical description and dispersion of the dataset - \n", df.describe()) # Generate descriptive statistics;
    
    plot_data_set(df); boxplot(df)
    
    print("\nCorrelation Between Various Features - It is always better to check the correlation between the features so that we can analyze that which feature is negatively correlated and which is positively correlated.")

    plt.figure(figsize=(20,12)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.3) # Set the parameters that control the scaling of plot elements
    sns.heatmap(df.corr(),annot=True,linewidth=2) # Plot rectangular data as a color-encoded matrix
    plt.title("Correlation Between Various Features")
    plt.tight_layout(); plt.show()
    
    sns.set_context('notebook',font_scale = 2.3) # Set the parameters that control the scaling of plot elements
    df.drop('AHD', axis=1).corrwith(df.AHD).plot(kind='bar', grid=True, figsize=(20, 10), title="Correlation with the AHD (Target) Feature") # Drop specified labels from rows or columns
    plt.tight_layout(); plt.show()
    
    print("\nCorrelation with the AHD feature - Except the ChestPain and MaxHR, all others are negatively correlated with the AHD (Target) Feature. ")
    
def Age_feature_analysis(df):
    
    plt.figure(figsize=(25,12)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.5) # Set the parameters that control the scaling of plot elements
    sns.barplot(x=df.Age.value_counts()[:10].index,y=df.Age.value_counts()[:10].values) # Show point estimates and confidence intervals as rectangular bars
    plt.xlabel("Age (in years)"); plt.ylabel("Count")
    plt.title("Ages and Their Counts")
    plt.tight_layout(); plt.grid(True); plt.show()
    
    print("\nAges and Their Counts - We observe that the 58 age group has the highest frequency.")
    
    print("\nRange of age in the dataset:-")
    
    print("Minimum Age - ", min(df.Age))
    print("Maximum Age - ", max(df.Age))
    print("Mean Age - ", df.Age.mean())

    Young = df[(df.Age>=29)&(df.Age<40)]
    Middle = df[(df.Age>=40)&(df.Age<55)]
    Elder = df[(df.Age>55)]
    
    plt.figure(figsize=(23,10)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.5) # Set the parameters that control the scaling of plot elements
    sns.barplot(x=['Young','Middle','Elder'], y=[len(Young),len(Middle),len(Elder)]) # Show point estimates and confidence intervals as rectangular bars
    plt.xlabel("Age (in years)"); plt.ylabel("Count")
    plt.title("\nDivide the Age feature into three parts – “Young”, “Middle” and “Elder”")
    plt.tight_layout(); plt.grid(True); plt.show()
    
    print("\nWe observe that elderly people are the most and the young ones are the least affected by heart disease.")
    
    colors = ['blue','green','yellow']; explode = [0,0,0.1]
    plt.figure(figsize=(10,10)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.2) # Set the parameters that control the scaling of plot elements
    plt.pie([len(Young),len(Middle),len(Elder)],labels=['Young','Middle','Elder'],explode=explode,colors=colors, autopct='%1.1f%%') # Plot a pie chart
    plt.title("Pie Representation of Age Feature Division")
    plt.tight_layout(); plt.grid(True); plt.show()
    
def Sex_feature_analysis(df):
    
    plt.figure(figsize=(18,9)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.5) # Set the parameters that control the scaling of plot elements
    sns.countplot(df['Sex']) # Show the counts of observations in each categorical bin using bars
    plt.title("Ratio of Sex (0 (Female) and 1 (Male))")
    plt.tight_layout(); plt.grid(True); plt.show()
    
    print("\nRatio of Sex - We observe that female to male ratio is approximately to 1:2.")
    
    plt.figure(figsize=(18,9)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.5) # Set the parameters that control the scaling of plot elements
    sns.countplot(df['Sex'],hue=df["Slope"]) # Show the counts of observations in each categorical bin using bars
    plt.title("Relationship Between Sex (0 (Female) and 1 (Male)) and Slope (The slope of the peak exercise ST segment, 1 (Downsloping), 2 (Flat) and 3 (Upsloping))")
    plt.tight_layout(); plt.grid(True); plt.show()
    
    print("\nRelationship between Sex and Slope - We see that the slope value is higher in the case of males than females.")
    
def ChestPain_feature_analysis(df):
    
    plt.figure(figsize=(18,9)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.5) # Set the parameters that control the scaling of plot elements
    sns.countplot(df['ChestPain']) # Show the counts of observations in each categorical bin using bars
    plt.title("Types of Chest Pain - 0 (asymptomatic), 1 (nonanginal), 2 (nontypical) and 3 (typical)")
    plt.tight_layout(); plt.grid(True); plt.show()
    
    plt.figure(figsize=(18,9)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.5) # Set the parameters that control the scaling of plot elements
    sns.countplot(df['ChestPain'],hue=df["AHD"]) # Show the counts of observations in each categorical bin using bars
    plt.title("Relationship Between ChestPain and AHD (0 - No (Normal) and 1 - Yes (Heart Disease))")
    plt.tight_layout(); plt.grid(True); plt.show()
    
    print("\nWe observe that:- \n(i) People having the least chest pain are not likely to have heart disease. \n(ii) People having severe chest pain are likely to have heart disease.")
    
def MaxHR_feature_analysis(df):
    
    plt.figure(figsize=(9, 7)) # Create a new figure, or activate an existing figure   
    plt.scatter(df.Age[df.AHD==1], df.MaxHR[df.AHD==1], c="salmon") # Scatter with postive examples
    plt.scatter(df.Age[df.AHD==0], df.MaxHR[df.AHD==0], c="lightblue") # Scatter with negative examples
    plt.xlabel("Age"); plt.ylabel("Maximum Heart Rate")
    plt.title("Heart Disease in Function of Age and Maximum Heart Rate")
    plt.legend(["Disease", "No Disease"]);
    plt.tight_layout(); plt.grid(True); plt.show()
    
def Thal_feature_analysis(df):
    
    plt.figure(figsize=(18,9)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.5) # Set the parameters that control the scaling of plot elements
    sns.countplot(df['Thal']) # Show the counts of observations in each categorical bin using bars
    plt.title("Types of Thal - 0 (NA), 1 (fixed), 2 (normal) and 3 (reversable)")
    plt.tight_layout(); plt.grid(True); plt.show()
    
def AHD_feature_analysis(df):
    
    plt.figure(figsize=(18,9)) # Create a new figure, or activate an existing figure
    sns.set_context('notebook',font_scale = 1.5) # Set the parameters that control the scaling of plot elements
    sns.countplot(df['AHD']) # Show the counts of observations in each categorical bin using bars
    plt.title("Acquired Heart Disease (AHD):- Output class, 0 - No (Normal) and 1 - Yes (Heart Disease)")
    plt.tight_layout(); plt.grid(True); plt.show()
    
    print("\nThe ratio between 1 and 0 is much less than 1.5 which indicates that the target feature is not imbalanced. So for a balanced dataset, we can use accuracy_score as evaluation metrics for our model.")

# Function to split the dataset:-
def splitdataset(df):
    
    # Create feature and target arrays:-
    X = df.drop('AHD', axis=1) # Data - Assume Independent Variable
    y = df.AHD # Target - Assume Dependent Variable
    
    # Split into training and test set:-
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)
    
    return X, y, X_train, X_test, y_train, y_test

# Function to calculate accuracy:-
def cal_accuracy(y_test, y_pred):
      
    print("\nConfusion Matrix -\n", confusion_matrix(y_test, y_pred)) # Compute confusion matrix to evaluate the accuracy of a classification
      
    print ("\nAccuracy - ", accuracy_score(y_test, y_pred)*100) # Accuracy classification score
      
    print("Report -\n", classification_report(y_test, y_pred)) # Build a text report showing the main classification metrics

def k_nearest_neighbor(df):
    
    X, y, X_train, X_test, y_train, y_test = splitdataset(df)
    neighbors = np.arange(1, 11) # Return evenly spaced values within a given interval
    
    # Return a new array of given shape, without initializing entries:-
    train_accuracy = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))
     
    classifier = KNeighborsClassifier() # Classifier implementing the k-nearest neighbors vote 
    classifier.fit(X_train, y_train) # Fit the k-nearest neighbors classifier from the training dataset   
    y_pred= classifier.predict(X_test) # Predict the test set result
    cal_accuracy(y_test, y_pred) # Function to calculate accuracy
    
    # Loop over K values:-    
    for i, k in enumerate(neighbors):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
         
        # Compute training and test data accuracy - Return the mean accuracy on the given test data and labels:-
        train_accuracy[i] = knn.score(X_train, y_train)
        test_accuracy[i] = knn.score(X_test, y_test)    
     
    # Generate plot:-    
    plt.figure(); plt.style.use('seaborn');
    plt.xlabel("n_neighbors"); plt.ylabel("Accuracy")
    plt.title("K-Nearest Neighbor (KNN) - Model Accuracy for Different K Values")
    plt.plot(neighbors, test_accuracy, label = 'Testing Dataset Accuracy')
    plt.plot(neighbors, train_accuracy, label = 'Training Dataset Accuracy')
    plt.legend(); plt.tight_layout(); plt.show()
    
# Function to make predictions:-
def prediction(X_test, clf_object):
  
    # Predicton on test with gini index:-
    y_pred = clf_object.predict(X_test)
    print("Predicted Values -"); print(y_pred)
    return y_pred

def naive_bayes_classifier(df):
    
    X, y, X_train, X_test, y_train, y_test = splitdataset(df)
    
    # Fitting Naive Bayes to the training set:-
    classifier = GaussianNB() # Gaussian Naive Bayes (GaussianNB)  
    classifier.fit(X_train, y_train) # Fit Gaussian Naive Bayes according to X, y    
    y_pred= classifier.predict(X_test) # Predict the test set result
    cal_accuracy(y_test, y_pred) # Function to calculate accuracy
    
def support_vector_machine(df):
    
    X, y, X_train, X_test, y_train, y_test = splitdataset(df)
    
    # Fitting the SVM classifier to the training set:-
    classifier = SVC(kernel='linear', random_state=0) # C-Support Vector Classification  
    classifier.fit(X_train, y_train) # Fit the SVM model according to the given training data      
    y_pred = classifier.predict(X_test) # Predict the test set result
    cal_accuracy(y_test, y_pred) # Function to calculate accuracy
    
def decision_tree(df):

    def train_using_gini(X_train, X_test, y_train):
      
        clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 3, min_samples_leaf = 5) # Create the classifier object
        clf_gini.fit(X_train, y_train) # Perform training        
        return clf_gini
    
    def train_using_entropy(X_train, X_test, y_train):
      
        clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,    max_depth = 3, min_samples_leaf = 5) # Decision tree with entropy
        clf_entropy.fit(X_train, y_train) # Perform training        
        return clf_entropy
        
    X, y, X_train, X_test, y_train, y_test = splitdataset(df)
    clf_gini = train_using_gini(X_train, X_test, y_train)
    clf_entropy = train_using_entropy(X_train, X_test, y_train)

    print("\n\t\t\t\t\t\t\t(i) Results Using Gini Index:-\n")      
    y_pred_gini = prediction(X_test, clf_gini) # Function to make predictions
    cal_accuracy(y_test, y_pred_gini) # Function to calculate accuracy
      
    print("\n\t\t\t\t\t\t\t(ii) Results Using Entropy:-\n")
    y_pred_entropy = prediction(X_test, clf_entropy) # Function to make predictions
    cal_accuracy(y_test, y_pred_entropy) # Function to calculate accuracy

def auc_roc_curve(df):
    
    model1 = KNeighborsClassifier(n_neighbors=3) # Classifier implementing the k-nearest neighbors vote
    model2 = GaussianNB() # Gaussian Naive Bayes (GaussianNB) 
    model3 = SVC(kernel='linear', random_state=0, probability=True) # C-Support Vector Classification
    model4 = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = 3, min_samples_leaf = 5) # A decision tree classifier
    model5 = DecisionTreeClassifier(criterion = "entropy", random_state = 100,    max_depth = 3, min_samples_leaf = 5) # A decision tree classifier
    
    X, y, X_train, X_test, y_train, y_test = splitdataset(df)
    
    # Fit Model:-
    model1.fit(X_train, y_train)
    model2.fit(X_train, y_train)
    model3.fit(X_train, y_train)
    model4.fit(X_train, y_train)
    model5.fit(X_train, y_train)
    
    # Predict Probabilities:-
    pred_prob1 = model1.predict_proba(X_test)
    pred_prob2 = model2.predict_proba(X_test)
    pred_prob3 = model3.predict_proba(X_test)
    pred_prob4 = model4.predict_proba(X_test)
    pred_prob5 = model5.predict_proba(X_test)
    
    # ROC Curve For Models:-
    fpr1, tpr1, thresh1 = roc_curve(y_test, pred_prob1[:,1], pos_label=1)
    fpr2, tpr2, thresh2 = roc_curve(y_test, pred_prob2[:,1], pos_label=1)
    fpr3, tpr3, thresh3 = roc_curve(y_test, pred_prob3[:,1], pos_label=1)
    fpr4, tpr4, thresh4 = roc_curve(y_test, pred_prob4[:,1], pos_label=1)
    fpr5, tpr5, thresh5 = roc_curve(y_test, pred_prob5[:,1], pos_label=1)
    
    # ROC Curve for tpr = fpr:-
    random_probs = [0 for i in range(len(y_test))]
    p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
    
    # AUC Scores:-
    auc_score1 = roc_auc_score(y_test, pred_prob1[:,1])
    auc_score2 = roc_auc_score(y_test, pred_prob2[:,1])
    auc_score3 = roc_auc_score(y_test, pred_prob3[:,1])
    auc_score4 = roc_auc_score(y_test, pred_prob4[:,1])
    auc_score5 = roc_auc_score(y_test, pred_prob5[:,1]) 
    
    print("\nAUC Scores-")
    print("(i) K-Nearest Neighbor (KNN): ", auc_score1)
    print("(ii) Naive Bayes Classifier: ", auc_score2)
    print("(iii) Support Vector Machine (SVM): ", auc_score3)
    print("(iv) Decision Tree Using Gini Index: ", auc_score4)
    print("(v) Decision Tree Using Entropy: ", auc_score5)
    
    # Plot ROC Curves:-
    plt.style.use('seaborn')
    
    plt.plot(fpr1, tpr1, linestyle='--', color='red', label='K-Nearest Neighbor (KNN): ' + str(round(auc_score1*100, 2)) + '%')
    plt.plot(fpr2, tpr2, linestyle='--', color='blue', label='Naive Bayes Classifier: ' + str(round(auc_score2*100, 2)) + '%')
    plt.plot(fpr3, tpr3, linestyle='--', color='green', label='Support Vector Machine (SVM): ' + str(round(auc_score3*100, 2)) + '%')
    plt.plot(fpr4, tpr4, linestyle='--', color='magenta', label='Decision Tree Using Gini Index: ' + str(round(auc_score4*100, 2)) + '%')
    plt.plot(fpr5, tpr5, linestyle='--', color='brown', label='Decision Tree Using Entropy: ' + str(round(auc_score5*100, 2)) + '%')
    plt.plot(p_fpr, p_tpr, linestyle='--', color='yellow')
    
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive rate")  
    plt.title("ROC Curve"); plt.legend(loc='best'); plt.show()

    
# Driver Code: main():-
def main():
    
    heading = "Attribute Information"
    print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    
    print("\n1. Age (in years)")
    print("2. Sex:- Two values, 0 (Female) and 1 (Male)")
    print("3. ChestPain:- Four values - 0 (asymptomatic), 1 (nonanginal), 2 (nontypical) and 3 (typical)")
    print("4. RestBP:- Resting Blood Pressure (mm Hg on admission to the hospital)")
    print("5. Chol:- Serum Cholestorol Measurement (mg/dl)")
    print("6. Fbs:- Fasting Blood Sugar > 120 mg/dl, 0 (False) and 1 (True)")
    print("7. RestECG:- Resting Electrocardiographic Results, \n\t\t0 (Showing probable or definite left ventricular hypertrophy by Estes’ criteria), \n\t\t1 (Normal), and \n\t\t2 (Having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV))")
    print("8. MaxHR:- Maximum Heart Rate Achieved")
    print("9. ExAng:- Exercise Induced Angina, 0 (No) and 1 (Yes)")
    print("10. Oldpeak:- ST depression induced by exercise relative to rest (‘ST’ relates to positions on the ECG plot)")
    print("11. Slope:- The slope of the peak exercise ST segment, 1 (Downsloping), 2 (Flat) and 3 (Upsloping)")
    print("12. Ca:- The number of major vessels, 0, 1, 2 and 3 colored by flourosopy.")
    print("13. Thal:- A blood disorder called thalassemia, \n\t\t0 - NA (Dropped from the dataset previously), \n\t\t1 - fixed (No blood flow in some part of the heart), \n\t\t2 - normal (Normal blood flow), and \n\t\t3 - reversable (A blood flow is observed but it is not normal)")
    print("14. Acquired Heart Disease (AHD):- Output class, 0 - No (Normal) and 1 - Yes (Heart Disease)")
    
    # Importing the data set:-
    heading = "Heart Disease Prediction Data Set"
    dataFrame = pd.read_csv("heart.csv")
    print_csv_file(dataFrame, heading)
    
    heading = "Exploratory Data Analysis"; print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    exploratory_data_analysis(dataFrame)
    
    heading = "Age Feature Analysis"; print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    Age_feature_analysis(dataFrame)
    
    heading = "Sex Feature Analysis"; print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    Sex_feature_analysis(dataFrame)
    
    heading = "Chest Pain Feature Analysis"; print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    ChestPain_feature_analysis(dataFrame)
    
    heading = "Maximum Heart Rate Achieved Feature Analysis"; print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    MaxHR_feature_analysis(dataFrame)
    
    heading = "Thalassemia Feature Analysis"; print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    Thal_feature_analysis(dataFrame)
    
    heading = "Acquired Heart Disease (AHD) Feature Analysis"; print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    AHD_feature_analysis(dataFrame)
    
    heading = "Complete Description of Continuous and Categorical Data"; print("\n")
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    
    categorical_val = []
    continous_val = []
    heading = "_"; print("\n")
    
    for column in dataFrame.columns:
        print('{:s}'.format('\u0332'.join(heading.center(100))))
        print(f"{column} : {dataFrame[column].unique()}")
        if len(dataFrame[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continous_val.append(column)
            
    plt.figure(figsize=(15, 15)) # Create a new figure, or activate an existing figure
    for i, column in enumerate(categorical_val, 1):
        plt.subplot(3, 3, i)
        dataFrame[dataFrame["AHD"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
        dataFrame[dataFrame["AHD"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
        plt.legend()
        plt.xlabel(column)
        
    print("\n1. Chest Pain (ChestPain):- People with chest pain equal to 1, 2 and 3 are more likely to have heart disease than people with chest pain equal to 0.")
    print("2. Resting Electrocardiographic Results (RestECG):- People with a value of 0 (showing probable or definite left ventricular hypertrophy by Estes’ criteria, which can range from mild symptoms to severe problems) are more likely to have heart disease.")
    print("3. Exercise Induced Angina (ExAng):- People with a value of 0 (No) have heart disease more than people with a value of 1 (Yes).")
    print("4. Slope:- People with a slope value equal to 1 (Down-sloping - Signs of Unhealthy Heart) are more likely to have heart disease than people with a slope value equal to 2 (Up-sloping - Better Heart Rate with Exercise) or 3 (Flat - Minimal Change, Typical Healthy Heart).")
        
    plt.figure(figsize=(15, 15)) # Create a new figure, or activate an existing figure
    for i, column in enumerate(continous_val, 1):
        plt.subplot(3, 2, i)
        dataFrame[dataFrame["AHD"] == 0][column].hist(bins=35, color='blue', label='Have Heart Disease = NO', alpha=0.6)
        dataFrame[dataFrame["AHD"] == 1][column].hist(bins=35, color='red', label='Have Heart Disease = YES', alpha=0.6)
        plt.legend()
        plt.xlabel(column)
        
    print("\n1. Age:- Elderly people (>50 years) are more likely to have heart disease.")
    print("2. Resting Blood Pressure (RestBP):- Anything between 120-140 (mm Hg on admission to the hospital) is typically cause for concern.")
    print("3. Serum Cholestorol Measurement (Chol):- Anything between 200-300 (mg/dl) is typically cause for concern.")
    
    heading = "Feature Engineering"    
    categorical_val.remove('AHD')
    new_dataFrame = pd.get_dummies(dataFrame, columns = categorical_val)
    print_csv_file(new_dataFrame, heading)
    print("\nRemoved the AHD (Target) column from our set of features and categorized all the categorical variables using the get dummies method which will create a separate column for each category.\n")
    
    heading = "Standard Feature Scaling"
    sc = StandardScaler()
    col_to_scale = ['Age', 'RestBP', 'Chol', 'MaxHR', 'Oldpeak']
    new_dataFrame[col_to_scale] = sc.fit_transform(new_dataFrame[col_to_scale])
    print_csv_file(new_dataFrame, heading)
    
    print("\n"); heading = "K-Nearest Neighbor (KNN)"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    k_nearest_neighbor(new_dataFrame)
    
    print("\n"); heading = "Naive Bayes Classifier"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    naive_bayes_classifier(new_dataFrame)
    
    print("\n"); heading = "Decision Tree"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    decision_tree(new_dataFrame)
    
    print("\n"); heading = "Support Vector Machine (SVM)"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    support_vector_machine(new_dataFrame)
    
    print("\n"); heading = "AUC-ROC Curve"
    print('{:s}'.format('\u0332'.join(heading.center(100))))
    auc_roc_curve(new_dataFrame)

# Call main function ; Execution starts here.
if __name__=="__main__":
    main()

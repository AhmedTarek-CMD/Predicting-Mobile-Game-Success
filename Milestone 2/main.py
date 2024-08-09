import pandas as pd
from typing import List
from preprocessing.traindata import DataPreprocessing
from Visualization.visualizing import GraphPlotter
from preprocessing.testdata import DataPreprocessing_Test
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import timeit
import pickle


print("####################################################################")
x = float(input("Enter 1 To Training Or 2 To Testing : "))
print("####################################################################")
if x == 1:
    data_preprocess = DataPreprocessing('Data/ms2-games-tas-test-v2.csv')
    data_preprocess.preprocess_all()
    X_train, y_train, X_test, y_test = data_preprocess.split_data_then_scale(0.2)

    train_Time = []
    train_accurices = []
    #########################Logistic Regression############################

    log_C_values = [0.1, 1, 10]
    solvers = ['sag', 'liblinear', 'lbfgs']
    logreg_models = {}
    print("-------------------Logistic Regression-------------------")

    for sol in solvers:
        for C in log_C_values:
            start = timeit.default_timer()
            logreg = LogisticRegression(solver=sol, C=C)
            logreg.fit(X_train, y_train)
            end = timeit.default_timer()
            train_Time.append(end-start)
            y_pred = logreg.predict(X_test)

            logaccuracy = accuracy_score(y_test, y_pred)
            train_accurices.append(logaccuracy)
            logmodel_name = f'{sol}_{C}'
            logreg_models[logmodel_name] = (logreg, logaccuracy)

            print(f"Solver: {sol}, C: {C}")
            print("----------------------------")
            print("Accuracy score:", logaccuracy)

    # Save the objects to a file using pickle
    logfilename = 'models/logreg_objects.pickle'
    with open(logfilename, 'wb') as file:
        pickle.dump((log_C_values, solvers, logreg_models), file)

    #########################SVM############################

    kernel_values = ['linear', 'rbf', 'sigmoid']
    SVM_C_values = [0.1, 1, 10]
    svm_models = {}

    print("-------------------SVM-------------------")

    for kernel in kernel_values:
        for C in SVM_C_values:
            start = timeit.default_timer()
            svm = SVC(kernel=kernel, C=C)
            svm.fit(X_train, y_train)
            end = timeit.default_timer()
            train_Time.append(end-start)
            y_pred = svm.predict(X_test)

            svmaccuracy = accuracy_score(y_test, y_pred)
            train_accurices.append(svmaccuracy)
            
            svmmodel_name = f'{kernel}_{C}'
            svm_models[svmmodel_name] = (svm, svmaccuracy)

            print(f"Kernel: {kernel}, C: {C}")
            print("----------------------------")
            
            print("Accuracy score:", svmaccuracy)


    # Save the objects to a file using pickle
    svmfilename = 'models/svm_objects.pickle'
    with open(svmfilename, 'wb') as file:
        pickle.dump((kernel_values, SVM_C_values, svm_models), file)
    #############################Naive Bayes################################

    var_smoothing_values = [1e-9, 1e-7, 1e-5]
    priors = [None, [0.3, 0.3, 0.4], [0.3, 0.3, 0.4]]
    NB_models = {}

    print("-------------------Naive Bayes-------------------")

    for prior in priors:
        for var_smoothing in var_smoothing_values:
            nb = GaussianNB(var_smoothing=var_smoothing, priors=prior)

            # Fit the model on the training data
            nb.fit(X_train, y_train)

            # Predict on the test data
            y_pred = nb.predict(X_test)

            # Calculate accuracy score and mean squared error
            NBaccuracy = accuracy_score(y_test, y_pred)
            
            NBmodel_name = f'{prior}_{var_smoothing}'
            NB_models[NBmodel_name] = (nb, NBaccuracy)

            print(f"Priors: {prior}, Var smoothing: {var_smoothing}")
            print("----------------------------")       
            print("Accuracy score:", NBaccuracy)

    # Save the objects to a file using pickle
    NBfilename = 'models/NB_objects.pickle'
    with open(NBfilename, 'wb') as file:
        pickle.dump((var_smoothing_values, priors, NB_models), file)
        
    ################### Decision Tree #######################   
    max_depth_values = [2, 5, 10]
    min_samples_split_values = [2, 5, 10]
    criterion_values = ['gini', 'entropy']
    dt_models = {}

    print("-------------------Decision Tree-------------------")

    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for criterion in criterion_values:
                dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, criterion=criterion)

                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)

                dt_accuracy = accuracy_score(y_test, y_pred)
                dt_model_name = f'{max_depth}_{min_samples_split}_{criterion}'
                dt_models[dt_model_name] = (dt, dt_accuracy)

                print(f"Max depth: {max_depth}, Min samples split: {min_samples_split}, Criterion: {criterion}")
                print("----------------------------")
                print("Accuracy score:", dt_accuracy)

    # Save the objects to a file using pickle
    dt_filename = 'models/dt_objects.pickle'
    with open(dt_filename, 'wb') as file:
        pickle.dump((max_depth_values, min_samples_split_values, criterion_values, dt_models),file)
        
    #------------------------------------------------------------------------------
    hyperparameters = ['sag_0.1', 'sag_1', 'sag_10', 'liblinear_0.1', 'liblinear_1', 'liblinear_10', 'lbfgs_0.1', 'lbfgs_1', 'lbfgs_10','linear_0.1', 'linear_1', 'linear_10', 'rbf_0.1', 'rbf_1', 'rbf_10', 'sigmoid_0.1', 'sigmoid_1', 'sigmoid_10']


    # Create an instance of the GraphPlotter class
    plotter = GraphPlotter(train_accurices, train_Time, 0, hyperparameters)

    # Plot the accuracy scores
    plotter.plot_accuracy()

    # Plot the total training times
    plotter.plot_train_time()

else:
    
    data_preprocess2 = DataPreprocessing_Test('Data/ms2-games-tas-test-v2.csv')
    data_preprocess2.preprocess_all()
    print(data_preprocess2.get_data())
    x , y = data_preprocess2.DataScaling()

    test_Time = []
    ############################# Naive Bayes################################
    
    
    print("-------------------Naive Bayes-------------------")

    var_smoothing_values, priors, NB_models = pd.read_pickle ('models/NB_objects.pickle')

    # Use the loaded objects to make predictions
    for prior in priors:
        for var_smoothing in var_smoothing_values:
            NBmodel_name = f'{prior}_{var_smoothing}'
            nb, accuracy = NB_models[NBmodel_name]
            y_pred = nb.predict(x)

            accuracy = accuracy_score(y, y_pred)
            print(f"Model: {NBmodel_name}")
            print("----------------------------")
            print("Accuracy score:", accuracy)
    
    
    ######################### Logistic Regression #########################
    
    print("-------------------Logistic Regression-------------------")
    start = timeit.default_timer()
    log_C_values, solvers, logreg_models = pd.read_pickle ('models/logreg_objects.pickle')

    # Use the loaded objects to make predictions
    for sol in solvers:
        for C in log_C_values:
            logmodel_name = f'{sol}_{C}'
            logreg, accuracy = logreg_models[logmodel_name]
            y_pred = logreg.predict(x)
            end = timeit.default_timer()
            test_Time.append(end-start)
            accuracy = accuracy_score(y, y_pred)
            print(f"Model: {logmodel_name}")
            print("----------------------------")
            print("Accuracy score:", accuracy)
            
    ############################# SVM ###################################
    
    print("------------------------SVM---------------------")
    start = timeit.default_timer()

    kernel_values, SVM_C_values, svm_models = pd.read_pickle ('models/svm_objects.pickle')
  
    for kernel in kernel_values:
        for C in SVM_C_values:
            svmmodel_name = f'{kernel}_{C}'
            svm, svmaccuracy = svm_models[svmmodel_name]
            y_pred = svm.predict(x)
            end = timeit.default_timer()
            test_Time.append(end-start)
            accuracy = accuracy_score(y, y_pred)
            print(f"Model: {svmmodel_name}")
            print("----------------------------")
            print("Accuracy score:", accuracy)
            


   ################### Decision Tree #######################   
    print("-------------------Decision Tree-------------------")
    max_depth_values, min_samples_split_values, criterion_values,dt_models = pd.read_pickle ('models/dt_objects.pickle')
    
    
    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            for criterion in criterion_values:

                dt_model_name = f'{max_depth}_{min_samples_split}_{criterion}'
                dt, dt_accuracy = dt_models[dt_model_name]
                dt_y_pred = dt.predict(x)
                dt_accuracy = accuracy_score(y,dt_y_pred)
                print(f"Max depth: {max_depth}, Min samples split: {min_samples_split}, Criterion: {criterion}")
                print("----------------------------")
                print("Accuracy score:", dt_accuracy)
   #-----------------------------------------------------------------         
    # Create an instance of the GraphPlotter class
    hyperparameters = ['sag_0.1', 'sag_1', 'sag_10', 'liblinear_0.1', 'liblinear_1', 'liblinear_10', 'lbfgs_0.1', 'lbfgs_1', 'lbfgs_10','linear_0.1', 'linear_1', 'linear_10', 'rbf_0.1', 'rbf_1', 'rbf_10', 'sigmoid_0.1', 'sigmoid_1', 'sigmoid_10']

    plotter = GraphPlotter(0, 0, test_Time, hyperparameters)

    # Plot the total training times
    plotter.plot_test_time()   

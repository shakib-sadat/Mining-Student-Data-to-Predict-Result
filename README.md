# Mining-Student-Data-to-Predict-Result
Machine Learning Final Project

This project aims to develop a predictive model to predict the academic performance of mining students using student data. We analyze a dataset of student records from a high school in the Portugal. To conduct this research, supervised and numeric data were collected from Kaggle having 395 instances, 28 attributes and 1 class attribute. The dataset has no missing values. The final dataset was generated by eliminating outliers and scaling the data. 

We used Naïve bayes, KNN, SVM, Kernel SVM, ANN, Logistic Regression, Decision tree and Random Forest classifiers to build different models. The training and test set ratio was 75:25. Among them, Decision tree and Random Forest algorithm performed best with 100% accuracy on test dataset. Lastly, ROC curve and AUC are also compared among applied algorithms. A dataset with large number of instances would be compatible to analyze the performance metrices more accurately.  

![image](https://user-images.githubusercontent.com/62327880/208235523-13cfda8b-0796-407d-90fd-f805aa7c7c61.png)
Fig: Flowchart of the solution

![image](https://user-images.githubusercontent.com/62327880/208235531-13b7f495-f025-4606-9518-6dabca15d867.png)
Fig: The preprocessed student related variables

![image](https://user-images.githubusercontent.com/62327880/208235540-049aed50-c51a-4f93-9d42-04d7e55b1a14.png)
Fig: Individual histogram of the attributes 

![image](https://user-images.githubusercontent.com/62327880/208235554-3a3bd441-292b-40e4-b491-d8011ce36592.png)
Fig: Histogram of the attributes altogether

![image](https://user-images.githubusercontent.com/62327880/208235564-d1a8b8ec-244a-44c0-ba53-030a68835985.png)
Fig: Bar plot of class attribute

![image](https://user-images.githubusercontent.com/62327880/208235573-b234ea29-3674-4816-b9db-28610d63c02e.png)
Fig: Correlation Heatmap

![image](https://user-images.githubusercontent.com/62327880/208235599-5c40fdee-cb5e-4e4d-b00a-63327bd04b27.png)
Fig: Confusion matrix for Decision Tree and Random Forest classifier

![image](https://user-images.githubusercontent.com/62327880/208235586-014c2bf6-3744-4ec5-94db-6884b7b4e573.png)
Fig: ROC Curve and AUC of applied classifiers

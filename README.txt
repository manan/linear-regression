>>
>>

I have committed two code snippets and two datasets to the repository and also a picture of the plot returned from calling classifier.plot() in the first snippet.

Using the framework is quite easy and here are the basic steps:
1. Initialize the classifier: lr.LinearRegression()
2. Load some data like ex1data1.txt or ex1data2.txt: classifier.load_data('filepath')
3. You can plot the data if you only have one X column and one Y column like ex1data.txt but you cannot plot data with multiple x columns and one Y column like ex1data2.txt: classifier.plot()
4. You can predict outcomes in either case: classifier.predict(matrix-shaped-(1xn))

Look at ex1data1.txt and ex1data2.txt to understand how your data should look. The last column is assumed to the Y column (and all the previous ones are assumed to be X columns) unless you give two files with X and Y matrice/vectors. 


>>
>>

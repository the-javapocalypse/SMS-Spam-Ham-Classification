# SMS-Spam-Ham-Classification
SMS Spam/Ham Classification using Multinomial Naive Bayes (from scratch) in Python

# Description
In this project, we will create three models based on Multinomial NB:
1. First model will classify using the standard formula for Multinomial Naive Bayes.

2. The second model will take in account the length of documents as features, since length have a visible effect on class (Spam or Ham) as demonstrated by the graph in code. This will increase accuracy of our model.
     
3. The third model will take the length along with the term frequency as features for classification. This will further increase accuracy of our model.

## Built With
* Pandas
* Numpy
* NLTK
* Matplotlib
* Seaborn
* Collections

Use pip to install any misisng module.

## Usage
Download or Clone the repo. SMSSpamCollection.txt contains raw sms data. This data is stored in data.csv in an organized form. The folder should contain a python script and jupyter notebook which contains the NB Classification code. Start by initializing an object for custom NaiveBayes Class.

The NaiveBayes.fit method takes two parameters, the trainig dataset and an integer that specify which model to use.
nb.fit(training_data,1) will use first model, which is the basic Multinomial Naive Bayes.
nb.fit(training_data,2) will use second model which takes in account length as a feature.
nb.fit(training_data,3) will use third model which takes in account length and term frequency as features.

NaiveBayes.validate method is used for batch validation.

NaiveBayes.predict method is used to preict a single sms. It returns 1 if the sms is spam else 0.

NaiveBayes.most_common_spam(K) and NaiveBayes.most_common_ham(K) returns most common K spam and ham terms respectively.



## Contributing
1. Fork it
2. Create your feature branch: git checkout -b my-new-feature
3. Commit your changes: git commit -am 'Add some feature'
4. Push to the branch: git push origin my-new-feature
5. Submit a pull request

## Authors
Muhammad Ali Zia

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

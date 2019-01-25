# Project4 

# FIFA 19 Ultimate Team Selector

### What is the main goal of our project and what hypothesis are we trying to test?

- Our hypothesis is that picking a player, using the easy "Overall" stats, is not the best investment for your game coins.  In fact, the majority of people who play FIFA 19 tend to make that mistake, since the "Overall" Rating is the main stat that appears on the front of player cards which makes it an easy choice for most recreational players. 
- Our hypothesis is that if you dig deeper into the individual player skills and carefully curate the optimal skill combo for each position on the field, you will end up with a much stronger combination of players and better team chemistry.

- Once we prove our hypothesis, we want to build a player recommendation system where you can input your budget and the position which you want to buy a player for and our selector will return a list of the top recommended players based on individual skills and not Overall rating.

- Lastly, we would like to automate the process via an algorithm which can automatically build a team based on maximizing the specific FIFA stats that are highly important for that position and based on the team formation that you want to play with and whether you want to alocate more budget towards your defensive or offensive players. 

**Obtaining Data:** 

We found a csv on Kaggle with detailed players stats for over 18,000 players in FIFA 19. After cleaning up the data, standardazing our features and creating our categorical variables,  we ended up with 17907 players with 53 features

**Outline of our process:** 

- Break down the DataFrame into data for the four main positions - GK, DF, MD, ST
- Run K-Means clustering to see what features the model will group players by
- Run four different regressions models which aim to predict a player's value based on 53 player attributes
- Isolate the top 15 features for each position by the weights that each model assigned
- Combined all the features and curate a list of the unique features
- Run our best model with those curated features on the four dataframes
- Apply the weights of our final model (XGBoost with 95% R2_score) to the important features for each position and use the resulting value as a “Selector” score, which we believe would be a "smarter" way of finding optimal players. 
- Show Graphs that compare teams built using our Algorithm vs. teams built using Overall rating

### Feature Engineering and Model Optimization:

The first step was to run a **K-Means Unsupervised** model and four regression models, which can help us isolate the top 15 player attributes for each position **Ridge Regression, Random Forest, SVM, and XGBoost***. Below you can see a few samples of those features in barcharts:

![](https://github.com/Botafogo1894/Project4/blob/master/pictures/RF_GK2.png)
![](https://github.com/Botafogo1894/Project4/blob/master/pictures/XGB_DF2.png)
![](https://github.com/Botafogo1894/Project4/blob/master/pictures/SVM_MD2.png)
![](https://github.com/Botafogo1894/Project4/blob/master/pictures/Ridge_ST2.png)

As you can see from the graph, PCA didn't improve performance in either model, so we decided to not use PCA moving forward.

Next things we wanted to do was **GridSearch** on the three top-performing models and pick the model with the combination of parameters that yielded the highest accuracy score.

**GridSearch on the Random Forest** improved performance from **41% to 43%** accuracy.

**GridSearch on the Gradient Boost** improved performance from **45% to 50%** accuracy.

**GridSearch on Naive Bayes** did not generate improved performance because the default parameters are optimal.

## Interpreting and communicating the final results:

Below you can see the graph of our top three models Final Performance after optimization.

![](https://github.com/Botafogo1894/Project3/blob/master/top%203%20models.png)

Our best model, **Gradient Boost after GridSearch yielded 50% accuracy**, which is just about four times better than random guessing, 12.5%. Even though it's not a stellar number, we are still impressed that given only 7200 lyrics we were able to train a model that can predict out of eight genres with 50% accuracy.

From experimenting with Grid Search and PCA optimizations, we found that Multinomial Naive Bayes was the fastest and simplest model and it yielded only 5% less accurate than the top model.

If you have a lot of features and optimization proves to be computationally expensive, you might opt to pick Naive Bayes. If you have sufficient time and computing power and you want to optimize accuracy score, grid search with Gradient Boost is the way to go.

**PART TWO:** Binary Classification for predicting if a song is a major hit based on song lyrics.

**Modeling:** We wanted to predict whether or not a song was on our top song hit list. We repeated the same steps as in the genre classifier model, this time creating Top 100 list of songs as our target. The target column contains a 1 for every song that was a hit and 0 to indicate songs that were not. We ran the same models and yielded the results found below and were able to predict with 96% accuracy which songs were going to be a hit.

![](https://github.com/Botafogo1894/Project3/blob/master/basic%205%20for%20binary%20problem.png)

Similarly to our first model, Lemmatized performed slightly better even though the results were much closer this time when there were two choices.

When we used PCA on Gaussian Naive Bayes, the performance was much lower with Multinomial Naive Bayes. So, we decided to not use PCA for further optimization because it didn't yield significant accuracy boost.

## Interpreting and communicating the final results:

All of our models had around 96% score. We decided to try PCA and grid search but the results indicated there was not much room for optimization, so when it comes to binary classification it appeared that we could go with either of the Top 3 performing models and not sacrifice much accuracy.

**PART THREE:** Using an Unsupervised Learning model to identify distinctive topics and keywords for each genre

We used gensim.corpora.Dictionary to create a frequency dictionary for the lemmed, tokenized word set. We grabbed keywords from each genre and generated a Topic Model score.

Using the TFIDF Matrix we ran a Topic Modeling LDA algorithm and printed the word clouds for the top Keywords in each genre that the unsupervised algorithm identified below:

![](https://github.com/Botafogo1894/Project3/blob/master/Pop_n_Metal.png)

![](https://github.com/Botafogo1894/Project3/blob/master/Jazz_n_rock.png)

![](https://github.com/Botafogo1894/Project3/blob/master/rnb_n_hip.png)

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
![](https://github.com/Botafogo1894/Project4/blob/master/pictures/Ridge_MD2.png)
![](https://github.com/Botafogo1894/Project4/blob/master/pictures/SVM_ST2.png)

As you can see from the graphs, Overall Rating is an important factor in some models, but not in all and there are a lot of other individual features taht were assigned higher weights than the Overall rating. 

Next things we wanted to do was isolate all these important skills and run our top model - **XGBoos** to get weighted coffeicients for each attribute and multiply those together to create a final "Selector" score, which is a weighted combination of the most important attributes for each position - GK, DF, MD or ST. 

**Next Step:** Plot players' overall scores and price versus their combined slector score

The Graphs below demonstrate that for a good number of the players, there is a somewaht linear correlation between Overall Rating and our Selector score. However, we can also clearly isolate many instances where players with a high Overall Rating, which cost a lot of money, do not actually posses high ratings for the individual skills that are important.

![](https://github.com/Botafogo1894/Project4/blob/master/pictures/COmp_Df2.png)

In the example above, we can see that there is a "value separation area" right around 1.4-1.6 selector score, where you have a decent number of expensive players wiht high overall rating, that have a lower selector score, than some cheaper players towards the left corner of the graph, where you could be getting players with a much better all-around skillset for cheaper, just because their overall rating is not that high, so most recreational players are not drawn to bid up their price.

**Writing Our team Selector Algorithm:** 

To continue our investigation further, we decided to write an algorithm that takes in a budget and a player position that you want to spend that budget on and gives you a recommended list of players, optimizing for selector score. Below you can see the code for this simple algo:

```
def pick_player_by_selector(budget, position, count = 3):
    count = count
    empty = []
    names = [item[0] for item in name_pos if item[1] == position]
    STs = ["ST", "LF", "RF", "CF", "RS", "LS", "LW", "RW"]
    MDs = ["RM", "LM", "CAM", "CDM", "LCM", "RCM", "RDM", "LDM", "RAM", "LAM"]
    DFs = ["CB", "LB", "RB", "RCB", "LCB", "RWB", "LWB"]
    if position == "GK":
        keepers = sorted_players_selector[0]
        for item in keepers:
            if item['name'] in names and item['price'] <= budget and count > 0:
                empty.append(item)
                count -= 1
    elif position in DFs:
        defenders = sorted_players_selector[1]
        for item in defenders:
            if item['name'] in names and item['price'] <= budget and count > 0:
                empty.append(item)
                count -= 1
    elif position in MDs:
        mids = sorted_players_selector[2]
        for item in mids:
            if item['name'] in names and item['price'] <= budget and count > 0:
                empty.append(item)
                count -= 1
    elif position in STs:
        strikers = sorted_players_selector[3]
        for item in strikers:
            if item['name'] in names and item['price'] <= budget and count > 0:
                empty.append(item)
                count -= 1             
    return empty
    
    ```


The Graph below was generated after running our player recommender for the Central Midfielder position - CM. This helps us identify some of those players with high Overall rating, but low Selector score and vice versa:

![](https://github.com/Botafogo1894/Project4/blob/master/pictures/Best_Pick_CM2.png)

From the graph above you can see that there are a few players who cost over 20M, who have a low selector score. Here is the example in reverse below, where we're plotting the players that our algorithm recommends, rating them by Selector Score. You can see that not all of them have a high Overall Rating, but we know from our prior invesstigation on individual Model weights and insights that the players with the higher selector scores posses all the strong individual qualities to make a good defender.

![](https://github.com/Botafogo1894/Project4/blob/master/pictures/CAM_selector.png)


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

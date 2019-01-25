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

**Comapre Overall vs Selector Scores:** Plot players' overall scores and price versus their combined slector score

The Graphs below demonstrate that for a good number of the players, there is a somewaht linear correlation between Overall Rating and our Selector score. However, we can also clearly isolate many instances where players with a high Overall Rating, which cost a lot of money, do not actually posses high ratings for the individual skills that are important.

![](https://github.com/Botafogo1894/Project4/blob/master/pictures/COmp_Df2.png)

In the example above, we can see that there is a "value separation area" right around 1.4-1.6 selector score, where you have a decent number of expensive players wiht high overall rating, that have a lower selector score, than some cheaper players towards the left corner of the graph, where you could be getting players with a much better all-around skillset for cheaper, just because their overall rating is not that high, so most recreational players are not drawn to bid up their price.

**Writing Our Player Selector Algorithm:** 

To continue our investigation further, we decided to write an algorithm that takes in a budget and a player position that you want to spend that budget on and gives you a recommended list of players, optimizing for selector score. Below you can see the code for this simple algo:

```python
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

## Writing our Ultimate Team Builder:

Below you can see the code for the final and most exciting part of our project - using the inishgts from our investigation and our selector score ratings to program an automatic team builder, which takes in the following paramters:

1. Total Budget you want to spent on your team
2. What team formation you want to use (ex. 4-4-2 vs 3-5-2)
3. What portions of your bugdet do you want to allocate for each position.

The default settings below, use the most popular soccer formation 4-4-2 and allocate a bigger portion of the budget on offensive player, which you'd do if you play a more offensive style. 

```python
def pick_team_by_selector(budget, DF = 4, MD = 4, ST = 2, GK_coef = 0.05, DF_coef = 0.2, MD_coef = 0.40, ST_coef = 0.35):
    #split budget accordingly per position
    GK_budget = budget*GK_coef
    DF_budget = budget*DF_coef
    MD_budget = budget*MD_coef
    ST_budget = budget*ST_coef
    
    #Create proper count for each position
    GK_count = 1
    DF_count = DF
    ST_count = ST+1
    MD_count = MD+1
    
    #Kreate our list of players per position
    keepers = sorted_players_selector[0]
    defenders = sorted_players_selector[1]
    mids = sorted_players_selector[2]
    strikers = sorted_players_selector[3]
    
    #Begin building final team
    list_of_names = []
    final_team = []
    for item in keepers:
        if GK_count > 0 and item['price'] <= GK_budget and item['name'] not in list_of_names:
            item['position'] = "GK"
            final_team.append(item)
            list_of_names.append(item['name'])
            GK_budget -= item['price']
            GK_count -= 1
            DF_budget += GK_budget
    for item in defenders:
        if DF_count > 0 and item['price'] <= DF_budget/DF_count and item['name'] not in list_of_names:
            item['position'] = "DF"
            final_team.append(item)
            list_of_names.append(item['name'])
            DF_budget -= item['price']
            DF_count -= 1        
    for item in mids:
        while MD_count == MD+1:
            MD_budget = MD_budget + DF_budget
            MD_count -= 1
        if MD_count > 0 and item['price'] <= MD_budget/MD_count and item['name'] not in list_of_names:
            item['position'] = "MD"
            final_team.append(item)
            list_of_names.append(item['name'])
            MD_budget -= item['price']
            MD_count -= 1
    for item in strikers:
        while ST_count == ST+1:
            ST_budget = ST_budget + MD_budget
            ST_count -= 1
        if ST_count > 0 and item['price'] <= ST_budget/ST_count and item['name'] not in list_of_names:
            item['position'] = "ST"
            final_team.append(item)
            list_of_names.append(item['name'])
            ST_budget -= item['price']
            ST_count -= 1
    leftover_budget = GK_budget + DF_budget + MD_budget + ST_budget
    rating_avg = round(sum([item['selector'] for item in final_team])/len(final_team), 3)
    overall_avg = round(sum([item['overall'] for item in final_team])/len(final_team), 3)
    return dict(team=final_team, budget_left = leftover_budget, AVG_selector=rating_avg, AVG_overall = overall_avg)
```

Below you can see the result of setting a max budget and seeing what the most optimal team looks like:

![](https://github.com/Botafogo1894/Project4/blob/master/pictures/Final_team_selector.png)

All of our models had around 96% score. We decided to try PCA and grid search but the results indicated there was not much room for optimization, so when it comes to binary classification it appeared that we could go with either of the Top 3 performing models and not sacrifice much accuracy.

**PART THREE:** Using an Unsupervised Learning model to identify distinctive topics and keywords for each genre

We used gensim.corpora.Dictionary to create a frequency dictionary for the lemmed, tokenized word set. We grabbed keywords from each genre and generated a Topic Model score.

Using the TFIDF Matrix we ran a Topic Modeling LDA algorithm and printed the word clouds for the top Keywords in each genre that the unsupervised algorithm identified below:

![](https://github.com/Botafogo1894/Project3/blob/master/Pop_n_Metal.png)

![](https://github.com/Botafogo1894/Project3/blob/master/Jazz_n_rock.png)

![](https://github.com/Botafogo1894/Project3/blob/master/rnb_n_hip.png)

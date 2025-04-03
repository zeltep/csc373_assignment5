Acknowledgements:  
- Dr. Natalia Khuri
- scikit-learn [website](https://scikit-learn.org)

# Assignment 5

**Description**  
Estimation of hours played in game, based on features provided in JSON files, with data-centric techniques
used such as removal of outliers and log-transforming the hours (log base 2(hours + 1))

Classification of hours played based on if it is above or below the median hours played for the game.  Additionally,
training models using reviews from before 2015 to evaluate on reviews written 2015 onwards, and the opposite.

Recommendation pipeline used to make predictions of log-transformed hours by using collaborating users

For all of these, we are reporting measures of accuracy and overpredicted/underpredicted instances.

**Output**  
In '../output',   We have two folders, 'data_reports' and 'model_scores'.  In 'data_reports' we have a summary of
the data set.  In 'model_scores' we have three folders, storing the scores for each task we are doing, classification,
estimation, and recommendation.  

The best-performing model will be saved to '../outputs/best_model.pkl'

## Usage

With assignment_5/scripts as the working directory, 

```{bash}
python asssignment_5.py 
```

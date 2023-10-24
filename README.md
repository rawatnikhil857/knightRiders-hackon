
# Fine Tuning LLM for Content Advising

This project fine tunes an LLM which is a training technique to adapt a pre-trained model to perform specific tasks or to improve its performance on a particular task. It involves taking a model that has already been trained on a large dataset and further training it on a smaller, task-specific dataset.

This project specifically using BLOOM 1B7 base model. And the fine tuning is done for the task of content advising based on a user's previous data.

We've implemented parameter efficient fine-tuning (PEFT) using Low-Rank Adaptation (LoRA). LoRA allows to fine tune the LM using very low percentage of parameters. In this project we're only required to train 0.09% of all parameters of bloom-1b7. And we can save model configuration that allows us to use the trained LoRA model without re-training it all over again.

## How to use this project to generate results?

 - Download "Context-Question-AnswerDatabase.csv" database
 - Open google collab link available in "GeneratingResults.ipynb"
 - Connect to a runtime
 - Upload the database to file browser in google collab
 - Refer Prompt Structure and Examples in the notebook for better understanding


## Dataset used

The dataset that is being used by our project is derived from two dataset: the movie dataset and the other one is rating dataset, from MovieLens 20M. Our model is being trained on the combined dataset.

 - Movie dataset
 ![alt text](https://i.ibb.co/ZVwKPNY/image.png)

 - Rating dataset
 ![alt text](https://i.ibb.co/JzJx2WG/image.png)

In this project the LLM is trained using a combined dataset that integrates user interactions and comprehensive movie data. This amalgamation allows the LLM to learn from user preferences, consequently tailoring movie recommendations based on the specific contexts provided.

The combined dataset encompasses user-related information, including watched movies, ratings, and genres of movies. Through this amalgamated dataset, the LLM is trained from the user past history by providing some watched and rated movies as context and some as recommended movies so that LLM recognizes the resemblances between different movies based on user past movies records and recommend some other movies that user has not watched yet.

The combined dataset can be found in "Context-Question-AnswerDatabase.csv"
## Fine tuning using combined dataset

We structure our prompts in a specific way to fine tune our model. We make the language model to learn the structure of the query that will be made and the structure of the answer it's supposed to generate by providing it multiple such prompts and training the parameters accordingly. For example - 

![alt](https://i.ibb.co/2K1zyrZ/image.png)

The entire dataset is mapped to the specific structure shown above. Followed by training the LoRA model by giving the context, question, answer triplet as the prompt.

An implementation can be found in "LoRA_Implementation.ipynb".
## Evaluation Metrics

 - Mean Absolute Error (MAE): For regression problems, MAE measures the average absolute difference between the predicted values and the actual values. Lower MAE indicates better accuracy.    
    MAE = (1 / n) * Σ |actual - predicted|, where n is the number  of data points.

 - Mean Squared Error (MSE): MSE is another metric for regression problems. It measures the average squared difference between the predicted values and the actual values. Lower MSE indicates better accuracy.
    
    MSE = (1 / n) * Σ (actual - predicted)^2, 
    where n is the number of data points.

 - Root Mean Squared Error (RMSE): RMSE is the square root of the MSE. It is also used for regression problems and provides the average error in the same units as the target variable.
    
    RMSE = sqrt(MSE)

 - Percentage of Ratings within a Tolerance: You can also define a tolerance range (e.g., within ±0.5) and calculate the percentage of ratings that fall within this range to assess how well the model predicts ratings within a certain margin of error

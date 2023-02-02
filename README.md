# Mitigate systematicity

In this project- I train 10 ML models to classify candidates that have the potential to be hired

'The data_preprocessing_individual_models.py' is the data preprocessing pipeline along with 10 individual models I have trained to fit the data. I write a custom function to spit out the results and do the necessary processing that can be applied to any dataset having similar structure


THEN BECOMES THE MAIN PROBLEM- WE WANT TO INCLUDE SOME BOUNDED RANDOMNESS IN MODELS 

In traditional hiring systems some candidates on the margin would get a chance because different hiring managers should have different different assessments of the candidate. With the use of ML softwares the hiring system has become automated so some candidates may be systematically deprived of a chance. 

This is the main problem I am trying to solve in this exercise.

I propose two methods for doing so

i) Introduce some bounded randomness through adding a gaussian error term so that a certain percentage of candidates getting flagged as low potential gets flagged as high potential
Here I do a 6.25%, but it the percentage can be adjusted to any number say 10%, 20%, 5%, etc.

ii) Voting Model where at prediction the algorithm choses a subset of models from a set of models (of similar accuracy)

Both of these methods introduce some bounded randomness but I show how in this case choosing option (i) produces better results when models of similar accuracy are not available

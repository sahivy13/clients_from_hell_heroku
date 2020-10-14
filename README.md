<center><h1>Classifier Algorithm Trainer Bot & Predictor</h1></center>

Type of A.I. : Supervised Machine Learning

>Website scrapped: <a href="https://clientsfromhell.net">Clients From Hell</a>

<center><img src="cfh_logo.png" alt="project_logo" height="150"></center>

## Technologies Used
- Scipy
- Streamlit
- Matplotlib
- Plotly
- Pickle
- Beautiful Soup
- Pandas
- Numpy
- Scikit Learn
- NLTK
- Imbalanced Learn
- SQL Alchemy

## Functionalities of the training application

1. These application allows for users to rescrappe the website, although posts have already been saved to Heroku's PostgreSQL server.

2. It allows for the user to choose manually the number of samples it would like to use from a range of going from the lowest amount of samples in a category to the highest. This number will be used to then proceed with over-sample and then under-sample the data in order to not create a bias towards one specific class.

3. Currently, the application can only work locally thus the github repo needs to be clone in user's local system in order to retrain the model to be used in the precition application which will pull the models trained by this application saved in Heroku's PostgreSQL database.

## Functionalities of the prediction application

1. The application will allow for the user to input his/her story as a freelancer by typing it directly to web browser or upload a file.

2. The user can choose from the models shown to be saved in the database to use as the classifier. the app will be displaying also the accuracy score for the models in the database.

## Full Description

This Project was created by me (Sahivy R. Gonzalez) in order to showcase some of the technologies I have learned to use. It consists of the following three parts as of 10/13/2020:

1. Local Streamlit Application (Model Trainer):
    - Due to exceed of Heroku's allowable ram memory, this is activated by running locally the following command: Streamlit run local_main.py
    - Back-End activity:
        1. Scrape the posts and categories from the Clients From Hell website utilizing Beautiful Soup.
        2. Clean the data utilizing NLTK.
        3. Save the data to Heroku's PostgreSQL Server.
        4. Oversample categories under a selected number of samples.
        5. Undersample categories over the same selected number of samples.
        6. Extract features from posts through TfidfVectorizer.
        7. Utilizing K-Folded Cross-Validation, train multiple versions of each type of model to be trained with different parameters, and selecting the version with the best accuracy for each type of model.
        8. Saves the best trained model for each type in the Heroku's PostgresSQL Server, with their best accuracy score, and the best parameters for each type of model. 
    - Front-End activity (Streamlit):
        1. Display the following widgets and text:
            - Title, and description of application.
            - Sidebar with the following:
            <br>
                a. Instructions on how to utilize Plotly graphs.
                <br>
                b. "Re-Scrape" button, allows user to direct the application to re-collect the data from the website scrapped. This would be done automatically if there is no data stored in Heroku's PostgreSQL Server.
                <br>
                c. "Manual" dropdown list, allows user to direct the application to change the number of samples to use per post's category. This triggers a slider to appear that will ask the user to choose the number of samples to have per category in order to oversample and undersample accordingly.
                <br>
                d. "Re-Train" button, allows user to direct the application to clear cache, and retrain models.
        2. Display two Plotly histograms of the following two things:
            - Posts per category before oversampling & undersampling. (Original names of the category labels are shown)
            - Posts per category after oversampling & undersampling. (Category labels have been transformed into numbers for simplification)

2. Heroku Streamlit Application (Model Trainer with disabled the ability to store data, and models in Heroku's PostgreSQL Server): ***PENDING***
    - This can be accessed through: ***[web link to app holder]***
    - The main file for this app is: heroku_trainer_main.py
3. Heroku Streamlit Application (Classifier): ***PENDING***
    - This can be accessed through: ***[web link to app holder]***
    - The main file for this app is: heroku_classifier_main.py
    - Back-End activity:
        1. Pull stored model, and model related info.
        2. Take input from user (word doctument, pdf, or direct input).
        3. Utilize the same functions for pre-processing as trainer.
        4. Pull vectorizer used on training from Heroku's PostgreSQL Server, and use it to vectorize the post inputed by user. ***(This is theoretical because I have never done this before. As I understand from documentation, the same fitted vectorizer must be used to transform post since shape of features must be the same as used in training)***.
        5. Feed vectorized post to the selected trained model, and return label.
    - Front-End activity:
        1. Display the following widgets and text:
            - Title, and description of application.
            - Sidebar with the following:
            <br>
                a. "Choose Model" dropdown menu, allows user to choose which saved model to use as classifier.
                <br>
                b. "Predict" button, instructs the application to feed post to selected model.

## Future Upgrades

***"You understand that the second you look in the mirror and you're happy with what you see, baby, you just lost the battle."*** - Dr. Cox

1. Check what is faster:
    a. Oversample and undersample first, then vectorize after.
    b. The opposite as previous mentioned.
2. Check the delta in accuracy if labels are One-Hot Encoded instead of scalar values.
3. Change "Manual" dropdown list to be a button.
4. Allow user to decide parameters to test for each model, and give a list of values for each parameter.
5. Find a nicer way to show results.

## Post Categories

Each post made on the website is a story from freelancers. These stories are of bad experiences the freelancers had with clients ***in real life...*** and each story is classified as the following:
- Chaotic Good
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-chaotic-good.svg" alt="drawing" width="25"/>
<br>
- Dounces
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-dunces.svg" alt="drawing" width="25"/>
<br>
- Cryptic
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-cryptic.svg" alt="drawing" width="25"/>
<br>
- Homophobes
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-homophobes.svg" alt="drawing" width="25"/>
<br>
- Sexist
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-sexist.svg" alt="drawing" width="25"/>
<br>
- Criminals
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-criminal.svg" alt="drawing" width="25"/>
<br>
- Deadbeats
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-deadbeats.svg" alt="drawing" width="25"/>
<br>
- Racists
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-homophobes.svg" alt="drawing" width="25"/>
<br>
- Ingrates
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-ingrates.svg" alt="drawing" width="25"/>
<br>
- Frenemies
<br>
<img src="https://clientsfromhell.net/wp-content/uploads/2018/09/ic-frenemies.svg" alt="drawing" width="25"/>
<br>

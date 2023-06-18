import wandb
import pandas as pd
import os
import shutil
import joblib
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from params import train_location, test_location, project_name


class TitanicClassification():
    def __init__(self, project_name, train_location, test_location):
        self.project_name = project_name
        self.train_location = train_location
        self.test_location = test_location

    def create_data_folder(self):
        path = "my_dataset"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)

    def perform_training(self, datadir, run):

        train_data = pd.read_csv(os.path.join(datadir, 'train.csv'))
        test_data = pd.read_csv(os.path.join(datadir, 'test.csv'))

        features_drop = ['PassengerId', 'Name', 'Cabin', 'Ticket']
        X_train = train_data.drop(features_drop, axis=1)
        y_train = train_data['Survived']
        X_train.drop('Survived', axis=1, inplace=True)
        X_test = test_data.drop(features_drop, axis=1)

        # Conversion of categorical values of sex feature to numerical values
        mapping = {'male': 1, 'female': 0}
        X_train['Sex'] = X_train['Sex'].map(mapping)
        X_test['Sex'] = X_test['Sex'].map(mapping)

        X_train.Age.fillna(X_train.Age.median(), inplace=True)
        X_test.Age.fillna(X_test.Age.median(), inplace=True)

        # Filling the missing values in Embarked feature with the most repeated value
        X_train.fillna(value='S', inplace=True)

        # Filling the missing values of Fare feature with the median in test data
        X_test.Fare.fillna(X_test.Fare.median(), inplace=True)

        mapping = {'S': 0, 'C': 1, 'Q': 2}
        X_train['Embarked'] = X_train['Embarked'].map(mapping)
        X_test['Embarked'] = X_test['Embarked'].map(mapping)

        # K-Fold Cross Validation
        k_fold = KFold(n_splits=10, shuffle=True, random_state=0)

        # kNN
        kNN = KNeighborsClassifier(n_neighbors=20)
        score = cross_val_score(kNN, X_train, y_train, cv=k_fold)
        knn_score = sum(score)/len(score)
        run.log({"KNN cross validation score": knn_score})

        # Random Forest
        RF = RandomForestClassifier(n_estimators=100)
        score = cross_val_score(RF, X_train, y_train, cv=k_fold)
        rf_score = sum(score)/len(score)
        run.log({"Random Forest cross validation score": rf_score})

        model_name = ""
        if knn_score > rf_score:
            model_name = 'knn'
        else:
            model_name = 'rf'

        print(X_test.iloc[0])
        print("##############################################################")
        print(X_test.iloc[3])
        # # Predicting on test data
        if model_name == 'rf':
            rf = RF.fit(X_train, y_train)
            y_predicted = rf.predict(X_test)
            print(y_predicted)
            self.model = rf
        else:
            knn = kNN.fit(X_train, y_train)
            y_predicted = rf.predict(X_test)
            print(y_predicted)
            self.model = knn

    def driver_code(self):
        # api_key = os.environ.get('WANDB_API_KEY')

        # run = wandb.init(project=self.project_name, api_key=api_key)

        with wandb.init(project=self.project_name) as run:

            self.create_data_folder()
            train_artifact = run.use_artifact(self.train_location)
            datadir = train_artifact.download('my_dataset')

            test_artifact = run.use_artifact(self.test_location)
            datadir = test_artifact.download('my_dataset')

            self.perform_training(datadir, run)

            joblib.dump(self.model, 'my_first_model.pkl')

            # Save the model to W&B
            # trained_model_artifact = wandb.Artifact(
            #     MODEL_NAME, type="model",
            #     description="trained inception v3",
            #     metadata=dict(cfg))

            best_model = wandb.Artifact(f"model_{run.id}", type='model')
            best_model.add_file('my_first_model.pkl')
            run.log_artifact(best_model)

            # Link the model to the Model Registry
            run.link_artifact(
                best_model, 'model-registry/My first Registered Model')

            shutil.rmtree(datadir)
            os.remove("my_first_model.pkl")
            run.finish()


titanic_classifier = TitanicClassification(
    project_name, train_location, test_location)
titanic_classifier.driver_code()

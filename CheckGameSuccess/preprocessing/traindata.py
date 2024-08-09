import pandas as pd
from typing import List

# label Enconder
from sklearn.preprocessing import LabelEncoder
# Scaling
from sklearn.preprocessing import MinMaxScaler
# splitting
from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings("ignore")


class DataPreprocessing:
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.dataset = None
        self.features = None
        self.label = None
        self.lblencoded = ('Name', 'URL', 'Icon URL',
                           'Description', 'Subtitle')

        self.columns_to_dropped = ['URL', 'ID', 'Subtitle',
                                   'Icon URL', 'Name', 'Description']
        
        self.extra = ['User Rating Count', 'Price',
                      'In-app Purchases', 'min_age', 'Days Since Release']

    def load_dataset(self):
        self.dataset = pd.read_csv(self.filepath, parse_dates=[
                                   'Original Release Date', 'Current Version Release Date'], dayfirst=True)

    def remove_duplicates(self):
        self.dataset.drop_duplicates(inplace=True)

    def Feature_Encoder(self):
        for c in self.lblencoded:
            lbl = LabelEncoder()
            lbl.fit(list(self.dataset[c].values))
            self.dataset[c] = lbl.transform(list(self.dataset[c].values))

    def remove_Unwantedcolumns(self, columns: List[str]):
        self.dataset.drop(columns=columns, inplace=True)

    def preprocess_in_app_purchases(self):
        # Convert each cell into a list of floats, and compute the mean of the list
        self.dataset['In-app Purchases'] = self.dataset['In-app Purchases'].apply(lambda x: sum(
            [float(i) for i in x.split(',')]) if pd.notnull(x) else x)

        # Replace the missing values with the mean of the In-app Purchases column
        mean_value = self.dataset['In-app Purchases'].mean()
        self.dataset['In-app Purchases'] = self.dataset['In-app Purchases'].fillna(
            mean_value)

    def preprocess_languages(self):
        # split the string values into lists of languages
        Lang_temp = self.dataset['Languages'].str.split(',')
        exploded_Lang = Lang_temp.explode()

        # fill missing values with mode value
        mode_value = exploded_Lang.mode()[0]
        self.dataset['Languages'] = self.dataset['Languages'].fillna(
            value=mode_value)
        
    def preprocess_Rate(self):
        self.dataset["Rate"] = self.dataset["Rate"].replace({"Low": 1, "Intermediate": 2, "High": 3})
        
        # self.dataTest["Rate"] = self.dataTest["Rate"].map(lambda x: x.replace(" ", ""))
        # self.dataTest["Rate"] = self.dataTest["Rate"].map(
        #     lambda x: 2 if x == 'High' else 1 if x == 'Intermediate' else 0 if x == 'Low' else x)


    def preprocess_age_rating(self):
        self.dataset['Age Rating'] = self.dataset['Age Rating'].astype(
            str).str.lower()
        # Rename column
        self.dataset.rename(columns={'Age Rating': 'min_age'}, inplace=True)
        self.dataset['min_age'] = self.dataset['min_age'].str.replace(
            '+', '').astype(int)

    def Enconding(self):
        self.dataset['Languages'] = self.dataset.groupby(
            'Languages')['Rate'].transform(lambda x: x.mean())
        self.dataset['Developer'] = self.dataset.groupby(
            'Developer')['Rate'].transform(lambda x: x.mean())
        self.dataset['Primary Genre'] = self.dataset.groupby(
            'Primary Genre')['Rate'].transform(lambda x: x.mean())
        self.dataset['Genres'] = self.dataset.groupby(
            'Genres')['Rate'].transform(lambda x: x.mean())
        
    def preprocess_dates(self):
        # Convert the date columns to datetime format
        StartDate = pd.to_datetime(self.dataset['Original Release Date'])
        EndDate = pd.to_datetime(self.dataset['Current Version Release Date'])

        # Create a new column with the difference in days between the two dates
        self.dataset['Days Since Release'] = (EndDate - StartDate).dt.days

        # Drop the original date columns
        self.dataset.drop(columns=['Original Release Date',
                                   'Current Version Release Date'], inplace=True)

    
    def apply_feature_selection(self, k):
        # Separate features and labels
        self.features = self.dataset.drop(columns=['Rate'])
        self.label = self.dataset['Rate']

        # Apply feature selection using SelectKBest
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(self.features, self.label)

        # Get the selected feature names
        selected_feature_names = self.features.columns[selector.get_support()].tolist()

        # Update the features dataset with the selected features
        self.features = pd.DataFrame(data=X_selected, columns=selected_feature_names)

        return self.features, self.label

    def apply_pca(self):
        # Define the range of components to consider
        n_components_range = range(1, 9)

        # Create a PCA object
        pca = PCA()

        # Perform grid search to find the best number of components
        grid_search = GridSearchCV(pca, {'n_components': n_components_range})
        grid_search.fit(self.dataset.drop(columns=['Rate']))

        # Get the best number of components
        best_n_components = grid_search.best_params_['n_components']

        # Perform feature selection using PCA with the best number of components
        pca = PCA(n_components=best_n_components)
        features_selected = pca.fit_transform(self.dataset.drop(columns=['Rate']))

        # Create a new dataset with the selected features
        self.features = pd.DataFrame(data=features_selected)
    
    def split_data_then_scale(self, testSize):
        self.label = self.dataset['Rate']  # Label
        # Split the data into train, test, and validate sets
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.label, test_size=testSize, random_state=0)

      # Initialize the scaler
        scaler = MinMaxScaler()
        # Fit the scaler to the training data
        scaler.fit(X_train)
        # Scale the training, validation, and test data
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # Convert the scaled training data to a dataframe
        X_train = pd.DataFrame(data=X_train_scaled, columns=X_train.columns)
        # Convert the scaled validation data to a dataframe
        # Convert the scaled test data to a dataframe
        X_test = pd.DataFrame(data=X_test_scaled, columns=X_test.columns)

        return X_train, y_train, X_test, y_test

    def get_data(self):
        return self.features

    def preprocess_all(self):
        self.load_dataset()
        self.remove_duplicates()
        self.Feature_Encoder()
        self.remove_Unwantedcolumns(self.columns_to_dropped)
        self.preprocess_in_app_purchases()
        self.preprocess_languages()
        self.preprocess_Rate()
        self.preprocess_age_rating()
        self.Enconding()
        self.preprocess_dates()
        self.apply_feature_selection(5)

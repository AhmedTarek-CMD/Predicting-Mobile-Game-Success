from typing import List
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Scaling
from sklearn.preprocessing import MinMaxScaler


import warnings
warnings.filterwarnings("ignore")


class DataPreprocessing_Test:
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

    def preprocess_in_Price(self):
        price_nan = self.dataset["Price"].isnull().sum()
        if (price_nan != 0):
            self.dataset['Price'] = self.dataset['Price'].fillna(
                self.dataset['Price'].mean())

    def preprocess_in_app_purchases(self):
        app_nan = self.dataset["In-app Purchases"].isnull().sum()
        if (app_nan != 0):
            # Convert each cell into a list of floats, and compute the mean of the list
            self.dataset['In-app Purchases'] = self.dataset['In-app Purchases'].apply(lambda x: sum(
                [float(i) for i in x.split(',')])/len(x.split(',')) if pd.notnull(x) else x)
            # Replace the missing values with the mean of the In-app Purchases column
            mean_value = self.dataset['In-app Purchases'].mean()
            self.dataset['In-app Purchases'] = self.dataset['In-app Purchases'].fillna(
                mean_value)

    def preprocess_User_Rating_Count(self):
        user_nan = self.dataset["User Rating Count"].isnull().sum()
        if (user_nan != 0):
            self.dataset['User Rating Count'] = self.dataset['User Rating Count'].fillna(
                self.dataset['User Rating Count'].mean())

    def preprocess_Developer(self):
        dev_nan = self.dataset["Developer"].isnull().sum()
        if (dev_nan != 0):
            # Drop the rows with missing values in the "Developer" column
            self.dataset = self.dataset.dropna(subset=['Developer'])

    def preprocess_age_rating(self):
        age_nan = self.dataset["Age Rating"].isnull().sum()
        if (age_nan != 0):
            mode_value = self.dataset["Age Rating"].mode()[0]
            self.dataset['Age Rating'] = self.dataset['Age Rating'].fillna(
                value=mode_value)

        # Convert Age Rating to string and lowercase
        self.dataset['Age Rating'] = self.dataset['Age Rating'].astype(
            str).str.lower()
        # Rename column
        self.dataset.rename(columns={'Age Rating': 'min_age'}, inplace=True)
        self.dataset['min_age'] = self.dataset['min_age'].str.replace(
            '+', '').astype(int)

    def preprocess_languages(self):
        Lang_nan = self.dataset["Languages"].isnull().sum()
        if (Lang_nan != 0):
            # split the string values into lists of languages
            Lang_temp = self.dataset['Languages'].str.split(',')
            # use explode to convert the list column into separate rows
            exploded_Lang = Lang_temp.explode()  # from 5214 to 20429
            # fill missing values with mode value
            mode_value = exploded_Lang.mode()[0]
            self.dataset['Languages'] = self.dataset['Languages'].fillna(
                value=mode_value)

    def preprocess_size(self):
        size_nan = self.dataset["Size"].isnull().sum()
        if (size_nan != 0):
            self.dataset['Size'] = self.dataset['Size'].fillna(
                self.dataset['Size'].mean())

    def preprocess_Primary_Genre(self):
        pri_nan = self.dataset["Primary Genre"].isnull().sum()
        if (pri_nan != 0):
            mode_value = self.dataset["Primary Genre"].mode()[0]
            self.dataset['Primary Genre'] = self.dataset['Primary Genre'].fillna(
                value=mode_value)

    def preprocess_Genres(self):
        gen_nan = self.dataset["Genres"].isnull().sum()
        if gen_nan != 0:
            # split the string values into lists of genres
            gen_temp = self.dataset['Genres'].str.split(',')
            # use explode to convert the list column into separate rows
            exploded_gen = gen_temp.explode()
            # compute the top three modes of the genres column
            top_three = exploded_gen.value_counts().nlargest(3)
            # fill missing values with the top three modes
            mode_values = ', '.join(list(top_three.index))
            self.dataset['Genres'] = self.dataset['Genres'].fillna(
                value=mode_values)

    def preprocess_Original_Release_Date(self):
        ori_nan = self.dataset["Original Release Date"].isnull().sum()
        if ori_nan != 0:
            # convert the Date column to a datetime format
            self.dataset["Original Release Date"] = pd.to_datetime(
                self.dataset["Original Release Date"])
            # get the average year, month, and day
            year = self.dataset["Original Release Date"].dt.year
            mean_year = int(year.mean())

            month = self.dataset["Original Release Date"].dt.month
            mean_month = int(month.mean())

            day = self.dataset["Original Release Date"].dt.day
            mean_day = int(day.mean())

            # create a date object from the average year, month, and day
            mean_date = pd.Timestamp(
                day=mean_day, month=mean_month, year=mean_year).date()
            formatted_date = mean_date.strftime('%d/%m/%Y')
            self.dataset['Original Release Date'] = self.dataset['Original Release Date'].fillna(
                formatted_date)

    def preprocess_Current_Release_Date(self):
        Cur_nan = self.dataset["Current Version Release Date"].isnull().sum()
        if Cur_nan != 0:
            # convert the Date column to a datetime format
            self.dataset["Current Version Release Date"] = pd.to_datetime(
                self.dataset["Current Version Release Date"])
            # get the average year, month, and day
            year = self.dataset["Current Version Release Date"].dt.year
            mean_year = int(year.mean())

            month = self.dataset["Current Version Release Date"].dt.month
            mean_month = int(month.mean())

            day = self.dataset["Current Version Release Date"].dt.day
            mean_day = int(day.mean())

            # create a date object from the average year, month, and day
            mean_date = pd.Timestamp(
                day=mean_day, month=mean_month, year=mean_year).date()
            formatted_date = mean_date.strftime('%d/%m/%Y')
            self.dataset['Current Version Release Date'] = self.dataset['Current Version Release Date'].fillna(
                formatted_date)

    def preprocess_Rate(self):
        self.dataset["Rate"] = self.dataset["Rate"].replace({"Low": 1, "Intermediate": 2, "High": 3})
        
        # self.dataTest["Rate"] = self.dataTest["Rate"].map(lambda x: x.replace(" ", ""))
        # self.dataTest["Rate"] = self.dataTest["Rate"].map(
        #     lambda x: 2 if x == 'High' else 1 if x == 'Intermediate' else 0 if x == 'Low' else x)

        avg_nan = self.dataset["Rate"].isnull().sum()
        AUR_mode = self.dataset['Rate'].mode()
        if avg_nan != 0:
            self.dataset['Rate'] = self.dataset['Rate'].fillna(AUR_mode.iloc[0])
        

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
        self.dataset.drop(
            columns=['Original Release Date', 'Current Version Release Date'], inplace=True)

    def DataScaling(self):
        x_test = self.dataset.drop(columns = ['Rate'])  # Features   
        y_test =  self.dataset['Rate']  # Label
        # Initialize the scaler
        scaler = MinMaxScaler()
        # Fit the scaler to the training data
        scaler.fit(x_test)

        # Scale the training, validation, and test data
        x_test_scaled = scaler.transform(x_test)
        # Convert the scaled training data to a dataframe
        x_test = pd.DataFrame(
            data=x_test_scaled, columns=x_test.columns)

        return x_test , y_test
    
    def get_data(self):
        return self.dataset

    def preprocess_all(self):
        self.load_dataset()
        self.remove_duplicates()
        self.Feature_Encoder()
        self.remove_Unwantedcolumns(self.columns_to_dropped)
        self.preprocess_Rate()
        self.preprocess_in_Price()
        self.preprocess_in_app_purchases()
        self.preprocess_User_Rating_Count()
        self.preprocess_Developer()
        self.preprocess_age_rating()
        self.preprocess_languages()
        self.preprocess_size()
        self.preprocess_Primary_Genre()
        self.preprocess_Genres()
        self.preprocess_Original_Release_Date()
        self.preprocess_Current_Release_Date()
        self.preprocess_dates()
        self.Enconding()
        self.remove_Unwantedcolumns(self.extra)



    
data_preprocess2 = DataPreprocessing_Test('Data/Test-data.csv')
data_preprocess2.preprocess_all()
print(data_preprocess2.get_data())
x , y = data_preprocess2.DataScaling()
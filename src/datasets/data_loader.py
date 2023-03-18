from sklearn.model_selection import train_test_split
import pandas as pd


def load_iris():
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
    dataset = pd.read_csv(url, names=attributes)
    dataset.columns = attributes

    train, test = train_test_split(dataset, test_size=0.3, stratify=dataset['species'], random_state=45)
    X_train = train[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    Y_train = train.species
    X_test = test[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    Y_test = test.species

    return X_train, Y_train, X_test, Y_test


def load_penguins():
    url = 'https://gist.githubusercontent.com/slopp/ce3b90b9168f2f921784de84fa445651/raw/4ecf3041f0ed4913e7c230758733948bc561f434/penguins.csv'
    attributes = ['rowid', 'species', 'island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g',
                  'sex', 'year']
    dataset = pd.read_csv(url, names=attributes)[1:]
    dataset.columns = attributes
    dataset[['rowid', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']] = \
        dataset[['rowid', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']].apply(pd.to_numeric)

    train, test = train_test_split(dataset, test_size=0.3, stratify=dataset['species'], random_state=45)
    X_train = train[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']]
    Y_train = train.species
    X_test = test[['island', 'bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'sex', 'year']]
    Y_test = test.species

    return X_train, Y_train, X_test, Y_test


def load_BMI():
    url = 'https://raw.githubusercontent.com/abhiwalia15/500-Person-Gender-Height-Weight-Body-Mass-Index/master/500_Person_Gender_Height_Weight_Index.csv'
    attributes = ['Gender', 'Height', 'Weight', 'BMI']
    dataset = pd.read_csv(url, names=attributes)[1:]
    dataset.columns = attributes
    dataset[['Height', 'Weight', 'BMI']] = dataset[['Height', 'Weight', 'BMI']].apply(pd.to_numeric)

    train, test = train_test_split(dataset, test_size=0.3, stratify=dataset['BMI'], random_state=45)
    X_train = train[['Gender', 'Height', 'Weight']]
    Y_train = train.BMI
    X_test = test[['Gender', 'Height', 'Weight']]
    Y_test = test.BMI

    return X_train, Y_train, X_test, Y_test


def load_Obesity():
    url = 'https://raw.githubusercontent.com/abhiwalia15/500-Person-Gender-Height-Weight-Body-Mass-Index/master/500_Person_Gender_Height_Weight_Index.csv'
    attributes = ['Gender', 'Height', 'Weight', 'BMI']
    dataset = pd.read_csv(url, names=attributes)[1:]
    dataset.columns = attributes
    dataset[['Height', 'Weight', 'BMI']] = dataset[['Height', 'Weight', 'BMI']].apply(pd.to_numeric)

    dataset['Obese'] = (dataset.BMI >= 4).astype('object')
    dataset.drop('BMI', axis=1, inplace=True)

    train, test = train_test_split(dataset, test_size=0.3, stratify=dataset['Obese'], random_state=60)
    X_train = train[['Gender', 'Height', 'Weight']]
    Y_train = train.Obese
    X_test = test[['Gender', 'Height', 'Weight']]
    Y_test = test.Obese

    return X_train, Y_train, X_test, Y_test

import pandas as pd
from sklearn.model_selection import train_test_split

file_path = "datasets/abalone.data"
columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight',
           'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
df = pd.read_csv(file_path, header=None, names=columns)

X = df.drop(columns=['Rings'])
y = df['Rings']

X = pd.get_dummies(X, columns=['Sex'], drop_first=True)

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, y, test_size=0.2, random_state=42)





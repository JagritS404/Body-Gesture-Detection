import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline 
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score # Accuracy metrics 
import pickle 
df=pd.read_csv(r'C:\Users\DELL\Jagrit Codes VS\book3.csv')
print(df.tail())
x=df.drop('class',axis=1)
y=df['class']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.3, random_state=1234)
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, Y_train)
    fit_models[algo] = model
for algo, model in fit_models.items():
    pred = model.predict(X_test)
    print(algo, accuracy_score(Y_test, pred))
with open('final_model.pkl', 'wb') as file_model:
    pickle.dump(fit_models['rf'], file_model)

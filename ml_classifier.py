import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def extract_features(strings, alphabet):
    alphabet=list(alphabet); bigrams=[a+b for a in alphabet for b in alphabet]; rows=[]
    for s in strings:
        n=len(s); counts=[s.count(c)/max(n,1) for c in alphabet]
        bg=[sum(s[i:i+2]==b for i in range(max(0,n-1)))/max(n-1,1) for b in bigrams]
        first=[int(bool(s) and s[0]==c) for c in alphabet]; last=[int(bool(s) and s[-1]==c) for c in alphabet]
        starts=[int(bool(s) and s.startswith(c)) for c in alphabet]; ends=[int(bool(s) and s.endswith(c)) for c in alphabet]
        changes=sum(s[i]!=s[i-1] for i in range(1,n))
        rows.append([n]+counts+bg+first+last+starts+ends+[changes])
    return np.asarray(rows,dtype=float).reshape(len(strings), 1+len(alphabet)+len(bigrams)+4*len(alphabet)+1)

def train_and_evaluate(X_train,y_train,X_test,y_test):
    models={'Decision Tree':DecisionTreeClassifier(max_depth=10,random_state=42),'Logistic Regression':LogisticRegression(max_iter=1000,random_state=42),'Random Forest':RandomForestClassifier(n_estimators=100,random_state=42)}
    result={}
    for name,model in models.items():
        model.fit(X_train,y_train); pred=model.predict(X_test)
        result[name]={'accuracy':accuracy_score(y_test,pred),'precision':precision_score(y_test,pred,zero_division=0),'recall':recall_score(y_test,pred,zero_division=0),'f1_score':f1_score(y_test,pred,zero_division=0),'confusion_matrix':confusion_matrix(y_test,pred,labels=[0,1]),'model':model}
    return result

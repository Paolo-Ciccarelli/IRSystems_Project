#pip install -U scikit-learn
from sklearn.linear_model import LogisticRegression

def is_linearly_seperable(X,y):
    model = LogisticRegression(penalty=None, max_iter=10000)
    model.fit(X,y)
    accuracy = model.score(X,y)
    return accuracy == 1.0

X_and = [[0,0],[0,1],[1,0],[1,1]]
y_and = [0,0,0,1]
print("Testing whether AND is linearly seperable:", is_linearly_seperable(X_and,y_and))

X_xor = [[0,0],[0,1],[1,0],[1,1]]
y_xor = [0,1,1,0]
print("Testing whether XOR is linearly seperable:", is_linearly_seperable(X_xor,y_xor))


import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import xgboost as xgb
from sklearn.ensemble import (RandomForestClassifier,GradientBoostingClassifier)

from model_esm import save_fasta
from model_esm import get_gt_representation
from model_esm import generate_ecfp
# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("./ML/XGBoost.pkl", "rb"))
X_train = np.load("./model-pretrain/X_train.npy")
y_train = np.load("./model-pretrain/y_train.npy")
model.fit(X_train, y_train.ravel())

@flask_app.route("/")
def Home():
    return render_template("index-ml.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    X=();
    features = [str(x) for x in request.form.values()]
    # 生成fasta文件
    save_fasta(features[0])
    generate_feature_P=get_gt_representation("./test.fasta")

    generate_feature_S=generate_ecfp(features[1])
    generate_feature_S=np.array(list(generate_feature_S)).astype(int)

    combined = X+(np.concatenate([generate_feature_S,generate_feature_P[0]]),)
    prediction = model.predict(combined[:1])
    print(prediction)
    return render_template("index-ml.html", prediction_result = "The NP-GT relationship is  {}".format(prediction))


if __name__ == "__main__":
    flask_app.run(debug=True)
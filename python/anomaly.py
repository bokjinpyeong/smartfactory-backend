import os, sys, json
import pandas as pd
import joblib, boto3, pymysql

S3_BUCKET = os.environ.get('S3_BUCKET_NAME', 'sensor-tive-lambda-layers')
MODEL_KEY = "models/iforest_model.pkl"
MODEL_PATH = f"/tmp/{MODEL_KEY.split('/')[-1]}"

DB_HOST = os.environ['DB_HOST']
DB_USER = os.environ['DB_USER']
DB_PW   = os.environ['DB_PW']
DB_NAME = os.environ['DB_NAME']

s3 = boto3.client('s3')
_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            s3.download_file(S3_BUCKET, MODEL_KEY, MODEL_PATH)
        _model = joblib.load(MODEL_PATH)
    return _model

def main():
    body = json.loads(sys.stdin.read() or "{}")
    record_id = body.get("record_id")
    features  = body.get("features", {})
    feature_order = ['voltage', 'current', 'temperature']
    X = pd.DataFrame({k:[features.get(k)] for k in feature_order})

    model = load_model()
    pred = model.predict(X)
    is_anomaly = 1 if pred[0] == -1 else 0

    if is_anomaly and record_id is not None:
        conn = pymysql.connect(host=DB_HOST, user=DB_USER, password=DB_PW, database=DB_NAME, autocommit=True)
        with conn.cursor() as cur:
            cur.execute("UPDATE power_data SET anomaly_flag = 1 WHERE id = %s", (record_id,))
        conn.close()

    print(json.dumps({"record_id": record_id, "is_anomaly": is_anomaly}))

if __name__ == "__main__":
    main()

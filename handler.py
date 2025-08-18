import json, numpy as np
def lambda_handler(event, context):
    detail = event.get("detail", {}) if isinstance(event, dict) else {}
    return {"statusCode":200,"headers":{"Content-Type":"application/json"},
            "body":json.dumps({"ok":True,"numpy":np.__version__,"received":detail})}

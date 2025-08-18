FROM public.ecr.aws/lambda/python:3.11
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
COPY handler.py ${LAMBDA_TASK_ROOT}/
CMD ["handler.lambda_handler"]

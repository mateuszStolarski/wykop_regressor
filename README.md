# wykop_regressor

#fastapi
python -m uvicorn main:app

#celery tasks
python -m celery -A tasks worker --loglevel=INFO
#windows
python -m celery -A tasks worker --pool=solo --loglevel=INFO

#celery beat
python -m celery -A tasks beat

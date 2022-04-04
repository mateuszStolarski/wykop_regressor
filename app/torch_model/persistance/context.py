import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

CONNECTION_STRING = 'mysql+pymysql://root:alchemy@127.0.0.1:3306/imdb'
engine = sqlalchemy.create_engine(CONNECTION_STRING)
_SessionFactory = sessionmaker(bind=engine)

Base = declarative_base()


def session_factory() -> Session:
    Base.metadata.create_all(engine)
    return _SessionFactory()


def get_engine():
    return engine

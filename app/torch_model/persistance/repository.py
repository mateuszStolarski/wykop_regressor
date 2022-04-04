from .context import session_factory
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from .models.basics import Basic
from .models.ratings import Rating
from sqlalchemy.orm import Load
import pandas as pd
from sqlalchemy.orm import Session


def get_all(table, session: Session):
    query = session.query(table)
    return query.all()


def is_empty(table, session: Session):
    query_result = session.query(table).limit(1).all()
    if len(query_result) == 0:
        result = False
    else:
        result = True

    return result


def insert_one(table, session: Session):
    try:
        session.add(table)
        session.commit()
    except IntegrityError as e:
        raise e.orig
    except SQLAlchemyError as e:
        raise e


def delete_one(table, session: Session):
    try:
        session.delete(table)
        session.commit()
    except IntegrityError as e:
        raise e.orig
    except SQLAlchemyError as e:
        raise e


def get_data_with_start_year(startYear: int, session: Session) -> pd.DataFrame:
    query = session.query(Basic, Rating).\
        join(Rating, Rating.id == Basic.rating_id).\
        options(Load(Basic).load_only(Basic.titleType,
                                      Basic.originalTitle,
                                      Basic.isAdult,
                                      Basic.startYear,
                                      Basic.runtimeMinutes,
                                      Basic.genres),
                Load(Rating).load_only(Rating.averageRating,
                                       Rating.numVotes)
                ).\
        filter(Basic.startYear >= startYear)
    data = pd.read_sql(query.statement, session.bind)
    return data.drop(columns=['id', 'id_1'])

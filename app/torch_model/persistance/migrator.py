from .repository import get_all, is_empty
from .models.names import Name
from .models.akas import Aka
from .models.basics import Basic
from .models.principals import Principal
from .models.ratings import Rating
import pandas as pd
from .context import session_factory
from sqlalchemy.orm import Session

BASE_PATH = "..\\data"


def create_name(row):
    return Name(row.nconst, row.primaryName, row.birthYear, row.deathYear,
                row.knownForTitles)


def create_aka(row):
    return Aka(row.titleId, row.ordering, row.title, row.region,
               row.language, row.types, row.attributes, row.isOriginalTitle)


def create_basic(row, rate):
    return Basic(row.tconst, row.titleType, row.primaryTitle, row.originalTitle,
                 row.isAdult, row.startYear, row.endYear, row.runtimeMinutes,
                 row.genres, rate)


def create_principal(row):
    return Principal(row.tconst, row.ordering, row.nconst, row.category,
                     row.job, row.characters)


def create_rating(tconst, averageRating, numVotes):
    return Rating(tconst, averageRating, numVotes)


def import_table(create, sub_path):
    session = session_factory()
    df = pd.read_csv(f'{BASE_PATH}\\{sub_path}', sep='\t')

    counter = 0
    for _, row in df.iterrows():
        name = create(row)
        session.add(name)
        counter += 1
        if counter >= 10000:
            break

    session.commit()
    session.close()


def import_related_table():
    session = session_factory()
    df_rating = pd.read_csv(f'{BASE_PATH}\\title.ratings.tsv\\title.ratings.tsv', sep='\t')
    df = pd.read_csv(f'{BASE_PATH}\\title.basics.tsv\\title.basics.tsv', sep='\t')

    counter = 0
    for _, row in df.iterrows():
        rate_row = df_rating[df_rating['tconst'] == row.tconst]
        for _, rr in rate_row.iterrows():
            rate = create_rating(rr[0], rr[1], rr[2])
        basic = create_basic(row, rate)
        session.add(rate)
        session.add(basic)
        counter += 1
        if counter % 1000 == 0:
            print(counter)
        if counter >= 10000:
            break

    session.commit()
    session.close()


def init_script(session: Session):
    tables = [(Name, create_name, 'name.basics.tsv\\name.basics.tsv'),
              (Aka, create_aka, 'title.akas.tsv\\title.akas.tsv'),
              (Principal, create_principal, 'title.principals.tsv\\title.principals.tsv'),
              #   (Rating, create_rating, 'title.ratings.tsv\\title.ratings.tsv'),
              #   (Basic, create_basic, 'title.basics.tsv\\title.basics.tsv'),
              ]

    if is_empty(Rating, session) and is_empty(Basic, session) == False:
        import_related_table()
    for table in tables:
        empty = is_empty(table[0], session)
        if empty == False:
            import_table(table[1], table[2])

    # result = get_all(Basic)

    # print(f'{result[0].originalTitle}: {result[0].rate.averageRating}')

import sqlalchemy


# engine = sqlalchemy.create_engine('mysql+pymysql://root:alchemy@127.0.0.1:3306/imdb')
# engine.execute("DROP DATABASE imdb")


engine = sqlalchemy.create_engine('mysql+pymysql://root:alchemy@127.0.0.1:3306')
engine.execute("CREATE DATABASE imdb")

from sqlalchemy import Column, Integer, String
from ..context import Base


class Name(Base):
    __tablename__ = "Names"
    id = Column(Integer, primary_key=True)
    nconst = Column(String(16), nullable=False)
    primaryName = Column(String(128), nullable=False)
    birthYear = Column(String(128), nullable=False)
    deathYear = Column(String(128), nullable=False)
    knownForTitles = Column(String(128), nullable=False)

    def __init__(self, nconst, primaryName, birthYear, deathYear, knownForTitles):
        self.nconst = nconst
        self.primaryName = primaryName
        self.birthYear = birthYear
        self.deathYear = deathYear
        self.knownForTitles = knownForTitles

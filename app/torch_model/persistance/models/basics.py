from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from ..context import Base


class Basic(Base):
    __tablename__ = "Basics"
    id = Column(Integer, primary_key=True)
    tconst = Column(String(128), nullable=False)
    titleType = Column(String(128), nullable=False)
    primaryTitle = Column(String(512), nullable=False)
    originalTitle = Column(String(512), nullable=False)
    isAdult = Column(Boolean, nullable=False)
    startYear = Column(String(128), nullable=False)
    endYear = Column(String(128), nullable=False)
    runtimeMinutes = Column(String(128), nullable=False)
    genres = Column(String(128), nullable=False)
    rating_id = Column(Integer, ForeignKey('Ratings.id'))

    def __init__(self, tconst, titleType, primaryTitle, originalTitle, isAdult, startYear, endYear, runtimeMinutes, genres, rate):
        self.tconst = tconst
        self.titleType = titleType
        self.primaryTitle = primaryTitle
        self.originalTitle = originalTitle
        self.isAdult = isAdult
        self.startYear = startYear
        self.endYear = endYear
        self.runtimeMinutes = runtimeMinutes
        self.genres = genres
        self.rate = rate

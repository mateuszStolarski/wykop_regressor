from sqlalchemy import Column, Integer, String, Numeric
from ..context import Base
from sqlalchemy.orm import relationship


class Rating(Base):
    __tablename__ = "Ratings"
    id = Column(Integer, primary_key=True)
    tconst = Column(String(128), nullable=False)
    averageRating = Column(String(128), nullable=False)
    numVotes = Column(String(128), nullable=False)
    basic = relationship("Basic", uselist=False, backref="rate")

    def __init__(self, tconst, averageRating, numVotes):
        self.tconst = tconst
        self.averageRating = averageRating
        self.numVotes = numVotes

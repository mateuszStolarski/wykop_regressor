from sqlalchemy import Column, Integer, String
from ..context import Base


class Principal(Base):
    __tablename__ = "Principals"
    id = Column(Integer, primary_key=True)
    tconst = Column(String(128), nullable=False)
    ordering = Column(Integer, nullable=False)
    nconst = Column(String(128), nullable=False)
    category = Column(String(128), nullable=False)
    job = Column(String(128), nullable=False)
    characters = Column(String(128), nullable=False)

    def __init__(self, tconst, ordering, nconst, category, job, characters):
        self.tconst = tconst
        self.ordering = ordering
        self.nconst = nconst
        self.category = category
        self.job = job
        self.characters = characters

from sqlalchemy import Column, Integer, String, Boolean
from ..context import Base


class Aka(Base):
    __tablename__ = "Akas"
    id = Column(Integer, primary_key=True)
    titleId = Column(String(128), nullable=False)
    ordering = Column(String(128), nullable=False)
    title = Column(String(512), nullable=False)
    region = Column(String(128), nullable=False)
    language = Column(String(128), nullable=False)
    types = Column(String(128), nullable=False)
    attributes = Column(String(128), nullable=False)
    isOriginalTitle = Column(Boolean, nullable=False)

    def __init__(self, titleId, ordering, title, region, language, types, attributes, isOriginalTitle):
        self.titleId = titleId
        self.ordering = ordering
        self.title = title
        self.region = region
        self.language = language
        self.types = types
        self.attributes = attributes
        self.isOriginalTitle = isOriginalTitle

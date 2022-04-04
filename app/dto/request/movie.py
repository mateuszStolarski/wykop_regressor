from pydantic import BaseModel


class Movie(BaseModel):
    titleType: str
    originalTitle: str
    isAdult: int
    startYear: int
    runtimeMinutes: int
    genres: str
    numVotes: int

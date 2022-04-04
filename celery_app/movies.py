import json


class Movies():
    def __init__(self, author: str = None, creation_date: int = None, content: str = None, number_of_pluses: int = None, number_of_comments: int = None):
        self.author = author
        self.creation_date = creation_date
        self.content = content
        self.number_of_pluses = number_of_pluses
        self.number_of_comments = number_of_comments

    @staticmethod
    def get_columns():
        return ["author", "creation_date", "content",
                "number_of_pluses", "number_of_comments"]

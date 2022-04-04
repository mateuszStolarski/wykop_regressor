from helpers import alreadyExists
import pandas as pd
from datetime import datetime
import calendar


def check_18(li) -> bool:
    div = li.find("div", {"class": "hide18unlogged"})
    if div == None:
        return True
    else:
        return False


def get_number_of_comments(li) -> int:
    div = li.find("div", {"class": "row elements"})
    if div != None:
        comments_a = div.find("a")
        comments = comments_a.get_text().split()

        if comments[0] == 'skomentuj':
            comments[0] = 0

        return comments[0]
    return 0


def get_author(li) -> str:
    div = li.find("div", {"class": "fix-tagline"})
    if div != None:
        author_a = div.find("a")
        author = author_a.get_text().split()

        return author[0]
    return ''


def get_creation_date(li) -> str:
    div = li.find("div", {"class": "row elements"})
    date_time = div.find("time")

    return date_time.__getitem__("datetime")


def get_plus_count(li) -> int:
    div = li.find("a")
    count_span = div.find("span")
    count = count_span.get_text()

    return count


def get_content(li) -> str:
    div = li.find("div", {"class": "description"})
    content_div = div.find("a")
    content = content_div.get_text()
    content = content.split()

    return ' '.join(content)


def get_creation_date(li) -> int:
    div = li.find("div", {"class": "row elements"})
    date_time = div.find("time")
    date = date_time.__getitem__("datetime")

    datetime_object = datetime.fromisoformat(date)
    utc_time = calendar.timegm(datetime_object.utctimetuple())

    return utc_time


def append_to_df(li, columns) -> pd.DataFrame:
    comments = get_number_of_comments(li)
    author = get_author(li)
    utc_time = get_creation_date(li)
    plus = get_plus_count(li)
    content = get_content(li)
    temp = pd.DataFrame(
        [[author, utc_time, content, plus, comments]], columns=columns)

    return temp


def check_and_append(df, li, columns) -> pd.DataFrame:
    if check_18(li) == True:
        temp = append_to_df(li, columns)
        if alreadyExists(temp.creation_date[0]) == False:
            df = pd.concat([df, temp], ignore_index=True)

    return df

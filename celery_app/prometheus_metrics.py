from pandas.core.frame import DataFrame
from prometheus_client import Gauge, push_to_gateway, CollectorRegistry, Counter
import consts
import collections
from langdetect import detect

registry = CollectorRegistry()

hp = Gauge('sum_of_plusses',
           'Summarize plusses by execution', registry=registry)
c = Gauge('data_count', 'Number of scrapped posts on wykop',
          registry=registry)
hc = Gauge('sum_of_comments',
           'Summarize comments by execution', registry=registry)
lang = Counter('language_info',
               'Statistics about data languages', registry=registry, labelnames=['language'])
proccessing_time = Gauge('task_processing_seconds',
                         'Time spent processing task', registry=registry)


def data_count(data: int):
    print(data)
    c.set(data)


def plus_histogram(pluses):
    sum = 0
    for plus in pluses:
        sum += int(plus)
    hp.set(sum)


def comments_histogram(comments):
    sum = 0
    for comment in comments:
        sum += int(comment)
    hc.set(sum)


def analyze_language(df: DataFrame):
    results = []
    for _, row in df.iterrows():
        content = row.content
        language = detect(content)
        results.append(language)

    counter = collections.Counter(results)
    data = dict(counter)

    for key in data:
        lang.labels(language=str(key)).inc(int(data[key]))


def set_metrics(results: DataFrame):
    data_count(results.shape[0])
    plus_histogram(results.number_of_pluses)
    comments_histogram(results.number_of_comments)
    analyze_language(df=results)

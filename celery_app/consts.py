# celery
broker_url = "amqp://user:user@rabbitmq"
redis_url = "redis://redis-master"
wykop_url = 'https://www.wykop.pl/tag/znaleziska/rozrywka/najlepsze/'

# crawler
list_id = 'itemsStream'

# mongo
mongo_connection_string = 'mongodb://root:root@mongo-mongodb/?authSource=admin'
database_name = 'Crawlers'
collection_name = 'Movies'

# prometheus
gateway = 'prometheus-pushgateway:9091'

# outside cluster
# broker_url = "amqp://user:user@localhost:5672"
# redis_url = "redis://localhost:6379/0"
# mongo_connection_string = 'mongodb://root:root@localhost/?authSource=admin'
#gateway = 'localhost:9091'

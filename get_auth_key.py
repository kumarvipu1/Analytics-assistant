import os
from google.cloud import bigquery
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/Mishael.Ralph-Gbobo/PycharmProjects/Analytics-assistant/service_account.json'

client = bigquery.Client()
print(client)

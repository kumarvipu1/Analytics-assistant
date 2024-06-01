
import logging
from google.cloud import bigquery
from google.oauth2 import service_account

def check_bigquery_access():
    try:
        key_path = '/Users/Mishael.Ralph-Gbobo/PycharmProjects/Analytics-assistant/service_account.json'
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = bigquery.Client(credentials=credentials, project='analyticsassistantproject')

        dataset_id = 'user_forms'
        dataset_ref = bigquery.DatasetReference(client.project, dataset_id)
        tables = list(client.list_tables(dataset_ref))

        if tables:
            logging.info("Access to dataset confirmed. List of tables:")
            for table in tables:
                logging.info(table.table_id)
        else:
            logging.info("Dataset exists but no tables found.")

    except Exception as e:
        logging.error(f"Exception occurred: {e}")

check_bigquery_access()
import logging
from google.cloud import bigquery
from google.oauth2 import service_account

def check_bigquery_access():
    try:
        key_path = '/Users/Mishael.Ralph-Gbobo/PycharmProjects/Analytics-assistant/service_account.json'
        credentials = service_account.Credentials.from_service_account_file(key_path)
        client = bigquery.Client(credentials=credentials, project='analyticsassistantproject')

        dataset_id = 'analyticsassistantproject.user_forms'
        dataset = client.get_dataset(dataset_id)  # Make an API request.

        full_dataset_id = "{}.{}".format(dataset.project, dataset.dataset_id)
        friendly_name = dataset.friendly_name
        print(
            "Got dataset '{}' with friendly_name '{}'.".format(
                full_dataset_id, friendly_name
            )
        )

        # View dataset properties.
        print("Description: {}".format(dataset.description))
        print("Labels:")
        labels = dataset.labels
        if labels:
            for label, value in labels.items():
                print("\t{}: {}".format(label, value))
        else:
            print("\tDataset has no labels defined.")

    except Exception as e:
        logging.error(f"Exception occurred: {e}")

check_bigquery_access()
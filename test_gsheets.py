import gspread
from oauth2client.service_account import ServiceAccountCredentials

def connect_to_gsheets(json_keyfile_name, sheet_name):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name(json_keyfile_name, scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name).sheet1
    return sheet

def save_to_gsheets(json_keyfile_name, sheet_name, firstname, lastname, email, occupation, company, phone):
    sheet = connect_to_gsheets(json_keyfile_name, sheet_name)
    row = [firstname, lastname, email, occupation, company, phone]
    sheet.append_row(row)

def test_save_to_gsheets():
    try:
        # Update with the path to your JSON keyfile and your Google Sheet name
        json_keyfile = '/Users/Mishael.Ralph-Gbobo/PycharmProjects/Analytics-assistant/service_account.json'
        sheet_name = 'AI_Analytics_App_User_Information'

        # Sample data to save
        firstname = 'John'
        lastname = 'Doe'
        email = 'john.doe@example.com'
        occupation = 'Software Engineer'
        company = 'Tech Inc.'
        phone = '123-456-7890'

        # Save the sample data to the Google Sheet
        save_to_gsheets(json_keyfile, sheet_name, firstname, lastname, email, occupation, company, phone)
        print("Data successfully saved to Google Sheets.")

        # Optional: Verify by reading the last row
        sheet = connect_to_gsheets(json_keyfile, sheet_name)
        last_row = sheet.get_all_values()[-1]
        print("Last row in the sheet:", last_row)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    test_save_to_gsheets()
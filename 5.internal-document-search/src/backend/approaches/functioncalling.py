import pytz
from datetime import datetime

def get_current_time(location):
    try:
        # Get the timezone for the city
        timezone = pytz.timezone(location)

        # Get the current time in the timezone
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")

        return current_time
    except:
        return "Sorry, I couldn't find the timezone for that location."
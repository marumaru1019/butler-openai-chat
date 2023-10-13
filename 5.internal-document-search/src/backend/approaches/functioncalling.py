import pytz
from datetime import datetime

def get_current_time(location):
    try:
        timezone = pytz.timezone(location)
        now = datetime.now(timezone)
        current_time = now.strftime("%I:%M:%S %p")
        return current_time
    except:
        return "現在時刻の取得に失敗しました"
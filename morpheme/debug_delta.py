import datetime
import time

now_dt = datetime.datetime.now()
print(f"Now: {now_dt}")
midnight = (now_dt + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
print(f"Midnight: {midnight}")
delta = (midnight - now_dt).total_seconds()
print(f"Delta: {delta}")

custom_end_time = time.time() + delta
print(f"Custom End Time: {custom_end_time}")
print(f"Time Remaining: {custom_end_time - time.time()}")

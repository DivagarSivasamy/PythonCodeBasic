from datetime import datetime, timedelta

print_interval = timedelta(seconds=2)

# Initialize the next print time. Using now() will print the first
# iteration and then every interval. To avoid printing the first
# time, just add print_interval here (i.e. uncomment).
next_print = datetime.now() # + print_interval

for i in range(int(1e8)):
    now = datetime.now()
    if now >= next_print:
        next_print = now + print_interval
        print(i)
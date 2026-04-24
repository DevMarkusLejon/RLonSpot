import re
import os
def get_next_log_file_number(log_dir: str, prefix: str = "logfile_", suffix: str = ".txt") -> str:
    os.makedirs(log_dir, exist_ok=True)
    pattern = re.compile(rf"{re.escape(prefix)}(\d+){re.escape(suffix)}")
    
    existing_numbers = []
    for filename in os.listdir(log_dir):
        match = pattern.match(filename)
        if match:
            existing_numbers.append(int(match.group(1)))

    return max(existing_numbers, default=0) + 1
    
logdir = "/home/sundt/thesis/colcon_ws/src/my_spot_thesis/spot_deploy_data/logs/"
logprefix = "logfile_"
logsuffix = ".txt"

number = get_next_log_file_number(log_dir=logdir, prefix=logprefix, suffix=logsuffix)
filename = logdir + logprefix + str(number) + logsuffix
file = open(filename, "w")
print(filename)
print(number)

file.close()
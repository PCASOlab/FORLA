# working_para/machine_configs.py
import os


# Define base paths for each machine
MACHINE_PATHS = {
    1: {
        "base": "/media/guiqiu/data/",
        # Add other machine 1 paths here
    },
    2: {
        "base": "./data/",
        # Add other machine 1 paths here
    },
    3: {
        "base": "./data/",
        # Add other machine 3 paths here
    },
    4: {
        "base": "./data/",
        # Add other machine 3 paths here
    }
}

def get_machine_path(key):
    MACHINE_ID = int(os.environ.get("MACHINE_ID", "1"))  # Default to Machine 1

    return MACHINE_PATHS[MACHINE_ID][key]
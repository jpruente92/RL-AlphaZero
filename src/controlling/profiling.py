import os

from pyinstrument import Profiler

from constants.logging_constants import PROFILE_FILE_PATH


def profile(func):
    def wrapper(*args, **kwargs):
        print("Start Profiler")
        profiler = Profiler()
        profiler.start()
        result = func(*args, **kwargs)
        profiler.stop()
        os.makedirs(os.path.dirname(PROFILE_FILE_PATH), exist_ok=True)
        with open(PROFILE_FILE_PATH, "w+") as f:
            f.write(profiler.output_text())
        print("Profile stored")
        return result

    return wrapper

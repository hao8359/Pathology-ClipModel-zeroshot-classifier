import os
from line_profiler import profile

# Checks if profiling is enabled
if os.getenv("LINE_PROFILE")=="1":
    from line_profiler import profile
else:
    def profile(func): # define a dummy profile function
        return func
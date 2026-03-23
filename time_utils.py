import time

def format_runtime(start_time, end_time):
    """Formats the runtime into hours, minutes, and seconds based on the elapsed time."""
    elapsed_time = int(end_time - start_time)  # Convert to integer to remove decimals

    # Calculate hours, minutes, and seconds
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    seconds = elapsed_time % 60

    # Format the output based on the elapsed time
    if hours > 0:
        runtime = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        runtime = f"{minutes}m {seconds}s"
    else:
        runtime = f"{seconds}s"

    return runtime

# You can also include a helper to capture the start time easily
def current_time():
    """Returns the current time in seconds."""
    return time.time()
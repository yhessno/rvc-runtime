import subprocess

def run_command(command):
    try:
        # Run the command and capture its output
        output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True)
        return output
    except subprocess.CalledProcessError as e:
        return f"Command failed with return code {e.returncode}:\n{e.output}"

if __name__ == "__main__":
    import os

    current_directory = os.getcwd()
    # Get the command from the user
    user_command = "snakeviz profiling/temp.dat"

    # Call the function to run the command and get the output
    result = run_command(user_command)

    # Print the result
    print("Command Output:")
    print(result)
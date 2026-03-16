import subprocess
import datetime
import threading

# make the timestamp utc-8
timestamp = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime("%Y%m%d-%H%M%S")
BEHAVIOR_ID = 1
DEVICE = 0
OUTPUT_PATH = f"output_base/BEHAVIOR_ID_{BEHAVIOR_ID}/{timestamp}"

def stream_reader(pipe, label):
    for line in pipe:
        #print(f"{label}:", line, end='')
        print(f"", line, end='')

def run_single_process(behavior_id: int, device: int, output_path: str,defense:str,behaviors_config:str):
    command = ["python", "attack_llm_core_base_ppl.py", "--id", str(behavior_id), "--device", str(device), "--output_path", output_path,"--defense",defense,"--behaviors_config",behaviors_config]
    print(" ".join(command))
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Create threads to read from stdout and stderr
    stdout_thread = threading.Thread(target=stream_reader, args=(process.stdout, "STDOUT"))
    stderr_thread = threading.Thread(target=stream_reader, args=(process.stderr, "STDERR"))
    
    # Start the threads
    stdout_thread.start()
    stderr_thread.start()

    # Wait for threads to finish
    stdout_thread.join()
    stderr_thread.join()

    # Ensure the process completes
    process.communicate()
    
if __name__ == "__main__":
    #  --device 0 --output_path Our_GCG_target_len_20/ours/20260226-002518 --defense no_defense --behaviors_config behaviors_config.json
    run_single_process(BEHAVIOR_ID, DEVICE, OUTPUT_PATH,"no_defense","behaviors_config.json")
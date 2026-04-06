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
    # parser.add_argument('--batch_size', type=int, default=6 )  # 新增：从命令行传参
    #     parser.add_argument('--top_k', type=int, default=256)  # 新增：从命令行传参
    #     parser.add_argument('--num_steps', type=int, default=1000)  # 新增：从命令行传参
    command = ["python", "attack_llm_core_base_select_base_embeding.py", "--id", str(behavior_id), "--device", str(device),
               "--output_path", output_path, "--defense", defense, "--behaviors_config", behaviors_config, "--batch_size", "512", "--top_k","256", "--num_steps","1000"]
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


def run_single_process_select_method(behavior_id: int, device: int,
                                     output_path: str,  behaviors_config: str, defense="",
                                     model_path = r"D:\Model\Llama-2-7b-chat-hf",
                                     num_steps=1000,
                                     batch_size=512,
                                     loss_type="cross_entropy",
                                     str_init="adv_init_suffix",
                                     use_ppl_filter = "False",
                                     use_multi_target = "False",
                                     use_contrast_loss = "False"):
    command = ["python", "-u" ,"attack_llm_core_base_select_mothed.py",
               "--id", str(behavior_id),
               "--device", str(device),
               "--output_path", output_path,
               "--model_path", model_path,
               # "--defense", defense,
               "--behaviors_config", behaviors_config,
               "--batch_size", str(batch_size),
               "--top_k", "256",
               "--num_steps", str(num_steps),
               "--loss_type", loss_type,
               "--str_init", str_init,
               "--use_ppl_filter", use_ppl_filter,
               "--use_multi_target", use_multi_target,
               "--use_contrast_loss", use_contrast_loss]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')

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
    #run_single_process(BEHAVIOR_ID, DEVICE, OUTPUT_PATH,"no_defense","behaviors_config.json")
    run_single_process_select_method(BEHAVIOR_ID, DEVICE, OUTPUT_PATH,
                                     behaviors_config="behaviors_ours_config_modify.json",
                                     str_init="adv_init_suffix2",
                                     use_ppl_filter="true")
import yaml
import argparse
def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def get_args():
    custom_parameters = [
        {"name": "--name", "type": str, "action": "store_true", "default": 'test', "help": "Name of the experiment to run or load. Overrides config file if provided."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--render", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--fix_cam", "action": "store_true", "default": True, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": 'Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)'},
        {"name": "--num_envs", "type": int, "help": "Number of environments to create. Overrides config file if provided."},
        {"name": "--seed", "type": int, "help": "Random seed. Overrides config file if provided."},
        {"name": "--max_iterations", "type": int, "help": "Maximum number of training iterations. Overrides config file if provided."},
        {"name": "--video", "action": "store_true", "default": False, "help": "record your display"},
        {"name": "--time", "type": int, "default": 8, "help": "display time(seconds)."},
        {"name": "--iter", "type": int, "default": None, "help": "display epoch times."},
        {"name": "--epochs", "type": int, "default": 1, "help": "display epoch times."},
        {"name": "--debug", "action": "store_true", "default": False, "help": "save data to excel"}
    ]
    # # parse arguments
    # args = gymutil.parse_arguments(
    #     description="RL Policy",
    #     custom_parameters=custom_parameters)
    #
    # # name allignment
    # args.sim_device_id = args.compute_device_id
    # args.sim_device = args.sim_device_type
    # if args.sim_device == 'cuda':
    #     args.sim_device += f":{args.sim_device_id}"
    parser = argparse.ArgumentParser(description='Process some filenames.')
    parser.add_argument('--name', type=str, default='test',required=True, help='The name of the file')
    parser.add_argument('--debug',default=True,required=False, help='save data to excel')
    parser.add_argument('--video',default=True,required=False, help='save video')
    args = parser.parse_args()
    return args
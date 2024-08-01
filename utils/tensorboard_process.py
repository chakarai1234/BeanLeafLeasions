from logging import Logger
import subprocess
from utils import log
from typing_extensions import Union

def start_tensorboard(log_dir:str, logger:Logger, project_name: Union[str, None], justify:int):
    try:
        if project_name != None:
            tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--bind_all", "--window_title", project_name])
        else:
            tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--bind_all"])

        log({"Starting Tensorboard": f"TensorBoard started with PID {tensorboard_process.pid}"}, logger, justify)
        return tensorboard_process
    except Exception as e:
        log(e, logger, justify)
        return None


def stop_tensorboard(tensorboard_process:subprocess.Popen, logger:Logger, justify:int):
    if tensorboard_process:
        log({"Stopping TensorBoard": "Preparing to stop"}, logger, justify)
        tensorboard_process.terminate()
        tensorboard_process.wait()  # Wait for the process to terminate
        log({"Stopping TensorBoard": "Stopped"}, logger, justify)

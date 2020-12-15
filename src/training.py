from subprocess import Popen, PIPE
import platform
from asyncio import sleep
from loadbar import Loadbar

async def monitor_training(monitor, process, args, model_name, file_pointer=None):
    """
    This method monitors the given subprocess 'process'.
    It recieves updates on training accuracy from 'monitor' and detects
    when the training process has terminated. If a max number of epochs
    to train has been set, this method kills the training process when this
    number of epochs is reached.
    """
    final_acc = 0
    if args.loadbar: # Create loadbar, if specified.
        loadbar = Loadbar(50, args.iter, f"Training '{model_name}' ('{args.lang}' dataset)")
        loadbar.print_bar()

    async for test_acc in monitor.on_epoch_complete(process):
        acc_str = ""
        if test_acc is not None:
            acc_str = f"Acc: {test_acc}"

        if args.loadbar:
            loadbar.step(text=acc_str) # Update loadbar progress.
        else:
            acc_str = test_acc if test_acc is not None else f"Epoch: {monitor.epoch}"
            print(acc_str)

        if test_acc is not None: # Accuracy has been intercepted from the process.
            final_acc = test_acc
            if file_pointer is not None:
                file_pointer.write(f"acc_iter_{monitor.epoch}: {final_acc}\n")

        if args.max_iter and monitor.epoch == args.iter:
            # Kill training process if max iterations is reached.
            print("Stopping process.")
            kill_cmd = (["kill", "-9", str(process.pid)] if platform.system() == "Linux"
                        else f"TASKKILL /F /PID {process.pid} /T")
            Popen(kill_cmd, stdout=PIPE)
            await sleep(2)
            break

    return final_acc, 0

def train_imported_model(tagger, train_data):
    """
    Train imported model (NLTK or Flair).
    """
    trained_model = tagger.train(train_data)

    if trained_model is not None:
        tagger.model = trained_model

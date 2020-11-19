from subprocess import Popen, PIPE
from asyncio import sleep
from loadbar import Loadbar

async def monitor_training(monitor, process, args, model_name, file_pointer=None):
    final_acc = 0
    if args.loadbar:
        loadbar = Loadbar(50, args.iter, f"Training '{model_name}' ('{args.lang}' dataset)")
        loadbar.print_bar()

    async for test_acc in monitor.on_epoch_complete(process):
        acc_str = ""
        if test_acc is not None:
            acc_str = f"Acc: {test_acc}"

        if args.loadbar:
            loadbar.step(text=acc_str)
        else:
            acc_str = test_acc if test_acc is not None else f"Epoch: {monitor.epoch}"
            print(acc_str)

        if test_acc is not None:
            final_acc = test_acc
            if file_pointer is not None:
                file_pointer.write(f"acc_iter_{monitor.epoch}: {final_acc}\n")

        if args.max_iter and monitor.epoch == args.iter:
            print("Stopping process.")
            Popen(f"TASKKILL /F /PID {process.pid} /T", stdout=PIPE)
            await sleep(2)
            print(process.returncode)
            break

    return final_acc, 0

def train_imported_model(tagger, train_data):
    trained_model = tagger.train(train_data)

    if trained_model is not None:
        tagger.model = trained_model

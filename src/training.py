from loadbar import Loadbar

async def monitor_training(monitor, process, args, file_pointer=None):
    final_acc = 0
    if args.loadbar:
        loadbar = Loadbar(30, args.iter, f"Training ({args.model_name})")
        loadbar.print_bar()
    async for test_acc in monitor.on_epoch_complete(process):
        if args.loadbar:
            loadbar.step(text=f"Acc: {test_acc:.2f}%")
        else:
            print(f"Test accuracy: {test_acc}")
        final_acc = test_acc
        if file_pointer is not None:
            file_pointer.write(f"acc_iter_{monitor.epoch}: {final_acc}\n")
        if monitor.epoch == args.iter:
            process.kill()
    return final_acc

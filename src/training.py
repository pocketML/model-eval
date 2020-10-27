import loadbar

async def monitor_training(monitor, args, file_pointer=None):
    final_acc = 0
    if not args.no_loadbar:
        loadbar_handler = loadbar.Loadbar(30, args.iter, f"Training ({args.model_name})")
        loadbar_handler.print_bar()
    async for test_acc in monitor.on_epoch_complete():
        if not args.no_loadbar:
            loadbar_handler.step(text=f"Acc: {test_acc:.2f}%")
        else:
            print(f"Test accuracy: {test_acc}")
        final_acc = test_acc
        if file_pointer is not None:
            file_pointer.write(f"acc_iter_{monitor.epoch}: {final_acc}\n")
        if monitor.epoch == args.iter:
            monitor.process_handler.kill()
    return final_acc

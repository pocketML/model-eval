import requests

URL = "https://mhooge.com/pocketml/save_status.php"

def make_request(url, data):
    requests.post(url, data=data)

def send_train_start(model_name, language, total_epochs=None):
    data = {"model_name": model_name, "language": language}
    if total_epochs is not None:
        data["total_epochs"] = total_epochs

    make_request(URL, data)

def send_train_status(epoch, acc=None):
    data = {"epoch": epoch}
    if acc is not None:
        data["accuracy"] = acc

    make_request(URL, data)

if __name__ == "__main__":
    #send_train_start("svmtool", "english", 50)
    send_train_status(1, 0.6)

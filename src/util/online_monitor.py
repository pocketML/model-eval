import requests

URL = "https://mhooge.com/pocketml/save_status.php"

def make_request(url, data):
    try:
        requests.post(url, data=data)
    except requests.exceptions.RequestException as e:
        print("Error ignored in online_monitor: " + str(e))

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

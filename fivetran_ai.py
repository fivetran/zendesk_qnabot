import requests
import sseclient

HOST = "https://fivetranai-api-sgynhqqdzq-uw.a.run.app"


def map_event(msg):
    return {
        "op": msg.event,
        "value": msg.data
    }



class FivetranAI:

    def __init__(self, token):
        self.token = token

    def chat_stream(self, message: str):
        url = f"{HOST}/beta/chat_stream"

        payload = {
            "message": message,
            "history": []  # TODO replace once implemented
        }
        headers = {
            'Content-Type': 'application/json',
            'Accept': 'text/event-stream',
            "Authorization": f"Bearer {self.token}",
        }
        response = requests.post(url=url, headers=headers, json=payload, stream=True)
        client = sseclient.SSEClient(response)
        return map(lambda msg: map_event(msg), client.events())

import json
import time

import requests
import sseclient

HOST = "https://fivetranai-api-sgynhqqdzq-uw.a.run.app"


def map_event(msg):
    if not msg.event or msg.event not in ('update', 'answer'):
        raise ValueError()

    if msg.event == 'update':
        return {
            "op": "update",
            "value": msg.data
        }

    if msg.event == 'answer':
        result = json.loads(msg.data)
        answer = result
        return {
            "op": "answer",
            "value": answer
        }


class FivetranAI:

    def __init__(self, token):
        self.token = token

    def chat_stream(self, message: str):
        url = f"{HOST}/chat_stream"

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

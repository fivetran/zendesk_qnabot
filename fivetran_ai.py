import json

import requests
import sseclient

HOST = "https://fivetranai-api-sgynhqqdzq-uw.a.run.app"


class FivetranAI:

    def __init__(self, token):
        self.token = token

    def chat(self, message: str):
        url = f"{HOST}/chat"

        payload = {
            "message": message,
            "history": []  # TODO replace once implemented
        }
        headers = {
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {self.token}",
        }

        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            result = json.loads(response.text)
            return result
        else:
            raise ValueError(response.text)

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
        return client.events()

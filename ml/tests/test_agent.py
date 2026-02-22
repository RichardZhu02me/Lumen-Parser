import sys
import os

# add backend to python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from unittest.mock import Mock, patch
import unittest
from ml.agent import Agent
from pydantic import BaseModel, Field
from typing import List
import json


class TestAgent(unittest.TestCase):
    def test_agent(self):
        agent = Agent()
        assert isinstance(agent, Agent)

    def test_agent_response_type(self):
        class SampleResponse(BaseModel):
            number: int = Field("a number that is not 42")
            text: str = Field("A passage from Shakespeare")

        class SampleBigResponse(BaseModel):
            sample_responses: List[SampleResponse] = Field("a list of sample responses")

        agent = Agent()
        response = agent.invoke(
            "make me a list of answers to this question: what is the meaning of life?",
            config={
                "response_mime_type": "application/json",
                "response_schema": SampleBigResponse,
            },
        )
        if isinstance(response, str):
            response = json.loads(response)
        self.assertEqual(type(SampleBigResponse(**response)), SampleBigResponse)

    @patch("ml.agent.Agent.invoke", autospec=True)
    def test_call_can_be_mocked(self, mock_chatcompletion):
        mock_chatcompletion.return_value = "mocked response"
        agent = Agent()
        response = agent.invoke("what is the meaning of life?")
        self.assertEqual(response, "mocked response")
        mock_chatcompletion.assert_called_with(agent, "what is the meaning of life?")


if __name__ == "__main__":
    unittest.main()

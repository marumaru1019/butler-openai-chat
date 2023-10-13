from typing import Any

import openai
# To uncomment when enabling asynchronous support.
# from azure.cosmos.aio import ContainerProxy
from approaches.approach import Approach
from approaches.chatlogging import write_chatlog, ApproachType
from core.messagebuilder import MessageBuilder
from core.modelhelper import get_gpt_model, get_max_token_from_messages
import json
from approaches.functioncalling import get_current_time

functions = [
    {
        "name": "get_current_time",
        "description": "Get the current time in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location name. The pytz is used to get the timezone for that location. Location names should be in a format like America/New_York, Asia/Bangkok, Europe/London",
                }
            },
            "required": ["location"],
        },
    }
]


# Simple read implementation, using the OpenAI APIs directly. It uses OpenAI to generate an completion 
# (answer) with that prompt.
class ChatReadApproach(Approach):

    def run(self, user_name: str, history: list[dict[str, str]], overrides: dict[str, Any]) -> Any:
        chat_model = overrides.get("gptModel")
        chat_gpt_model = get_gpt_model(chat_model)
        chat_deployment = chat_gpt_model.get("deployment")

        systemPrompt =  overrides.get("systemPrompt")
        temaperature = float(overrides.get("temperature"))

        user_q = history[-1]["user"]
        message_builder = MessageBuilder(systemPrompt)
        messages = message_builder.get_messages_from_history(
            history, 
            user_q
            )

        max_tokens = get_max_token_from_messages(messages, chat_model)

        # Generate a contextual and content specific answer using chat history
        # Change create type ChatCompletion.create → ChatCompletion.acreate when enabling asynchronous support.
        chat_completion = openai.ChatCompletion.create(
            engine=chat_deployment, 
            messages=messages,
            functions=functions,
            function_call="auto",
            temperature=temaperature, 
            max_tokens=1024,
            n=1)

        # function calling の場合、content が存在しない
        response_text = chat_completion.choices[0]["message"]
        total_tokens = chat_completion.usage.total_tokens

        if "function_call" in response_text:
            function_name = response_text["function_call"]["name"]
            function_args = json.loads(response_text["function_call"]["arguments"])
            
            print("----------------------------------------")
            print("function calling: ")
            print(function_name)

            print("----------------------------------------")
            print("function args: ")
            print(function_args)

            # Call the function if it's the expected one
            if function_name == "get_current_time":
                result = get_current_time(**function_args)
                # Add the function result to the messages and continue the conversation with the AI

                messages.append(
                    {
                        "role": response_text["role"],
                        "function_call": {
                            "name": response_text["function_call"]["name"],
                            "arguments": response_text["function_call"]["arguments"],
                        },
                        "content": None
                    }
                )

                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": result,
                })

                response = openai.ChatCompletion.create(
                            messages=messages,
                            deployment_id=chat_deployment
                        )
                
                response_text = response.choices[0]["message"]
                
            print("final message")
            print("----------------------------------------")
            print(messages)

            print("final response")
            print("----------------------------------------")
            print(response_text["content"])

        # logging
        input_text = history[-1]["user"]
        write_chatlog(ApproachType.Chat, user_name, total_tokens, input_text, response_text)

        return { "answer": response_text["content"] }
    
import json
import boto3
import logging
from agents.memory import load_history, save_message

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)

MODEL_ID = "apac.amazon.nova-lite-v1:0"
BEDROCK_PROFILES = ["own", "default"]
BEDROCK_REGION = "ap-south-1"


def get_bedrock_client():
    """Find a working Bedrock client across profiles."""
    for profile in BEDROCK_PROFILES:
        try:
            session = boto3.Session(profile_name=profile, region_name=BEDROCK_REGION)
            client = session.client("bedrock-runtime")
            client.converse(
                modelId=MODEL_ID,
                messages=[{"role": "user", "content": [{"text": "hi"}]}]
            )
            return client
        except Exception:
            continue
    raise RuntimeError("All Bedrock profiles failed. Check credentials/payment.")


class BaseAgent:
    AGENT_KEY: str = None
    CAPABILITIES: list = []
    SYSTEM_PROMPT: str = "You are a specialized AWS agent. Use your tools to complete the task."

    def __init__(self, name: str, region: str = "us-east-1"):
        self.name = name
        self.region = region
        self.logger = logging.getLogger(name)
        self.session = boto3.Session(region_name=region)
        self._bedrock = None

    @property
    def bedrock(self):
        if not self._bedrock:
            self._bedrock = get_bedrock_client()
        return self._bedrock

    def report(self, status: str, message: str, data: dict = None):
        result = {"agent": self.name, "status": status, "message": message, "data": data or {}}
        self.logger.info(f"[{status}] {message}")
        return result

    @classmethod
    def get_tools(cls) -> list:
        return [
            {
                "toolSpec": {
                    "name": f"{cls.AGENT_KEY}__{cap['name']}",
                    "description": cap["description"],
                    "inputSchema": {"json": cap["input_schema"]},
                }
            }
            for cap in cls.CAPABILITIES
        ]

    def _get_dispatcher(self) -> dict:
        """Map tool_name → callable for this agent's own tools."""
        dispatcher = {}
        for cap in self.CAPABILITIES:
            tool_name = f"{self.AGENT_KEY}__{cap['name']}"
            action = cap["name"]
            dispatcher[tool_name] = lambda inputs, a=action: self.execute({"action": a, **inputs})
        return dispatcher

    def execute(self, task: dict) -> dict:
        """Direct execution — bypasses LLM, calls AWS API directly."""
        raise NotImplementedError

    def run(self, task_description: str, session_id: str = None) -> dict:
        """
        True agentic run — this agent uses its own LLM to reason about
        the task and decide which tools to call.
        """
        self.logger.info(f"Agent received task: {task_description}")
        tools = self.get_tools()
        dispatcher = self._get_dispatcher()

        # Load memory if session_id provided
        history = load_history(session_id) if session_id else []
        new_msg = {"role": "user", "content": [{"text": task_description}]}
        messages = history + [new_msg]
        if session_id:
            save_message(session_id, new_msg)

        all_results = []
        max_iterations = 10

        for iteration in range(max_iterations):
            response = self.bedrock.converse(
                modelId=MODEL_ID,
                system=[{"text": self.SYSTEM_PROMPT}],
                messages=messages,
                toolConfig={"tools": tools},
            )
            output_message = response["output"]["message"]
            messages.append(output_message)
            stop_reason = response["stopReason"]

            if stop_reason == "end_turn":
                summary = next(
                    (b["text"] for b in output_message.get("content", []) if "text" in b), ""
                )
                self.logger.info(f"Task complete: {summary}")
                if session_id:
                    save_message(session_id, output_message)
                return self.report("success", summary, {"results": all_results})

            if stop_reason != "tool_use":
                break

            if session_id:
                save_message(session_id, output_message)

            tool_results = []
            for block in output_message.get("content", []):
                if block.get("type") != "toolUse" and "toolUse" not in block:
                    continue
                tool_block = block.get("toolUse") or block
                tool_name = tool_block["name"]
                tool_use_id = tool_block["toolUseId"]
                inputs = tool_block["input"]

                self.logger.info(f"→ {tool_name}({json.dumps(inputs, default=str)})")
                handler = dispatcher.get(tool_name)
                result = handler(inputs) if handler else {"status": "error", "message": f"Unknown tool: {tool_name}"}
                all_results.append(result)
                self.logger.info(f"← {result['status']}: {result['message']}")

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": json.dumps(result, default=str)}],
                    }
                })

            tool_result_msg = {"role": "user", "content": tool_results}
            messages.append(tool_result_msg)
            if session_id:
                save_message(session_id, tool_result_msg)

        return self.report("completed", f"Agent {self.name} finished", {"results": all_results})

    def run_streaming(self, task_description: str, session_id: str = None):
        """
        Same as run() but yields events for UI streaming:
          {kind: tool_call,   tool, inputs}
          {kind: tool_result, status, message}
          {kind: done,        result}
        """
        tools = self.get_tools()
        dispatcher = self._get_dispatcher()
        history = load_history(session_id) if session_id else []
        new_msg = {"role": "user", "content": [{"text": task_description}]}
        messages = history + [new_msg]
        if session_id:
            save_message(session_id, new_msg)

        all_results = []
        for _ in range(10):
            response = self.bedrock.converse(
                modelId=MODEL_ID,
                system=[{"text": self.SYSTEM_PROMPT}],
                messages=messages,
                toolConfig={"tools": tools},
            )
            output_message = response["output"]["message"]
            messages.append(output_message)
            stop_reason = response["stopReason"]

            if stop_reason == "end_turn":
                summary = next(
                    (b["text"] for b in output_message.get("content", []) if "text" in b), ""
                )
                if session_id:
                    save_message(session_id, output_message)
                result = self.report("success", summary, {"results": all_results})
                yield {"kind": "done", "result": result}
                return

            if stop_reason != "tool_use":
                break

            if session_id:
                save_message(session_id, output_message)

            tool_results = []
            for block in output_message.get("content", []):
                if block.get("type") != "toolUse" and "toolUse" not in block:
                    continue
                tool_block = block.get("toolUse") or block
                tool_name  = tool_block["name"]
                tool_use_id = tool_block["toolUseId"]
                inputs = tool_block["input"]

                yield {"kind": "tool_call", "tool": tool_name, "inputs": inputs}

                handler = dispatcher.get(tool_name)
                result  = handler(inputs) if handler else {"status": "error", "message": f"Unknown tool: {tool_name}"}
                all_results.append(result)

                yield {"kind": "tool_result", "status": result["status"], "message": result["message"]}

                tool_results.append({
                    "toolResult": {
                        "toolUseId": tool_use_id,
                        "content": [{"text": json.dumps(result, default=str)}],
                    }
                })

            tool_result_msg = {"role": "user", "content": tool_results}
            messages.append(tool_result_msg)
            if session_id:
                save_message(session_id, tool_result_msg)

        result = self.report("completed", f"Agent {self.name} finished", {"results": all_results})
        yield {"kind": "done", "result": result}

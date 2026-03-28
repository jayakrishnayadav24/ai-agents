import json
import os
import boto3
import logging
from datetime import datetime, timezone
from typing import List

logger = logging.getLogger("Memory")

TABLE_NAME = os.getenv("MEMORY_TABLE", "agent-memory")
AWS_REGION  = os.getenv("AWS_REGION", "ap-south-1")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")


def _table():
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    return session.resource("dynamodb", region_name=AWS_REGION).Table(TABLE_NAME)


def ensure_table_exists():
    """Create the DynamoDB table if it doesn't exist."""
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    ddb = session.client("dynamodb", region_name=AWS_REGION)
    existing = [t for t in ddb.list_tables()["TableNames"] if t == TABLE_NAME]
    if existing:
        return
    ddb.create_table(
        TableName=TABLE_NAME,
        KeySchema=[
            {"AttributeName": "session_id", "KeyType": "HASH"},
            {"AttributeName": "ts",         "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "session_id", "AttributeType": "S"},
            {"AttributeName": "ts",         "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    ddb.get_waiter("table_exists").wait(TableName=TABLE_NAME)
    logger.info(f"DynamoDB table '{TABLE_NAME}' created")


def save_turn(session_id: str, role: str, content: str):
    """Append one conversation turn to DynamoDB."""
    try:
        _table().put_item(Item={
            "session_id": session_id,
            "ts":         datetime.now(timezone.utc).isoformat(),
            "role":       role,
            "content":    content,
        })
    except Exception as e:
        logger.warning(f"Memory save failed: {e}")


def load_history(session_id: str, max_turns: int = 20) -> List[dict]:
    """
    Load the last `max_turns` plain text messages for a session.
    Skips tool_use / toolResult blocks — those are only valid within
    a single conversation turn and cannot be replayed across runs.
    """
    try:
        resp = _table().query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("session_id").eq(session_id),
            ScanIndexForward=True,
        )
        messages = []
        for item in resp.get("Items", []):
            try:
                content = json.loads(item["content"])
            except (json.JSONDecodeError, TypeError):
                content = [{"text": str(item["content"])}]

            # Keep only blocks that are plain text — drop toolUse / toolResult
            text_blocks = [b for b in content if "text" in b]
            if text_blocks:
                messages.append({"role": item["role"], "content": text_blocks})

        # Return last 6 turns to stay within context window
        messages = messages[-6:]
        # Bedrock requires conversation to start with user
        while messages and messages[0]["role"] != "user":
            messages.pop(0)
        return messages
    except Exception as e:
        logger.warning(f"Memory load failed: {e}")
        return []


def save_message(session_id: str, message: dict):
    """Save a full Bedrock message dict (role + content list)."""
    content = message.get("content", [])
    # Truncate large tool results to avoid filling context window
    trimmed = []
    for block in content:
        if "toolResult" in block:
            text_blocks = block["toolResult"].get("content", [])
            for tb in text_blocks:
                if "text" in tb and len(tb["text"]) > 2000:
                    tb["text"] = tb["text"][:2000] + "...[truncated]"
        trimmed.append(block)
    save_turn(session_id, message["role"], json.dumps(trimmed, default=str))


def save_display(session_id: str, role: str, text: str):
    """Save a clean human-readable message for sidebar display."""
    try:
        _table().put_item(Item={
            "session_id": session_id,
            "ts":         datetime.now(timezone.utc).isoformat(),
            "role":       role,
            "content":    json.dumps([{"text": text}]),
            "display":    "true",
        })
    except Exception as e:
        logger.warning(f"Display save failed: {e}")


def load_display_history(session_id: str) -> List[dict]:
    """Load only clean display messages (user prompts + final AI answers) for the sidebar."""
    try:
        resp = _table().query(
            KeyConditionExpression=boto3.dynamodb.conditions.Key("session_id").eq(session_id),
            FilterExpression=boto3.dynamodb.conditions.Attr("display").eq("true"),
            ScanIndexForward=True,
        )
        result = []
        for item in resp.get("Items", []):
            try:
                content = json.loads(item["content"])
            except (json.JSONDecodeError, TypeError):
                content = [{"text": str(item["content"])}]
            text = " ".join(b["text"] for b in content if "text" in b).strip()
            if text:
                result.append({"role": item["role"], "text": text})
        return result
    except Exception as e:
        logger.warning(f"Display history load failed: {e}")
        return []


def list_sessions() -> List[dict]:
    """Return all sessions that have display messages, with their first user message as title."""
    try:
        resp = _table().scan(
            FilterExpression=boto3.dynamodb.conditions.Attr("display").eq("true")
                           & boto3.dynamodb.conditions.Attr("role").eq("user"),
            ProjectionExpression="session_id, ts, content",
        )
        seen = {}
        for item in resp.get("Items", []):
            sid = item["session_id"]
            if sid not in seen or item["ts"] < seen[sid]["ts"]:
                try:
                    content = json.loads(item["content"])
                    title = " ".join(b["text"] for b in content if "text" in b).strip()[:50]
                except Exception:
                    title = sid
                seen[sid] = {"session_id": sid, "ts": item["ts"], "title": title or sid}
        return sorted(seen.values(), key=lambda x: x["ts"], reverse=True)
    except Exception as e:
        logger.warning(f"list_sessions failed: {e}")
        return []

import json
from agents.base_agent import BaseAgent


class IAMAgent(BaseAgent):
    AGENT_KEY = "iam"
    SYSTEM_PROMPT = """You are an IAM specialist agent. You manage IAM users, groups, roles, and policies.

RULES:
- For creating a user with permissions: create user → create/find group → attach policy to group → add user to group.
- For read-only EC2 access use policy ARN: arn:aws:iam::aws:policy/AmazonEC2ReadOnlyAccess
- For read-only S3 access use policy ARN: arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess
- For admin access use policy ARN: arn:aws:iam::aws:policy/AdministratorAccess
- Never pass policy ARNs as role names.
- Always check if user/group/role exists before creating.
- Act autonomously — never ask for missing info, use sensible defaults."""

    CAPABILITIES = [
        # ── Users ──────────────────────────────────────────────────────────────
        {
            "name": "create_user",
            "description": "Create an IAM user",
            "input_schema": {
                "type": "object",
                "properties": {"username": {"type": "string"}},
                "required": ["username"],
            },
        },
        {
            "name": "list_users",
            "description": "List all IAM users",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "delete_user",
            "description": "Delete an IAM user",
            "input_schema": {
                "type": "object",
                "properties": {"username": {"type": "string"}},
                "required": ["username"],
            },
        },
        {
            "name": "attach_user_policy",
            "description": "Attach a managed policy directly to an IAM user",
            "input_schema": {
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                    "policy_arn": {"type": "string"},
                },
                "required": ["username", "policy_arn"],
            },
        },
        {
            "name": "detach_user_policy",
            "description": "Detach a managed policy from an IAM user",
            "input_schema": {
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                    "policy_arn": {"type": "string"},
                },
                "required": ["username", "policy_arn"],
            },
        },
        {
            "name": "list_user_policies",
            "description": "List all policies attached to an IAM user",
            "input_schema": {
                "type": "object",
                "properties": {"username": {"type": "string"}},
                "required": ["username"],
            },
        },
        {
            "name": "create_access_key",
            "description": "Create an access key for an IAM user",
            "input_schema": {
                "type": "object",
                "properties": {"username": {"type": "string"}},
                "required": ["username"],
            },
        },
        # ── Groups ─────────────────────────────────────────────────────────────
        {
            "name": "create_group",
            "description": "Create an IAM group",
            "input_schema": {
                "type": "object",
                "properties": {"group_name": {"type": "string"}},
                "required": ["group_name"],
            },
        },
        {
            "name": "list_groups",
            "description": "List all IAM groups",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "add_user_to_group",
            "description": "Add an IAM user to a group",
            "input_schema": {
                "type": "object",
                "properties": {
                    "username": {"type": "string"},
                    "group_name": {"type": "string"},
                },
                "required": ["username", "group_name"],
            },
        },
        {
            "name": "attach_group_policy",
            "description": "Attach a managed policy to an IAM group",
            "input_schema": {
                "type": "object",
                "properties": {
                    "group_name": {"type": "string"},
                    "policy_arn": {"type": "string"},
                },
                "required": ["group_name", "policy_arn"],
            },
        },
        # ── Roles ──────────────────────────────────────────────────────────────
        {
            "name": "check_role",
            "description": "Check if an IAM role exists",
            "input_schema": {
                "type": "object",
                "properties": {"role_name": {"type": "string"}},
                "required": ["role_name"],
            },
        },
        {
            "name": "create_role",
            "description": "Create an IAM role for an AWS service with optional managed policies",
            "input_schema": {
                "type": "object",
                "properties": {
                    "role_name": {"type": "string"},
                    "service": {"type": "string", "description": "e.g. ecs-tasks, eks, ec2"},
                    "policies": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["role_name", "service"],
            },
        },
        {
            "name": "list_roles",
            "description": "List all IAM roles",
            "input_schema": {"type": "object", "properties": {}},
        },
        # ── Policies ───────────────────────────────────────────────────────────
        {
            "name": "list_policies",
            "description": "List AWS managed policies, optionally filtered by keyword",
            "input_schema": {
                "type": "object",
                "properties": {"filter": {"type": "string", "description": "keyword to filter policy names e.g. EC2, S3"}},
            },
        },
    ]

    def __init__(self, region="us-east-1"):
        super().__init__("IAMAgent", region)
        self.iam = self.session.client("iam")

    def execute(self, task: dict) -> dict:
        action = task.get("action")
        if action == "create_user":         return self.create_user(task["username"])
        elif action == "list_users":        return self.list_users()
        elif action == "delete_user":       return self.delete_user(task["username"])
        elif action == "attach_user_policy": return self.attach_user_policy(task["username"], task["policy_arn"])
        elif action == "detach_user_policy": return self.detach_user_policy(task["username"], task["policy_arn"])
        elif action == "list_user_policies": return self.list_user_policies(task["username"])
        elif action == "create_access_key": return self.create_access_key(task["username"])
        elif action == "create_group":      return self.create_group(task["group_name"])
        elif action == "list_groups":       return self.list_groups()
        elif action == "add_user_to_group": return self.add_user_to_group(task["username"], task["group_name"])
        elif action == "attach_group_policy": return self.attach_group_policy(task["group_name"], task["policy_arn"])
        elif action == "check_role":        return self.check_role(task["role_name"])
        elif action == "create_role":       return self.create_role(task["role_name"], task.get("service"), task.get("policies", []))
        elif action == "list_roles":        return self.list_roles()
        elif action == "list_policies":     return self.list_policies(task.get("filter", ""))
        return self.report("error", f"Unknown IAM action: {action}")

    # ── Users ──────────────────────────────────────────────────────────────────
    def create_user(self, username: str) -> dict:
        try:
            self.iam.create_user(UserName=username)
            return self.report("created", f"IAM user '{username}' created", {"username": username})
        except self.iam.exceptions.EntityAlreadyExistsException:
            return self.report("exists", f"IAM user '{username}' already exists", {"username": username})
        except Exception as e:
            return self.report("error", str(e))

    def list_users(self) -> dict:
        try:
            users = [{"username": u["UserName"], "arn": u["Arn"], "created": str(u["CreateDate"])}
                     for u in self.iam.list_users()["Users"]]
            return self.report("success", f"Found {len(users)} IAM users", {"users": users})
        except Exception as e:
            return self.report("error", str(e))

    def delete_user(self, username: str) -> dict:
        try:
            # Detach all policies first
            policies = self.iam.list_attached_user_policies(UserName=username)["AttachedPolicies"]
            for p in policies:
                self.iam.detach_user_policy(UserName=username, PolicyArn=p["PolicyArn"])
            # Remove from all groups
            groups = self.iam.list_groups_for_user(UserName=username)["Groups"]
            for g in groups:
                self.iam.remove_user_from_group(UserName=username, GroupName=g["GroupName"])
            # Delete access keys
            keys = self.iam.list_access_keys(UserName=username)["AccessKeyMetadata"]
            for k in keys:
                self.iam.delete_access_key(UserName=username, AccessKeyId=k["AccessKeyId"])
            self.iam.delete_user(UserName=username)
            return self.report("deleted", f"IAM user '{username}' deleted")
        except Exception as e:
            return self.report("error", str(e))

    def attach_user_policy(self, username: str, policy_arn: str) -> dict:
        try:
            self.iam.attach_user_policy(UserName=username, PolicyArn=policy_arn)
            return self.report("success", f"Policy '{policy_arn}' attached to user '{username}'")
        except Exception as e:
            return self.report("error", str(e))

    def detach_user_policy(self, username: str, policy_arn: str) -> dict:
        try:
            self.iam.detach_user_policy(UserName=username, PolicyArn=policy_arn)
            return self.report("success", f"Policy detached from user '{username}'")
        except Exception as e:
            return self.report("error", str(e))

    def list_user_policies(self, username: str) -> dict:
        try:
            policies = self.iam.list_attached_user_policies(UserName=username)["AttachedPolicies"]
            return self.report("success", f"User '{username}' has {len(policies)} policies",
                               {"policies": [{"name": p["PolicyName"], "arn": p["PolicyArn"]} for p in policies]})
        except Exception as e:
            return self.report("error", str(e))

    def create_access_key(self, username: str) -> dict:
        try:
            key = self.iam.create_access_key(UserName=username)["AccessKey"]
            return self.report("created", f"Access key created for '{username}'", {
                "access_key_id": key["AccessKeyId"],
                "secret_access_key": key["SecretAccessKey"],
            })
        except Exception as e:
            return self.report("error", str(e))

    # ── Groups ─────────────────────────────────────────────────────────────────
    def create_group(self, group_name: str) -> dict:
        try:
            self.iam.create_group(GroupName=group_name)
            return self.report("created", f"IAM group '{group_name}' created")
        except self.iam.exceptions.EntityAlreadyExistsException:
            return self.report("exists", f"IAM group '{group_name}' already exists")
        except Exception as e:
            return self.report("error", str(e))

    def list_groups(self) -> dict:
        try:
            groups = [{"name": g["GroupName"], "arn": g["Arn"]} for g in self.iam.list_groups()["Groups"]]
            return self.report("success", f"Found {len(groups)} IAM groups", {"groups": groups})
        except Exception as e:
            return self.report("error", str(e))

    def add_user_to_group(self, username: str, group_name: str) -> dict:
        try:
            self.iam.add_user_to_group(UserName=username, GroupName=group_name)
            return self.report("success", f"User '{username}' added to group '{group_name}'")
        except Exception as e:
            return self.report("error", str(e))

    def attach_group_policy(self, group_name: str, policy_arn: str) -> dict:
        try:
            self.iam.attach_group_policy(GroupName=group_name, PolicyArn=policy_arn)
            return self.report("success", f"Policy '{policy_arn}' attached to group '{group_name}'")
        except Exception as e:
            return self.report("error", str(e))

    # ── Roles ──────────────────────────────────────────────────────────────────
    def check_role(self, role_name: str) -> dict:
        try:
            role = self.iam.get_role(RoleName=role_name)
            return self.report("success", f"Role '{role_name}' exists", {"arn": role["Role"]["Arn"]})
        except self.iam.exceptions.NoSuchEntityException:
            return self.report("not_found", f"Role '{role_name}' does not exist")

    def create_role(self, role_name: str, service: str, policies: list) -> dict:
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow", "Principal": {"Service": f"{service}.amazonaws.com"}, "Action": "sts:AssumeRole"}],
        }
        try:
            role = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description=f"Auto-created role for {service}",
            )
            for policy_arn in policies:
                self.iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
            return self.report("created", f"Role '{role_name}' created", {"arn": role["Role"]["Arn"]})
        except self.iam.exceptions.EntityAlreadyExistsException:
            role = self.iam.get_role(RoleName=role_name)
            return self.report("exists", f"Role '{role_name}' already exists", {"arn": role["Role"]["Arn"]})
        except Exception as e:
            return self.report("error", str(e))

    def list_roles(self) -> dict:
        try:
            roles = [{"name": r["RoleName"], "arn": r["Arn"]} for r in self.iam.list_roles()["Roles"]]
            return self.report("success", f"Found {len(roles)} IAM roles", {"roles": roles})
        except Exception as e:
            return self.report("error", str(e))

    # ── Policies ───────────────────────────────────────────────────────────────
    def list_policies(self, filter: str = "") -> dict:
        try:
            paginator = self.iam.get_paginator("list_policies")
            policies = []
            for page in paginator.paginate(Scope="AWS"):
                for p in page["Policies"]:
                    if not filter or filter.lower() in p["PolicyName"].lower():
                        policies.append({"name": p["PolicyName"], "arn": p["Arn"]})
            return self.report("success", f"Found {len(policies)} policies", {"policies": policies[:50]})
        except Exception as e:
            return self.report("error", str(e))

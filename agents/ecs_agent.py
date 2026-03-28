from agents.base_agent import BaseAgent
from agents.iam_agent import IAMAgent

ECS_TASK_EXECUTION_ROLE = "ecsTaskExecutionRole"
ECS_TASK_EXECUTION_POLICY = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"


class ECSAgent(BaseAgent):
    AGENT_KEY = "ecs"
    SYSTEM_PROMPT = """You are an ECS specialist agent. You manage ECS clusters, services, and task definitions.
RULES:
- NEVER ask the user for subnet IDs, VPC IDs, or security group IDs. Always discover them yourself.
- Before creating a service, ALWAYS call ecs__list_subnets to get real subnet IDs.
- Before creating a service, ALWAYS call ecs__list_security_groups to get the default security group ID.
- Use the first 2 subnet IDs from ecs__list_subnets and the default security group ID from ecs__list_security_groups.
- For DELETE: call ecs__list_services first, delete each service, then delete the cluster.
- For DIAGNOSE: use ecs__diagnose_service and ecs__fix_stopped_tasks.
- Auto-create IAM execution roles when needed.
- Never stop and ask for input. Always proceed autonomously."""

    CAPABILITIES = [
        {
            "name": "create_cluster",
            "description": "Create an ECS cluster",
            "input_schema": {"type": "object", "properties": {"cluster_name": {"type": "string"}}, "required": ["cluster_name"]},
        },
        {
            "name": "list_clusters",
            "description": "List all ECS clusters in the account",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "describe_cluster",
            "description": "Describe an ECS cluster — shows status, running/pending task counts, registered container instances",
            "input_schema": {"type": "object", "properties": {"cluster": {"type": "string"}}, "required": ["cluster"]},
        },
        {
            "name": "register_task",
            "description": "Register an ECS task definition (auto-creates execution role if missing)",
            "input_schema": {
                "type": "object",
                "properties": {
                    "family": {"type": "string"},
                    "cpu": {"type": "string"},
                    "memory": {"type": "string"},
                    "network_mode": {"type": "string"},
                    "requires_compatibilities": {"type": "array", "items": {"type": "string"}},
                    "container_definitions": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["family", "container_definitions"],
            },
        },
        {
            "name": "create_service",
            "description": "Create an ECS service in a cluster",
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster": {"type": "string"},
                    "service_name": {"type": "string"},
                    "task_definition": {"type": "string"},
                    "desired_count": {"type": "integer"},
                    "launch_type": {"type": "string"},
                    "subnets": {"type": "array", "items": {"type": "string"}},
                    "security_groups": {"type": "array", "items": {"type": "string"}},
                    "assign_public_ip": {"type": "string"},
                },
                "required": ["cluster", "service_name", "task_definition", "subnets"],
            },
        },
        {
            "name": "list_services",
            "description": "List all service names in an ECS cluster",
            "input_schema": {"type": "object", "properties": {"cluster": {"type": "string"}}, "required": ["cluster"]},
        },
        {
            "name": "describe_services",
            "description": "Describe ECS services — use list_services first to get service names, never pass empty list",
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster": {"type": "string"},
                    "services": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["cluster", "services"],
            },
        },
        {
            "name": "diagnose_service",
            "description": "Diagnose an ECS service — checks desired/running/pending counts, events, and deployment status",
            "input_schema": {
                "type": "object",
                "properties": {"cluster": {"type": "string"}, "service": {"type": "string"}},
                "required": ["cluster", "service"],
            },
        },
        {
            "name": "fix_stopped_tasks",
            "description": "Diagnose stopped ECS tasks in a cluster — returns stop reasons for each task",
            "input_schema": {"type": "object", "properties": {"cluster": {"type": "string"}}, "required": ["cluster"]},
        },
        {
            "name": "list_running_tasks",
            "description": "List all running tasks in an ECS cluster with their task definition and container status",
            "input_schema": {"type": "object", "properties": {"cluster": {"type": "string"}}, "required": ["cluster"]},
        },
        {
            "name": "delete_service",
            "description": "Delete an ECS service (sets desired count to 0 then deletes)",
            "input_schema": {
                "type": "object",
                "properties": {"cluster": {"type": "string"}, "service": {"type": "string"}},
                "required": ["cluster", "service"],
            },
        },
        {
            "name": "delete_cluster",
            "description": "Delete an ECS cluster",
            "input_schema": {"type": "object", "properties": {"cluster": {"type": "string"}}, "required": ["cluster"]},
        },
        {
            "name": "list_subnets",
            "description": "List all subnets with IDs, VPC, AZ — use this to get subnet IDs before creating a service",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "list_security_groups",
            "description": "List all security groups — use this to get the default security group ID before creating a service",
            "input_schema": {"type": "object", "properties": {}},
        },
    ]

    def __init__(self, region="us-east-1"):
        super().__init__("ECSAgent", region)
        self.ecs = self.session.client("ecs")
        self.ec2 = self.session.client("ec2")
        self.iam_agent = IAMAgent(region)

    def list_subnets(self) -> dict:
        try:
            subnets = [{"subnet_id": s["SubnetId"], "vpc_id": s["VpcId"], "az": s["AvailabilityZone"]}
                       for s in self.ec2.describe_subnets()["Subnets"]]
            return self.report("success", f"Found {len(subnets)} subnets", {"subnets": subnets})
        except Exception as e:
            return self.report("error", str(e))

    def list_security_groups(self) -> dict:
        try:
            sgs = [{"group_id": sg["GroupId"], "name": sg["GroupName"], "vpc_id": sg.get("VpcId", "")}
                   for sg in self.ec2.describe_security_groups()["SecurityGroups"]]
            return self.report("success", f"Found {len(sgs)} security groups", {"security_groups": sgs})
        except Exception as e:
            return self.report("error", str(e))

    def _get_default_subnets(self) -> list:
        """Auto-fetch first 2 subnet IDs."""
        try:
            subnets = self.ec2.describe_subnets()["Subnets"]
            return [s["SubnetId"] for s in subnets[:2]]
        except Exception:
            return []

    def _get_default_sg(self) -> list:
        """Auto-fetch the default security group ID."""
        try:
            sgs = self.ec2.describe_security_groups(
                Filters=[{"Name": "group-name", "Values": ["default"]}]
            )["SecurityGroups"]
            return [sgs[0]["GroupId"]] if sgs else []
        except Exception:
            return []

    def execute(self, task: dict) -> dict:
        action = task.get("action")
        if action == "create_cluster":      return self.create_cluster(task["cluster_name"])
        elif action == "list_clusters":     return self.list_clusters()
        elif action == "describe_cluster":  return self.describe_cluster(task["cluster"])
        elif action == "register_task":     return self.register_task_definition(task)
        elif action == "create_service":    return self.create_service(task)
        elif action == "list_services":     return self.list_services(task["cluster"])
        elif action == "describe_services": return self.describe_services(task["cluster"], task.get("services", []))
        elif action == "diagnose_service":  return self.diagnose_service(task["cluster"], task["service"])
        elif action == "fix_stopped_tasks": return self.fix_stopped_tasks(task["cluster"])
        elif action == "list_running_tasks": return self.list_running_tasks(task["cluster"])
        elif action == "delete_service":    return self.delete_service(task["cluster"], task["service"])
        elif action == "delete_cluster":    return self.delete_cluster(task["cluster"])
        elif action == "list_subnets":      return self.list_subnets()
        elif action == "list_security_groups": return self.list_security_groups()
        return self.report("error", f"Unknown ECS action: {action}")

    def _ensure_execution_role(self) -> str:
        result = self.iam_agent.check_role(ECS_TASK_EXECUTION_ROLE)
        if result["status"] == "not_found":
            result = self.iam_agent.create_role(ECS_TASK_EXECUTION_ROLE, "ecs-tasks", [ECS_TASK_EXECUTION_POLICY])
        return result["data"].get("arn", "")

    def _ensure_service_linked_role(self):
        try:
            self.iam_agent.iam.create_service_linked_role(AWSServiceName="ecs.amazonaws.com")
        except self.iam_agent.iam.exceptions.InvalidInputException:
            pass

    def create_cluster(self, cluster_name: str) -> dict:
        try:
            resp = self.ecs.create_cluster(clusterName=cluster_name, settings=[{"name": "containerInsights", "value": "enabled"}])
            return self.report("created", f"ECS cluster '{cluster_name}' created", {"cluster_arn": resp["cluster"]["clusterArn"]})
        except Exception as e:
            return self.report("error", str(e))

    def list_clusters(self) -> dict:
        try:
            arns = self.ecs.list_clusters()["clusterArns"]
            names = [a.split("/")[-1] for a in arns]
            return self.report("success", f"Found {len(names)} clusters", {"clusters": names})
        except Exception as e:
            return self.report("error", str(e))

    def describe_cluster(self, cluster: str) -> dict:
        try:
            resp = self.ecs.describe_clusters(clusters=[cluster])["clusters"]
            if not resp:
                return self.report("not_found", f"Cluster '{cluster}' not found")
            c = resp[0]
            return self.report("success", f"Cluster '{cluster}' status: {c['status']}", {
                "status": c["status"],
                "running_tasks": c.get("runningTasksCount", 0),
                "pending_tasks": c.get("pendingTasksCount", 0),
                "registered_instances": c.get("registeredContainerInstancesCount", 0),
                "active_services": c.get("activeServicesCount", 0),
            })
        except Exception as e:
            return self.report("error", str(e))

    def register_task_definition(self, task: dict) -> dict:
        role_arn = self._ensure_execution_role()
        try:
            resp = self.ecs.register_task_definition(
                family=task["family"],
                networkMode=task.get("network_mode", "awsvpc"),
                requiresCompatibilities=task.get("requires_compatibilities", ["FARGATE"]),
                cpu=task.get("cpu", "256"),
                memory=task.get("memory", "512"),
                executionRoleArn=role_arn,
                containerDefinitions=task["container_definitions"],
            )
            td = resp["taskDefinition"]
            return self.report("registered", f"Task definition '{task['family']}' registered", {"task_definition_arn": td["taskDefinitionArn"]})
        except Exception as e:
            return self.report("error", str(e))

    def create_service(self, task: dict) -> dict:
        self._ensure_service_linked_role()
        # Auto-discover subnets and security groups if not provided
        subnets = task.get("subnets") or self._get_default_subnets()
        security_groups = task.get("security_groups") or self._get_default_sg()
        if not subnets:
            return self.report("error", "No subnets found in this account/region.")
        try:
            resp = self.ecs.create_service(
                cluster=task["cluster"],
                serviceName=task["service_name"],
                taskDefinition=task["task_definition"],
                desiredCount=task.get("desired_count", 1),
                launchType=task.get("launch_type", "FARGATE"),
                networkConfiguration={
                    "awsvpcConfiguration": {
                        "subnets": subnets,
                        "securityGroups": security_groups,
                        "assignPublicIp": task.get("assign_public_ip", "ENABLED"),
                    }
                },
            )
            return self.report("created", f"ECS service '{task['service_name']}' created", {"service_arn": resp["service"]["serviceArn"]})
        except Exception as e:
            return self.report("error", str(e))

    def list_services(self, cluster: str) -> dict:
        try:
            arns = self.ecs.list_services(cluster=cluster)["serviceArns"]
            names = [a.split("/")[-1] for a in arns]
            return self.report("success", f"Found {len(names)} services", {"services": names})
        except Exception as e:
            return self.report("error", str(e))

    def describe_services(self, cluster: str, services: list) -> dict:
        try:
            resp = self.ecs.describe_services(cluster=cluster, services=services)
            return self.report("success", f"Described {len(resp['services'])} services", {"services": resp["services"]})
        except Exception as e:
            return self.report("error", str(e))

    def diagnose_service(self, cluster: str, service: str) -> dict:
        try:
            resp = self.ecs.describe_services(cluster=cluster, services=[service])
            if not resp["services"]:
                return self.report("not_found", f"Service '{service}' not found in cluster '{cluster}'")
            svc = resp["services"][0]
            events = [e["message"] for e in svc.get("events", [])[:5]]
            deployments = [
                {"status": d["status"], "desired": d["desiredCount"], "running": d["runningCount"], "failed": d["failedTasks"]}
                for d in svc.get("deployments", [])
            ]
            return self.report("success", f"Service '{service}' status: {svc['status']}", {
                "status": svc["status"],
                "desired": svc["desiredCount"],
                "running": svc["runningCount"],
                "pending": svc["pendingCount"],
                "task_definition": svc["taskDefinition"],
                "recent_events": events,
                "deployments": deployments,
            })
        except Exception as e:
            return self.report("error", str(e))

    def fix_stopped_tasks(self, cluster: str) -> dict:
        try:
            stopped = self.ecs.list_tasks(cluster=cluster, desiredStatus="STOPPED")["taskArns"]
            if not stopped:
                return self.report("success", "No stopped tasks found")
            details = self.ecs.describe_tasks(cluster=cluster, tasks=stopped)["tasks"]
            reasons = {t["taskArn"].split("/")[-1]: t.get("stoppedReason", "unknown") for t in details}
            return self.report("diagnosed", f"Found {len(stopped)} stopped tasks", {"reasons": reasons})
        except Exception as e:
            return self.report("error", str(e))

    def list_running_tasks(self, cluster: str) -> dict:
        try:
            arns = self.ecs.list_tasks(cluster=cluster, desiredStatus="RUNNING")["taskArns"]
            if not arns:
                return self.report("success", "No running tasks found", {"tasks": []})
            details = self.ecs.describe_tasks(cluster=cluster, tasks=arns)["tasks"]
            tasks = [
                {
                    "task_id": t["taskArn"].split("/")[-1],
                    "task_definition": t["taskDefinitionArn"].split("/")[-1],
                    "last_status": t["lastStatus"],
                    "containers": [{"name": c["name"], "status": c["lastStatus"]} for c in t.get("containers", [])],
                }
                for t in details
            ]
            return self.report("success", f"Found {len(tasks)} running tasks", {"tasks": tasks})
        except Exception as e:
            return self.report("error", str(e))

    def delete_service(self, cluster: str, service: str) -> dict:
        try:
            self.ecs.update_service(cluster=cluster, service=service, desiredCount=0)
            self.ecs.delete_service(cluster=cluster, service=service, force=True)
            return self.report("deleted", f"Service '{service}' deleted from cluster '{cluster}'")
        except Exception as e:
            return self.report("error", str(e))

    def delete_cluster(self, cluster: str) -> dict:
        try:
            # Stop all running tasks first
            task_arns = self.ecs.list_tasks(cluster=cluster, desiredStatus="RUNNING").get("taskArns", [])
            for arn in task_arns:
                self.ecs.stop_task(cluster=cluster, task=arn, reason="Cluster deletion")
            self.ecs.delete_cluster(cluster=cluster)
            return self.report("deleted", f"Cluster '{cluster}' deleted")
        except Exception as e:
            return self.report("error", str(e))

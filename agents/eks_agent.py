import time
from agents.base_agent import BaseAgent
from agents.iam_agent import IAMAgent

EKS_CLUSTER_ROLE = "eksClusterRole"
EKS_NODE_ROLE = "eksNodeRole"
EKS_CLUSTER_POLICY = "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
EKS_WORKER_POLICIES = [
    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
]


class EKSAgent(BaseAgent):
    AGENT_KEY = "eks"
    SYSTEM_PROMPT = """You are an EKS specialist agent. You manage EKS clusters and nodegroups.
- Always call eks__list_supported_versions first before creating a cluster, use the latest version.
- After creating a cluster, call eks__wait_for_cluster before creating nodegroups.
- For DELETE: call eks__list_nodegroups, delete each nodegroup (wait for deletion), then delete the cluster. Never call create or wait tools during delete.
- For DIAGNOSE: use eks__diagnose_cluster and eks__describe_nodegroup."""

    CAPABILITIES = [
        {
            "name": "list_clusters",
            "description": "List all EKS clusters in the account",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "create_cluster",
            "description": "Create an EKS cluster (auto-creates cluster IAM role if missing). Call list_supported_versions first.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster_name": {"type": "string"},
                    "k8s_version": {"type": "string"},
                    "subnets": {"type": "array", "items": {"type": "string"}},
                    "security_groups": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["cluster_name", "subnets"],
            },
        },
        {
            "name": "describe_cluster",
            "description": "Describe an EKS cluster — status, endpoint, k8s version, VPC config",
            "input_schema": {"type": "object", "properties": {"cluster_name": {"type": "string"}}, "required": ["cluster_name"]},
        },
        {
            "name": "wait_for_cluster",
            "description": "Wait until an EKS cluster reaches ACTIVE status (polls every 30s, max 20 min). Call before creating nodegroups.",
            "input_schema": {"type": "object", "properties": {"cluster_name": {"type": "string"}}, "required": ["cluster_name"]},
        },
        {
            "name": "diagnose_cluster",
            "description": "Diagnose an EKS cluster — checks status, logging, endpoint access, and nodegroups",
            "input_schema": {"type": "object", "properties": {"cluster_name": {"type": "string"}}, "required": ["cluster_name"]},
        },
        {
            "name": "create_nodegroup",
            "description": "Create a managed node group (auto-creates node IAM role if missing). Cluster must be ACTIVE first.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster_name": {"type": "string"},
                    "nodegroup_name": {"type": "string"},
                    "instance_type": {"type": "string"},
                    "min_size": {"type": "integer"},
                    "max_size": {"type": "integer"},
                    "desired_size": {"type": "integer"},
                    "subnets": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["cluster_name", "subnets"],
            },
        },
        {
            "name": "list_nodegroups",
            "description": "List all node groups in an EKS cluster",
            "input_schema": {"type": "object", "properties": {"cluster_name": {"type": "string"}}, "required": ["cluster_name"]},
        },
        {
            "name": "describe_nodegroup",
            "description": "Describe the status, scaling config, and health of an EKS node group",
            "input_schema": {
                "type": "object",
                "properties": {"cluster_name": {"type": "string"}, "nodegroup_name": {"type": "string"}},
                "required": ["cluster_name", "nodegroup_name"],
            },
        },
        {
            "name": "fix_nodegroup",
            "description": "Update scaling config of an EKS node group. Only call when nodegroup is ACTIVE.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster_name": {"type": "string"},
                    "nodegroup_name": {"type": "string"},
                    "min_size": {"type": "integer"},
                    "max_size": {"type": "integer"},
                    "desired_size": {"type": "integer"},
                },
                "required": ["cluster_name", "nodegroup_name"],
            },
        },
        {
            "name": "list_supported_versions",
            "description": "List supported Kubernetes versions for EKS in this region",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "delete_nodegroup",
            "description": "Delete a managed node group from an EKS cluster",
            "input_schema": {
                "type": "object",
                "properties": {
                    "cluster_name": {"type": "string"},
                    "nodegroup_name": {"type": "string"},
                },
                "required": ["cluster_name", "nodegroup_name"],
            },
        },
        {
            "name": "delete_cluster",
            "description": "Delete an EKS cluster. Must delete all nodegroups first.",
            "input_schema": {
                "type": "object",
                "properties": {"cluster_name": {"type": "string"}},
                "required": ["cluster_name"],
            },
        },
    ]

    def __init__(self, region="us-east-1"):
        super().__init__("EKSAgent", region)
        self.eks = self.session.client("eks")
        self.iam_agent = IAMAgent(region)

    def execute(self, task: dict) -> dict:
        action = task.get("action")
        if action == "list_clusters":           return self.list_clusters()
        elif action == "create_cluster":        return self.create_cluster(task)
        elif action == "describe_cluster":      return self.describe_cluster(task["cluster_name"])
        elif action == "wait_for_cluster":      return self.wait_for_cluster(task["cluster_name"])
        elif action == "diagnose_cluster":      return self.diagnose_cluster(task["cluster_name"])
        elif action == "create_nodegroup":      return self.create_nodegroup(task)
        elif action == "list_nodegroups":       return self.list_nodegroups(task["cluster_name"])
        elif action == "describe_nodegroup":    return self.describe_nodegroup(task["cluster_name"], task["nodegroup_name"])
        elif action == "fix_nodegroup":         return self.fix_nodegroup(task)
        elif action == "list_supported_versions": return self.list_supported_versions()
        elif action == "delete_nodegroup":      return self.delete_nodegroup(task["cluster_name"], task["nodegroup_name"])
        elif action == "delete_cluster":        return self.delete_cluster(task["cluster_name"])
        return self.report("error", f"Unknown EKS action: {action}")

    def _ensure_cluster_role(self) -> str:
        result = self.iam_agent.check_role(EKS_CLUSTER_ROLE)
        if result["status"] == "not_found":
            result = self.iam_agent.create_role(EKS_CLUSTER_ROLE, "eks", [EKS_CLUSTER_POLICY])
        return result["data"].get("arn", "")

    def _ensure_node_role(self) -> str:
        result = self.iam_agent.check_role(EKS_NODE_ROLE)
        if result["status"] == "not_found":
            result = self.iam_agent.create_role(EKS_NODE_ROLE, "ec2", EKS_WORKER_POLICIES)
        return result["data"].get("arn", "")

    def list_clusters(self) -> dict:
        try:
            names = self.eks.list_clusters()["clusters"]
            return self.report("success", f"Found {len(names)} clusters", {"clusters": names})
        except Exception as e:
            return self.report("error", str(e))

    def create_cluster(self, task: dict) -> dict:
        role_arn = self._ensure_cluster_role()
        try:
            resp = self.eks.create_cluster(
                name=task["cluster_name"],
                version=task.get("k8s_version", "1.32"),
                roleArn=role_arn,
                resourcesVpcConfig={
                    "subnetIds": task["subnets"],
                    "securityGroupIds": task.get("security_groups", []),
                    "endpointPublicAccess": True,
                },
            )
            return self.report("creating", f"EKS cluster '{task['cluster_name']}' creation initiated", {"cluster_arn": resp["cluster"]["arn"]})
        except self.eks.exceptions.ResourceInUseException:
            return self.report("exists", f"Cluster '{task['cluster_name']}' already exists, proceeding")
        except Exception as e:
            return self.report("error", str(e))

    def describe_cluster(self, cluster_name: str) -> dict:
        try:
            resp = self.eks.describe_cluster(name=cluster_name)
            c = resp["cluster"]
            return self.report("success", f"Cluster '{cluster_name}' status: {c['status']}", {
                "status": c["status"],
                "version": c.get("version"),
                "endpoint": c.get("endpoint"),
                "role_arn": c.get("roleArn"),
                "logging": c.get("logging", {}),
            })
        except Exception as e:
            return self.report("error", str(e))

    def wait_for_cluster(self, cluster_name: str) -> dict:
        self.logger.info(f"Waiting for cluster '{cluster_name}' to become ACTIVE...")
        for _ in range(40):
            try:
                status = self.eks.describe_cluster(name=cluster_name)["cluster"]["status"]
                if status == "ACTIVE":
                    return self.report("success", f"Cluster '{cluster_name}' is ACTIVE", {"status": status})
                if status == "FAILED":
                    return self.report("error", f"Cluster '{cluster_name}' FAILED")
                self.logger.info(f"Cluster status: {status}, waiting 30s...")
            except Exception as e:
                return self.report("error", str(e))
            time.sleep(30)
        return self.report("error", f"Cluster '{cluster_name}' did not become ACTIVE in time")

    def diagnose_cluster(self, cluster_name: str) -> dict:
        try:
            c = self.eks.describe_cluster(name=cluster_name)["cluster"]
            nodegroups = self.eks.list_nodegroups(clusterName=cluster_name)["nodegroups"]
            ng_details = []
            for ng in nodegroups:
                ng_resp = self.eks.describe_nodegroup(clusterName=cluster_name, nodegroupName=ng)["nodegroup"]
                ng_details.append({
                    "name": ng,
                    "status": ng_resp["status"],
                    "desired": ng_resp["scalingConfig"]["desiredSize"],
                    "min": ng_resp["scalingConfig"]["minSize"],
                    "max": ng_resp["scalingConfig"]["maxSize"],
                    "health": ng_resp.get("health", {}).get("issues", []),
                })
            issues = []
            if c["status"] != "ACTIVE":
                issues.append(f"Cluster status is {c['status']}, expected ACTIVE")
            for ng in ng_details:
                if ng["status"] != "ACTIVE":
                    issues.append(f"Nodegroup '{ng['name']}' status is {ng['status']}")
                if ng["health"]:
                    issues.append(f"Nodegroup '{ng['name']}' has health issues: {ng['health']}")
            return self.report("success" if not issues else "issues_found", f"Cluster '{cluster_name}' diagnosis complete", {
                "cluster_status": c["status"],
                "version": c.get("version"),
                "nodegroups": ng_details,
                "issues": issues,
            })
        except Exception as e:
            return self.report("error", str(e))

    def create_nodegroup(self, task: dict) -> dict:
        node_role_arn = self._ensure_node_role()
        try:
            resp = self.eks.create_nodegroup(
                clusterName=task["cluster_name"],
                nodegroupName=task.get("nodegroup_name", "default-ng"),
                scalingConfig={"minSize": task.get("min_size", 1), "maxSize": task.get("max_size", 3), "desiredSize": task.get("desired_size", 2)},
                subnets=task["subnets"],
                instanceTypes=[task.get("instance_type", "t3.medium")],
                nodeRole=node_role_arn,
            )
            return self.report("creating", f"Node group '{task.get('nodegroup_name')}' creation initiated", {"nodegroup_arn": resp["nodegroup"]["nodegroupArn"]})
        except self.eks.exceptions.ResourceInUseException:
            return self.report("exists", f"Nodegroup '{task.get('nodegroup_name')}' already exists")
        except Exception as e:
            return self.report("error", str(e))

    def list_nodegroups(self, cluster_name: str) -> dict:
        try:
            names = self.eks.list_nodegroups(clusterName=cluster_name)["nodegroups"]
            return self.report("success", f"Found {len(names)} nodegroups", {"nodegroups": names})
        except Exception as e:
            return self.report("error", str(e))

    def describe_nodegroup(self, cluster_name: str, nodegroup_name: str) -> dict:
        try:
            ng = self.eks.describe_nodegroup(clusterName=cluster_name, nodegroupName=nodegroup_name)["nodegroup"]
            return self.report("success", f"Nodegroup '{nodegroup_name}' status: {ng['status']}", {
                "status": ng["status"],
                "nodegroup_arn": ng["nodegroupArn"],
                "instance_types": ng.get("instanceTypes", []),
                "scaling": ng["scalingConfig"],
                "health_issues": ng.get("health", {}).get("issues", []),
            })
        except Exception as e:
            return self.report("error", str(e))

    def fix_nodegroup(self, task: dict) -> dict:
        try:
            self.eks.update_nodegroup_config(
                clusterName=task["cluster_name"],
                nodegroupName=task["nodegroup_name"],
                scalingConfig={"minSize": task.get("min_size", 1), "maxSize": task.get("max_size", 5), "desiredSize": task.get("desired_size", 2)},
            )
            return self.report("success", f"Node group '{task['nodegroup_name']}' scaling updated")
        except Exception as e:
            return self.report("error", str(e))

    def delete_nodegroup(self, cluster_name: str, nodegroup_name: str) -> dict:
        try:
            self.eks.delete_nodegroup(clusterName=cluster_name, nodegroupName=nodegroup_name)
            self.logger.info(f"Waiting for nodegroup '{nodegroup_name}' to be deleted...")
            for _ in range(40):  # max 20 min
                try:
                    status = self.eks.describe_nodegroup(clusterName=cluster_name, nodegroupName=nodegroup_name)["nodegroup"]["status"]
                    self.logger.info(f"Nodegroup status: {status}")
                    time.sleep(30)
                except self.eks.exceptions.ResourceNotFoundException:
                    return self.report("deleted", f"Nodegroup '{nodegroup_name}' deleted from cluster '{cluster_name}'")
            return self.report("error", f"Nodegroup '{nodegroup_name}' did not delete in time")
        except Exception as e:
            return self.report("error", str(e))

    def delete_cluster(self, cluster_name: str) -> dict:
        try:
            self.eks.delete_cluster(name=cluster_name)
            return self.report("deleted", f"EKS cluster '{cluster_name}' deletion initiated")
        except Exception as e:
            return self.report("error", str(e))

    def list_supported_versions(self) -> dict:
        try:
            resp = self.eks.describe_addon_versions()
            versions = sorted(set(
                v["clusterVersion"]
                for a in resp.get("addons", [])
                for av in a.get("addonVersions", [])
                for v in av.get("compatibilities", [])
                if "clusterVersion" in v
            ), reverse=True)
            if not versions:
                versions = ["1.32", "1.31", "1.30", "1.29", "1.28"]
            return self.report("success", f"Supported k8s versions: {versions}", {"versions": versions})
        except Exception as e:
            return self.report("error", str(e))

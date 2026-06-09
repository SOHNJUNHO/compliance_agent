# Deployment Guide — AWS (ECS Fargate + GPU EC2)

## Architecture overview

```
Internet
  │
  ▼
ALB (HTTPS, ACM cert)
  │   /healthz  /query
  ▼
ECS Fargate service: compliance-api   ← FastAPI + in-process reranker (CPU)
  │   (private subnet)
  ├─── SG-scoped HTTP → GPU EC2 :8000 (vLLM chat — Qwen3-8B)
  ├─── SG-scoped HTTP → GPU EC2 :8001 (vLLM embed — qwen3-embedding-0.6B)
  │
  └─── NAT egress → Qdrant Cloud (vector DB, external SaaS)
                  → Langfuse Cloud  (observability, external SaaS)
```

**Component decisions:**

| Component | Choice | Rationale |
|---|---|---|
| LLM + embeddings | vLLM on `g5.xlarge` (A10G 24 GB) | Qwen3-8B fits; OpenAI-compatible API; no cold start |
| App serving | ECS Fargate + ALB | CPU-only (reranker in-process); no GPU infra to manage |
| Vector DB | Qdrant Cloud (managed SaaS) | Already supported via env; zero ops |
| Observability | Langfuse Cloud (managed SaaS) | Already supported via env; zero ops |
| Secrets | AWS Secrets Manager | `QDRANT_API_KEY`, `LANGFUSE_SECRET_KEY` |
| Logs | CloudWatch Logs | ECS awslogs driver |

---

## Prerequisites

- AWS account with ECR, ECS, EC2, ALB, Secrets Manager, CloudWatch permissions.
- Docker + AWS CLI configured locally.
- A VPC with at least 2 AZs; public subnets for ALB + NAT GW; private subnets for Fargate + EC2.
- An ACM certificate for your domain (can be a wildcard).
- Qdrant Cloud cluster and API key.
- Langfuse Cloud account and keys.

---

## Step 1 — Launch the vLLM GPU EC2 instance

See **`deploy/vllm.md`** for the full runbook (instance sizing, docker run commands, verification).

After completing that runbook, note the **private IP** of the EC2 instance.

---

## Step 2 — Store secrets in AWS Secrets Manager

```bash
# Qdrant API key
aws secretsmanager create-secret \
  --name compliance/qdrant-api-key \
  --secret-string '{"QDRANT_API_KEY":"<your-key>"}'

# Langfuse secret key
aws secretsmanager create-secret \
  --name compliance/langfuse-secret-key \
  --secret-string '{"LANGFUSE_SECRET_KEY":"<your-key>"}'
```

---

## Step 3 — Build and push the Docker image to ECR

```bash
REGION=ap-northeast-2
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
ECR_REPO="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/compliance-api"

# Create ECR repo (once)
aws ecr create-repository --repository-name compliance-api --region $REGION

# Authenticate
aws ecr get-login-password --region $REGION | \
  docker login --username AWS --password-stdin "${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"

# Build (reranker model baked in — takes ~5 min on first build)
docker build -t compliance-api .

# Tag and push
docker tag compliance-api:latest "${ECR_REPO}:latest"
docker push "${ECR_REPO}:latest"
```

> **Image size:** expect ~6–8 GB (Python base + deps + reranker model).
> After the first push, subsequent pushes are incremental (only changed layers).

---

## Step 4 — Create the ECS cluster and task definition

### 4a. ECS cluster

```bash
aws ecs create-cluster --cluster-name compliance --region $REGION
```

### 4b. Task definition (save as `deploy/task-definition.json`)

Key settings:
- `cpu`: 2048 (2 vCPU), `memory`: 8192 (8 GB) — reranker needs ~2.5 GB; leave headroom.
- `awslogs` for CloudWatch logs.
- Secrets Manager references for `QDRANT_API_KEY` and `LANGFUSE_SECRET_KEY`.

```json
{
  "family": "compliance-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::<account>:role/ecsTaskExecutionRole",
  "taskRoleArn":      "arn:aws:iam::<account>:role/ecsTaskRole",
  "containerDefinitions": [{
    "name": "compliance-api",
    "image": "<account>.dkr.ecr.ap-northeast-2.amazonaws.com/compliance-api:latest",
    "portMappings": [{"containerPort": 8000}],
    "essential": true,
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/compliance-api",
        "awslogs-region": "ap-northeast-2",
        "awslogs-stream-prefix": "ecs"
      }
    },
    "environment": [
      {"name": "LLM_API_BASE",         "value": "http://<EC2-private-ip>:8000/v1"},
      {"name": "LLM_MODEL",            "value": "Qwen3-8B"},
      {"name": "LLM_API_KEY",          "value": "EMPTY"},
      {"name": "EMBED_API_BASE",        "value": "http://<EC2-private-ip>:8001/v1"},
      {"name": "EMBED_MODEL",           "value": "qwen3-embedding-0.6B"},
      {"name": "EMBED_API_KEY",         "value": "EMPTY"},
      {"name": "QDRANT_URL",            "value": "https://<cluster>.qdrant.tech"},
      {"name": "QDRANT_COLLECTION",     "value": "compliance_agents"},
      {"name": "QDRANT_VECTOR_DIM",     "value": "1024"},
      {"name": "USE_QDRANT",            "value": "1"},
      {"name": "USE_RERANKER",          "value": "1"},
      {"name": "RERANKER_MODEL",        "value": "BAAI/bge-reranker-v2-m3"},
      {"name": "LANGFUSE_PUBLIC_KEY",   "value": "pk-lf-..."},
      {"name": "LANGFUSE_BASE_URL",     "value": "https://cloud.langfuse.com"}
    ],
    "secrets": [
      {
        "name": "QDRANT_API_KEY",
        "valueFrom": "arn:aws:secretsmanager:<region>:<account>:secret:compliance/qdrant-api-key:QDRANT_API_KEY::"
      },
      {
        "name": "LANGFUSE_SECRET_KEY",
        "valueFrom": "arn:aws:secretsmanager:<region>:<account>:secret:compliance/langfuse-secret-key:LANGFUSE_SECRET_KEY::"
      }
    ],
    "healthCheck": {
      "command": ["CMD-SHELL", "curl -f http://localhost:8000/healthz || exit 1"],
      "interval": 30,
      "timeout": 5,
      "retries": 3,
      "startPeriod": 120
    }
  }]
}
```

```bash
# Create CloudWatch log group
aws logs create-log-group --log-group-name /ecs/compliance-api --region $REGION

# Register task definition
aws ecs register-task-definition \
  --cli-input-json file://deploy/task-definition.json \
  --region $REGION
```

### 4c. ECS service

```bash
aws ecs create-service \
  --cluster compliance \
  --service-name compliance-api \
  --task-definition compliance-api \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={
    subnets=[<private-subnet-1>,<private-subnet-2>],
    securityGroups=[<fargate-sg>],
    assignPublicIp=DISABLED
  }" \
  --load-balancers "targetGroupArn=<tg-arn>,containerName=compliance-api,containerPort=8000" \
  --region $REGION
```

---

## Step 5 — Configure ALB

1. **Target group:** type `ip`, port 8000, health check path `/healthz`, healthy threshold 2.
2. **ALB listener:** HTTPS :443 → forward to the target group.
3. **HTTP redirect:** HTTP :80 → HTTPS :443 (optional but recommended).
4. **Certificate:** attach the ACM cert to the HTTPS listener.

---

## Step 6 — Security groups

| SG | Inbound | Outbound |
|---|---|---|
| `alb-sg` | 0.0.0.0/0 :443, :80 | Fargate SG :8000 |
| `fargate-sg` | ALB SG :8000 | GPU EC2 SG :8000/:8001, 0.0.0.0/0 :443 (Qdrant/Langfuse egress) |
| `gpu-ec2-sg` | Fargate SG :8000/:8001 | 0.0.0.0/0 :443 (HuggingFace download) |

---

## Step 7 — Smoke test

```bash
ALB_DNS=<your-alb-dns>

# Liveness
curl -s https://${ALB_DNS}/healthz
# → {"status":"ok"}

# Readiness
curl -s https://${ALB_DNS}/readyz
# → {"status":"ready"}

# Query
curl -s -X POST https://${ALB_DNS}/query \
  -H "Content-Type: application/json" \
  -d '{"query":"설명의무 위반으로 손해배상 책임이 인정된 판례가 있나요?","user_id":"test-001"}' \
  | jq '{cited_ids,cited_agents,agents_used}'
```

Expected: `cited_ids` non-empty, `cited_agents` subset of `agents_used`, trace visible in Langfuse.

---

## Autoscaling (recommended after initial deploy)

```bash
# Register scalable target
aws application-autoscaling register-scalable-target \
  --service-namespace ecs \
  --resource-id service/compliance/compliance-api \
  --scalable-dimension ecs:service:DesiredCount \
  --min-capacity 2 \
  --max-capacity 10

# Target tracking: scale at 70% average CPU
aws application-autoscaling put-scaling-policy \
  --policy-name cpu70 \
  --service-namespace ecs \
  --resource-id service/compliance/compliance-api \
  --scalable-dimension ecs:service:DesiredCount \
  --policy-type TargetTrackingScaling \
  --target-tracking-scaling-policy-configuration '{
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ECSServiceAverageCPUUtilization"
    }
  }'
```

---

## Updating the deployment

```bash
# Rebuild and push new image
docker build -t compliance-api . && \
docker tag  compliance-api:latest "${ECR_REPO}:latest" && \
docker push "${ECR_REPO}:latest"

# Force ECS to pull the new image (rolling update, zero downtime with desired-count ≥ 2)
aws ecs update-service \
  --cluster compliance \
  --service compliance-api \
  --force-new-deployment \
  --region $REGION
```

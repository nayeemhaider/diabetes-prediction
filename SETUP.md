# CI/CD Setup Guide

Complete instructions to get all four GitHub Actions workflows running.

---

## Step 1 — Push the project to GitHub

```bash
cd "F:\Portfolio Project\diabetes_analysis"

git init
git add .
git commit -m "feat: initial commit — diabetes prediction project"

# Create a new repo on github.com, then:
git remote add origin https://github.com/YOUR_USERNAME/diabetes-prediction.git
git branch -M main
git push -u origin main
```

---

## Step 2 — Add GitHub Secrets

Go to your repo on GitHub:
**Settings → Secrets and variables → Actions → New repository secret**

| Secret name          | Value                          | Where to get it |
|---|---|---|
| `DOCKERHUB_USERNAME` | Your Docker Hub username       | hub.docker.com → your profile |
| `DOCKERHUB_TOKEN`    | Docker Hub access token        | hub.docker.com → Account Settings → Security → New Access Token |

> **Never use your Docker Hub password** — always use an access token.

---

## Step 3 — Protect the main branch

Go to: **Settings → Branches → Add branch protection rule**

- Branch name pattern: `main`
- Check: **Require status checks to pass before merging**
- Add these required checks:
  - `Lint (Ruff)`
  - `Tests (pytest)`
  - `Docker build (verify)`
  - `Lint & format` (from PR checks)
  - `Security scan`
  - `Test coverage (min 70%)`
- Check: **Require a pull request before merging**
- Check: **Do not allow bypassing the above settings**

---

## Step 4 — Verify workflows appear

After pushing, go to your repo → **Actions** tab.
You should see four workflows listed:

| Workflow | File | Triggers |
|---|---|---|
| CI | `ci.yml` | Every push, every PR |
| CD | `cd.yml` | Push to main only |
| PR checks | `pr_checks.yml` | Every PR to main |
| Retrain model | `retrain.yml` | Sunday 02:00 UTC + manual |

---

## Step 5 — Trigger your first run

The CI workflow triggers automatically on push.
Watch it run: **Actions → CI → latest run**

To manually trigger the retrain workflow:
**Actions → Retrain model → Run workflow → Run workflow**

---

## How the full pipeline flows

```
Feature branch push
        │
        ▼
    CI workflow                   ← runs on every branch push
    lint → test → docker-build

        │ open PR to main
        ▼
    PR checks workflow            ← runs on every PR
    lint → security → coverage → summary comment

        │ PR merged to main
        ▼
    CD workflow                   ← runs on merge to main
    test gate → build → push to Docker Hub

        │ every Sunday 02:00 UTC (or manual)
        ▼
    Retrain workflow
    train → compare AUC → promote if improved → triggers CD
```

---

## Pulling and running the deployed image

Once CD has pushed the image to Docker Hub:

```bash
# Pull latest
docker pull YOUR_DOCKERHUB_USERNAME/diabetes-prediction-api:latest

# Run it
docker run -p 8000:8000 YOUR_DOCKERHUB_USERNAME/diabetes-prediction-api:latest

# Or with docker-compose (edit docker-compose.yml to use the Hub image)
docker-compose up
```

---

## Troubleshooting

**CI fails on `python train.py`**
The runner needs the dataset. Either commit `diabetes.csv` to the repo,
or add a step to download it before training:
```yaml
- name: Download dataset
  run: |
    # Option A: commit the CSV to the repo (simplest for a portfolio project)
    # Option B: download from a URL or GitHub release asset
```

**Docker push fails with "unauthorized"**
Check that `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are set correctly.
Tokens expire — regenerate at hub.docker.com → Account Settings → Security.

**Coverage gate fails**
Add more tests or lower `MIN_COVERAGE` in `pr_checks.yml`.
Current coverage focuses on `app/main.py` endpoints.

**Retrain doesn't promote**
The new model's AUC must beat the current one by at least 0.5%.
Override with `force_promote=true` in the manual dispatch form.

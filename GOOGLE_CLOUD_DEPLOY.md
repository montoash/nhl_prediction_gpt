# Google Cloud Deployment Guide

## 🚀 Deploy to Google Cloud Run (Recommended)

### Prerequisites
1. **Google Account**: Sign up at [cloud.google.com](https://cloud.google.com)
2. **Free Credits**: $300 free credits for new accounts
3. **Project**: Create a new Google Cloud project

### Method 1: One-Click Deploy (Easiest)

[![Run on Google Cloud](https://deploy.cloud.run/button.svg)](https://deploy.cloud.run/?git_repo=https://github.com/montoash/nfl-prediction-gpt.git&revision=main)

1. Click the button above
2. Sign in to Google Cloud
3. Select or create a project
4. Click "Deploy" - that's it!

### Method 2: Manual Deployment

#### Step 1: Setup Google Cloud CLI
```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

#### Step 2: Deploy to Cloud Run
```bash
# Clone your repo
git clone https://github.com/montoash/nfl-prediction-gpt.git
cd nfl-prediction-gpt

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Deploy (builds automatically)
gcloud run deploy nfl-prediction-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300s \
  --max-instances 10
```

#### Step 3: Get Your URL
After deployment, you'll get a URL like:
```
https://nfl-prediction-api-[hash]-uc.a.run.app
```

### Method 3: GitHub Integration (Auto-Deploy)

1. **Connect Repository**:
   - Go to [Google Cloud Console](https://console.cloud.google.com)
   - Navigate to Cloud Run
   - Click "Create Service" → "Continuously deploy from repository"
   - Connect your GitHub repo

2. **Configure Build**:
   - Build Type: Dockerfile
   - Dockerfile path: `Dockerfile.cloudrun`
   - Branch: `main`

3. **Service Settings**:
   - Memory: 2 GiB
   - CPU: 2
   - Timeout: 300 seconds
   - Allow unauthenticated invocations: ✅

## 📊 Cost Comparison

| Service | Free Tier | Memory | CPU | Cold Start | Monthly Cost |
|---------|-----------|--------|-----|------------|--------------|
| **Render Free** | ❌ 512MB limit | 512MB | 0.5 | 30-60s | $0 |
| **Google Cloud Run** | ✅ 2M requests | 2GB+ | 2+ vCPU | 1-3s | $0* |
| **Render Starter** | No limits | 512MB | 0.5 | None | $7 |

*Within free tier limits

## 🎯 Benefits of Google Cloud Run

### ✅ **Massive Memory Increase**
- **2GB+** vs Render's 512MB
- Can handle full NFL dataset loading
- No more memory optimization needed

### ✅ **Better Performance**
- **Faster cold starts** (1-3s vs 30-60s)
- **More CPU power** (2+ vCPUs vs 0.5)
- **Better concurrency** handling

### ✅ **Generous Free Tier**
- **2 million requests/month**
- **400,000 GB-seconds compute**
- **More than enough** for Custom GPT usage

### ✅ **Production Features**
- **Auto-scaling** from 0 to unlimited
- **Custom domains** included
- **SSL certificates** automatic
- **Global CDN** for faster responses
- **Monitoring** and logging built-in

## 🔄 Migration Steps

1. **Deploy to Cloud Run** using any method above
2. **Test the new URL** with your Custom GPT
3. **Update OpenAPI schema** with new Cloud Run URL
4. **Optional**: Keep Render as backup or delete it

Your NFL prediction API will run much better on Google Cloud Run! 🚀
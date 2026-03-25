# Deploy to Render - Complete Guide

Deploy your Diabetes Prediction API to the web in 5 minutes using Render.com

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Step 1: Prepare Your GitHub Repository](#step-1-prepare-your-github-repository)
- [Step 2: Sign Up on Render](#step-2-sign-up-on-render)
- [Step 3: Connect GitHub](#step-3-connect-github)
- [Step 4: Create Web Service](#step-4-create-web-service)
- [Step 5: Deploy](#step-5-deploy)
- [Step 6: Access Your API](#step-6-access-your-api)
- [Troubleshooting](#troubleshooting)
- [Maintenance](#maintenance)

---

## Prerequisites

✅ GitHub account (free at [github.com](https://github.com))
✅ Render account (free at [render.com](https://render.com))
✅ Your project with these files:
- `main.py`
- `index.html`
- `requirements_api.txt`
- `Dockerfile`
- `model/` (with trained model files)
- `.gitignore`
- `render.yaml` (optional, for easier setup)

---

## Step 1: Prepare Your GitHub Repository

### 1.1 Initialize Git (if not already done)

```bash
# Navigate to your project directory
cd C:\Users\Hp\VSCode_Files\AI_Engineering_Projects\Coding_Tests

# Initialize git repo
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Diabetes prediction API"
```

### 1.2 Create GitHub Repository

1. Go to [github.com/new](https://github.com/new)
2. Repository name: `diabetes-prediction-api`
3. Choose **Public** (easier for Render)
4. Click **Create repository**

### 1.3 Push Code to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/diabetes-prediction-api.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

✅ **Your code is now on GitHub!**

---

## Step 2: Sign Up on Render

1. Go to [render.com](https://render.com)
2. Click **Sign up**
3. Choose **Sign up with GitHub** (recommended)
4. Authorize Render to access your GitHub account
5. Complete your profile

✅ **You're ready to deploy!**

---

## Step 3: Connect GitHub

1. After signing up, you'll see the Render dashboard
2. Click **"Connect a repository"** or **"New +"**
3. Select **"Web Service"**
4. Click **"Connect repository"**
5. Search for `diabetes-prediction-api`
6. Click **"Connect"** next to your repo

✅ **Render can now access your code**

---

## Step 4: Create Web Service

After connecting your repo, fill in these settings:

| Setting | Value |
|---------|-------|
| **Name** | `diabetes-api` |
| **Environment** | `Docker` |
| **Region** | `Oregon` or closest to you |
| **Branch** | `main` |
| **Plan** | `Free` (or Starter for $7/mo if you want always-on) |

### Important Settings:

**Build Command** (leave empty - Docker handles it)
```
(leave blank)
```

**Start Command** (leave empty - Dockerfile handles it)
```
(leave blank)
```

**Advanced Settings → Health Check Path**
```
/health
```

✅ **Configuration complete**

---

## Step 5: Deploy

1. Click **"Create Web Service"** button
2. Render will start building your Docker image
3. Watch the build logs in real-time
4. Deployment takes **3-5 minutes**

### What's Happening:
- Render pulls your code from GitHub
- Builds Docker image from your Dockerfile
- Runs the container
- Allocates a public URL

📊 **Status indicator** will show:
- 🟡 **Building** - Creating Docker image
- 🟢 **Live** - Your API is online!

---

## Step 6: Access Your API

Once deployment is complete, you'll see a **public URL** like:

```
https://diabetes-api-xxxxx.onrender.com
```

### Use Your API:

**Visit the Web UI:**
```
https://diabetes-api-xxxxx.onrender.com
```
You'll see your beautiful diabetes prediction form!

**API Documentation:**
```
https://diabetes-api-xxxxx.onrender.com/docs
```
Interactive Swagger UI with "Try it out"

**Health Check:**
```bash
curl https://diabetes-api-xxxxx.onrender.com/health
```

**Make a Prediction (curl):**
```bash
curl -X POST https://diabetes-api-xxxxx.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "age": 45.0,
    "hypertension": 0,
    "heart_disease": 0,
    "smoking_history": "never",
    "bmi": 25.5,
    "HbA1c_level": 5.8,
    "blood_glucose_level": 120
  }'
```

✅ **Your API is live on the internet!**

---

## Auto-Deploy on Git Push

The `render.yaml` file configures automatic deployments:

### How It Works:

1. You make changes locally
2. Push to GitHub: `git push origin main`
3. Render automatically detects the push
4. Renders rebuilds and redeploys your API
5. New version is live in 2-3 minutes

**No manual steps needed!** 🚀

### Test Auto-Deploy:

1. Make a small change to `main.py`
2. Commit and push:
   ```bash
   git add .
   git commit -m "Update: test auto-deploy"
   git push origin main
   ```
3. Go to Render dashboard
4. Watch it rebuild automatically
5. Your changes are live within 2-3 minutes!

---

## Troubleshooting

### Build Fails

**Check logs:**
1. Go to Render dashboard
2. Click your service name
3. Click **"Logs"** tab
4. Look for error messages

**Common issues:**
- **Missing files**: Check `.gitignore` - make sure you're not excluding important files
- **Models too large**: If `models/` folder is large (>100MB), it may timeout
- **Port mismatch**: Ensure Dockerfile uses port 8000

### API Slow to Respond

**Free tier behavior:**
- Free tier goes to sleep after 15 minutes of inactivity
- First request after sleep takes 30-60 seconds
- Subsequent requests are fast

**Solution**: Upgrade to Starter or Pro plan for always-on

### "Service not found" Error

- Wait 5-10 minutes for deployment to fully initialize
- Refresh the page
- Check logs for build errors
- Try Render dashboard → "Manual Deploy"

### Upload Files Not Working

Ensure `index.html` is being copied in Dockerfile:
```dockerfile
COPY index.html .
```

---

## Maintenance

### Monitor Your Deployment

**Dashboard:**
- CPU/Memory usage
- Build history
- Deployment status
- Error logs

**View logs anytime:**
1. Render dashboard
2. Click your service
3. Select **"Logs"** tab
4. Filter by date/time

### Update Your API

**To deploy new changes:**

```bash
# Make changes locally
# ... edit code ...

# Stage changes
git add .

# Commit
git commit -m "Update: describe your changes"

# Push to GitHub
git push origin main

# Render automatically rebuilds!
```

### Upgrade Plan (Optional)

**Free tier limitations:**
- Sleeps after 15 minutes of inactivity
- 0.5 CPU / 512 MB RAM
- 100GB bandwidth/month

**Starter plan ($7/month):**
- Always on (no sleeping)
- 0.5 CPU / 1 GB RAM
- Unlimited bandwidth

**Upgrade:**
1. Render dashboard → Your service
2. Click **"Settings"**
3. Scroll to **"Plan"**
4. Click **"Upgrade"**

---

## Performance Tips

### Reduce Model Size

If deployment is slow, compress your models:

```python
# In train_and_save_model.py
joblib.dump(model, 'models/xgboost_model.pkl', compress=3)
```

### Add Caching

For frequently requested predictions, add Redis:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cache_prediction(patient_hash):
    # Your prediction logic
    pass
```

### Monitor Performance

On Render dashboard:
- Check CPU usage
- Monitor memory
- Review response times in logs

---

## Cost Breakdown

### Free Tier
- **Cost**: $0
- **Limitations**: Sleeps after 15 min inactivity
- **Best for**: Development & testing

### Starter Plan
- **Cost**: $7/month
- **Benefits**: Always on, better resources
- **Best for**: Production use

### Scale as Needed

Add more services for:
- Database (PostgreSQL)
- Background jobs (Celery)
- Additional APIs

---

## What's Next?

After deployment:

1. **Add Authentication** - Protect your API with API keys
2. **Enable Custom Domain** - Use your own domain name (diabetes-api.com)
3. **Set Up Monitoring** - Get alerts if API goes down
4. **Add Database** - Store prediction history
5. **Implement Caching** - Improve response times

---

## Useful Links

- [Render Documentation](https://render.com/docs)
- [Render GitHub Integration](https://render.com/docs/github)
- [Render CLI](https://render.com/docs/cli)
- [FastAPI on Render](https://render.com/docs/deploy-fastapi)
- [Docker on Render](https://render.com/docs/docker)

---

## Support

### If Something Goes Wrong:

1. **Check logs**: Render dashboard → Logs tab
2. **Read error messages**: They're usually descriptive
3. **Manual deploy**: Click "Manual Deploy" to rebuild
4. **Restart**: Click "Restart" to restart the service
5. **Render Support**: [render.com/support](https://render.com/support)

---

## Summary

✅ Your Diabetes Prediction API is now live on the internet!
✅ Anyone can visit your URL and make predictions
✅ Auto-deploys when you push to GitHub
✅ Free tier available
✅ Easy to upgrade or scale

**Your API is accessible at:**
```
https://diabetes-api-xxxxx.onrender.com
```

Share this link with anyone to let them predict diabetes risk! 🎉

---

**Last Updated**: March 25, 2026  
**Deployment Platform**: Render.com  
**Status**: Production Ready

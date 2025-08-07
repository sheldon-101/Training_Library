import 'dotenv/config';
import express from 'express';
import fs from 'fs';
import cors from 'cors';
import { generateEmbeddings, isCacheValid, CACHE_FILE } from './generateEmbeddings.mjs';

const app = express();
app.use(cors());
app.use(express.json());

const PORT = process.env.PORT || 3001;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;

let embeddedData = [];
const test = 'test'
function cosineSimilarity(a, b) {
  const dot = a.reduce((sum, ai, i) => sum + ai * b[i], 0);
  const normA = Math.sqrt(a.reduce((sum, ai) => sum + ai * ai, 0));
  const normB = Math.sqrt(b.reduce((sum, bi) => sum + bi * bi, 0));
  return dot / (normA * normB);
}

async function getQueryEmbedding(text) {
  const response = await fetch('https://api.openai.com/v1/embeddings', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${OPENAI_API_KEY}`
    },
    body: JSON.stringify({
      input: text,
      model: 'text-embedding-3-small'
    })
  });

  const data = await response.json();
  return data.data[0].embedding;
}

// Load embeddings data
async function loadEmbeddedData() {
  try {
    if (fs.existsSync(CACHE_FILE)) {
      console.log('Loading cached embeddings...');
      embeddedData = JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
      console.log(`Loaded ${embeddedData.length} cached embeddings.`);
    } else {
      console.log('No cached embeddings found. Generating initial embeddings...');
      embeddedData = await generateEmbeddings();
    }
  } catch (error) {
    console.error('Failed to load embeddings:', error.message);
    throw error;
  }
}

// Check for daily refresh
function scheduleRefresh() {
  const now = new Date();
  const tomorrow = new Date(now);
  tomorrow.setDate(tomorrow.getDate() + 1);
  tomorrow.setHours(0, 0, 0, 0); // Midnight
  
  const msUntilMidnight = tomorrow - now;
  
  setTimeout(async () => {
    console.log('Starting daily embedding refresh...');
    try {
      embeddedData = await generateEmbeddings(true);
      console.log('Daily refresh completed successfully.');
    } catch (error) {
      console.error('Daily refresh failed:', error.message);
    }
    
    // Schedule next refresh in 24 hours
    scheduleRefresh();
  }, msUntilMidnight);
  
  console.log(`Next embedding refresh scheduled for ${tomorrow.toISOString()}`);
}

app.post('/search', async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).json({ error: 'Missing query' });

  try {
    // Ensure we have data loaded
    if (embeddedData.length === 0) {
      return res.status(503).json({ error: 'Embeddings not yet loaded' });
    }

    const queryEmbedding = await getQueryEmbedding(query);

    const ranked = embeddedData
      .map(item => ({
        ...item,
        score: cosineSimilarity(queryEmbedding, item.embedding)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, 6);

    res.json(ranked);
  } catch (err) {
    console.error('Search error:', err.message);
    res.status(500).json({ error: 'Failed to process query' });
  }
});

// Manual refresh endpoint
app.post('/refresh', async (req, res) => {
  try {
    console.log('Manual refresh requested...');
    embeddedData = await generateEmbeddings(true);
    res.json({ 
      success: true, 
      message: `Refreshed ${embeddedData.length} embeddings`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error('Manual refresh failed:', error.message);
    res.status(500).json({ error: 'Refresh failed' });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    embeddingsLoaded: embeddedData.length > 0,
    cacheValid: isCacheValid(),
    timestamp: new Date().toISOString()
  });
});

// Initialize server
async function startServer() {
  try {
    // Load embeddings on startup
    await loadEmbeddedData();
    
    // Schedule daily refresh
    scheduleRefresh();
    
    // Start the server
    app.listen(PORT, () => {
      console.log(`🚀 Semantic search API running at http://localhost:${PORT}`);
      console.log(`📊 Loaded ${embeddedData.length} training resources`);
      console.log(`⏰ Daily refresh scheduled for midnight`);
      console.log(`🔍 Available endpoints:`);
      console.log(`  POST /search - Search training materials`);
      console.log(`  POST /refresh - Manual cache refresh`);
      console.log(`  GET /health - Health check`);
    });
  } catch (error) {
    console.error('Failed to start server:', error.message);
    process.exit(1);
  }
}

startServer();

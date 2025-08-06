import 'dotenv/config';
import fs from 'fs';
import path from 'path';

const OPENAI_KEY = process.env.OPENAI_API_KEY;
const RESOURCE_API = 'https://teamavalonpontoons.com/api/traininglibrary.php';
const CACHE_FILE = 'embedded-resources.json';
const CACHE_META_FILE = 'cache-metadata.json';
const CACHE_DURATION_HOURS = 24;

function getCacheMetadata() {
  try {
    if (fs.existsSync(CACHE_META_FILE)) {
      return JSON.parse(fs.readFileSync(CACHE_META_FILE, 'utf8'));
    }
  } catch (error) {
    console.warn('Failed to read cache metadata:', error.message);
  }
  return null;
}

function setCacheMetadata(metadata) {
  try {
    fs.writeFileSync(CACHE_META_FILE, JSON.stringify(metadata, null, 2));
  } catch (error) {
    console.error('Failed to write cache metadata:', error.message);
  }
}

function isCacheValid() {
  const metadata = getCacheMetadata();
  if (!metadata || !metadata.lastUpdated) return false;
  
  const lastUpdated = new Date(metadata.lastUpdated);
  const now = new Date();
  const hoursSinceUpdate = (now - lastUpdated) / (1000 * 60 * 60);
  
  return hoursSinceUpdate < CACHE_DURATION_HOURS && fs.existsSync(CACHE_FILE);
}

async function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

async function getEmbedding(text, maxRetries = 5) {
  if (!OPENAI_KEY) {
    throw new Error('OPENAI_API_KEY environment variable is required');
  }
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      const response = await fetch("https://api.openai.com/v1/embeddings", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${OPENAI_KEY}`
        },
        body: JSON.stringify({
          model: "text-embedding-3-small",
          input: text
        })
      });

      if (!response.ok) {
        const errorText = await response.text().catch(() => 'Unknown error');
        
        // Retry on server errors (5xx) and rate limits (429)
        if (response.status >= 500 || response.status === 429) {
          const delay = Math.min(1000 * Math.pow(2, attempt - 1), 30000); // Cap at 30 seconds
          console.warn(`OpenAI API error ${response.status}, attempt ${attempt}/${maxRetries}. Retrying in ${delay}ms...`);
          
          if (attempt < maxRetries) {
            await sleep(delay);
            continue;
          }
        }
        
        throw new Error(`OpenAI API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const result = await response.json();
      return result.data[0].embedding;
      
    } catch (error) {
      if (attempt === maxRetries) {
        throw error;
      }
      
      // Retry on network errors
      const delay = Math.min(1000 * Math.pow(2, attempt - 1), 30000);
      console.warn(`Network error on attempt ${attempt}/${maxRetries}: ${error.message}. Retrying in ${delay}ms...`);
      await sleep(delay);
    }
  }
}

async function generateEmbeddings(forceRefresh = false, resumeFromPartial = false) {
  try {
    // Check if cache is valid and not forcing refresh
    if (!forceRefresh && !resumeFromPartial && isCacheValid()) {
      console.log('Cache is still valid. Skipping embedding generation.');
      return JSON.parse(fs.readFileSync(CACHE_FILE, 'utf8'));
    }

    console.log('Fetching training library data...');
    const res = await fetch(RESOURCE_API);
    
    if (!res.ok) {
      throw new Error(`Failed to fetch training data: ${res.status} ${res.statusText}`);
    }
    
    const items = await res.json();
    console.log(`Processing ${items.length} items...`);
    
    // Check for partial file to resume from
    const partialFile = `${CACHE_FILE}.partial`;
    let processedItems = [];
    let startIndex = 0;
    
    if (resumeFromPartial && fs.existsSync(partialFile)) {
      try {
        processedItems = JSON.parse(fs.readFileSync(partialFile, 'utf8'));
        startIndex = processedItems.length;
        console.log(`🔄 Resuming from partial file: ${startIndex} items already processed`);
      } catch (error) {
        console.warn('Failed to read partial file, starting fresh:', error.message);
        processedItems = [];
        startIndex = 0;
      }
    }

    // Generate embeddings with progressive rate limiting  
    let consecutiveErrors = 0;
    
    for (let i = startIndex; i < items.length; i++) {
      const item = items[i];
      const combinedText = `${item.Title} ${item.Topic} ${item.Description}`;
      
      try {
        item.embedding = await getEmbedding(combinedText);
        processedItems.push(item);
        consecutiveErrors = 0;
        console.log(`✅ Processed item ${i + 1}/${items.length}: ${item.Title}`);
        
        // Progressive rate limiting: increase delay after errors
        const baseDelay = 200; // Increased from 100ms
        const errorMultiplier = Math.min(consecutiveErrors * 500, 5000);
        const delay = baseDelay + errorMultiplier;
        
        if (i < items.length - 1) {
          await sleep(delay);
        }
        
        // Save progress every 25 items
        if ((i + 1) % 25 === 0) {
          const partialFile = `${CACHE_FILE}.partial`;
          fs.writeFileSync(partialFile, JSON.stringify(processedItems, null, 2));
          console.log(`💾 Progress saved: ${i + 1}/${items.length} items`);
        }
        
      } catch (error) {
        consecutiveErrors++;
        console.error(`❌ Failed to generate embedding for item ${i + 1}: ${item.Title}`);
        console.error(`Error: ${error.message}`);
        
        // Save partial results before failing
        if (processedItems.length > 0) {
          const partialFile = `${CACHE_FILE}.partial`;
          fs.writeFileSync(partialFile, JSON.stringify(processedItems, null, 2));
          console.log(`💾 Partial results saved (${processedItems.length} items) before failure`);
        }
        
        throw new Error(`Failed after processing ${processedItems.length}/${items.length} items: ${error.message}`);
      }
    }
    
    // Clean up partial file on success
    if (fs.existsSync(partialFile)) {
      fs.unlinkSync(partialFile);
    }

    // Save to cache
    fs.writeFileSync(CACHE_FILE, JSON.stringify(processedItems, null, 2));
    
    // Update cache metadata
    setCacheMetadata({
      lastUpdated: new Date().toISOString(),
      itemCount: items.length,
      version: '1.0'
    });

    console.log(`✅ All ${processedItems.length} embeddings generated and cached.`);
    return processedItems;
    
  } catch (error) {
    console.error('Error generating embeddings:', error.message);
    throw error;
  }
}

// Export for use by server
export { generateEmbeddings, isCacheValid, CACHE_FILE };

// Run if called directly
if (import.meta.url === `file://${process.argv[1]}`) {
  const forceRefresh = process.argv.includes('--force');
  const resumeFromPartial = process.argv.includes('--resume');
  
  if (resumeFromPartial) {
    console.log('🔄 Attempting to resume from partial file...');
  }
  
  generateEmbeddings(forceRefresh, resumeFromPartial).catch(console.error);
}

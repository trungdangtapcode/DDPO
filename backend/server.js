const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

// Middleware
app.use(cors());
app.use(express.json());

// Health check
app.get('/api/health', async (req, res) => {
    try {
        const response = await axios.get(`${PYTHON_API_URL}/health`);
        res.json({ 
            status: 'healthy', 
            backend: 'ok',
            pythonApi: response.data 
        });
    } catch (error) {
        res.status(503).json({ 
            status: 'unhealthy', 
            backend: 'ok',
            pythonApi: 'unreachable',
            error: error.message 
        });
    }
});

// Proxy endpoint for image generation with streaming
app.get('/api/generate', async (req, res) => {
    const { prompt, steps } = req.query;

    if (!prompt) {
        return res.status(400).json({ error: 'Prompt is required' });
    }

    try {
        // Set headers for Server-Sent Events
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('Access-Control-Allow-Origin', '*');

        // Make request to Python API
        const response = await axios.get(`${PYTHON_API_URL}/generate`, {
            params: { prompt, steps: steps || 20 },
            responseType: 'stream'
        });

        // Forward the stream to the client
        response.data.on('data', (chunk) => {
            res.write(chunk);
        });

        response.data.on('end', () => {
            res.end();
        });

        response.data.on('error', (error) => {
            console.error('Stream error:', error);
            res.end();
        });

        // Handle client disconnect
        req.on('close', () => {
            response.data.destroy();
        });

    } catch (error) {
        console.error('Error proxying to Python API:', error.message);
        
        if (!res.headersSent) {
            res.status(500).json({ 
                error: 'Failed to connect to Python API',
                details: error.message 
            });
        } else {
            res.end();
        }
    }
});

// Root endpoint
app.get('/', (req, res) => {
    res.json({
        message: 'Mock Diffusion Backend API',
        endpoints: {
            health: '/api/health',
            generate: '/api/generate?prompt=your_prompt&steps=20'
        }
    });
});

app.listen(PORT, () => {
    console.log(`Node.js backend server running on http://localhost:${PORT}`);
    console.log(`Proxying to Python API at ${PYTHON_API_URL}`);
});

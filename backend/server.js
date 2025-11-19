const express = require('express');
const cors = require('cors');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3001;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increase limit for base64 images
app.use(express.urlencoded({ limit: '50mb', extended: true }));

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

// Proxy endpoint for image generation with streaming (POST for large payloads)
app.post('/api/generate', async (req, res) => {
    const requestBody = req.body;

    // SHOW SOME PAYLOAD
    // console.log('=== Incoming Generate Request ===');
    // console.log('Prompt:', requestBody.prompt || '(empty)');
    // console.log('Prompt Image:', requestBody.prompt_image ? 'Yes (base64)' : 'No');
    // console.log('Init Image:', requestBody.init_image ? 'Yes (base64)' : 'No');
    // console.log('Model:', requestBody.model);
    // console.log('Steps:', requestBody.steps);
    // console.log('================================');

    // SHOW FULL PAYLOAD
    console.log('=== Full Request Body ===');
    console.log(requestBody);
    console.log('=========================');

    // Allow empty prompt if prompt_image is provided
    if (!requestBody.prompt && !requestBody.prompt_image) {
        console.error('❌ Validation error: Neither prompt nor prompt_image provided');
        return res.status(400).json({ error: 'Either prompt or prompt_image is required' });
    }

    try {
        // Set headers for Server-Sent Events
        res.setHeader('Content-Type', 'text/event-stream');
        res.setHeader('Cache-Control', 'no-cache');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('Access-Control-Allow-Origin', '*');

        console.log('🔄 Forwarding request to Python API...');

        // Forward the entire request body to Python API
        const response = await axios.post(`${PYTHON_API_URL}/generate`, requestBody, {
            responseType: 'stream',
            headers: {
                'Content-Type': 'application/json'
            }
        });

        console.log('✓ Connected to Python API stream');

        // Forward the stream to the client
        response.data.on('data', (chunk) => {
            res.write(chunk);
        });

        response.data.on('end', () => {
            console.log('✓ Stream completed successfully');
            res.end();
        });

        response.data.on('error', (error) => {
            console.error('❌ Stream error:', error);
            res.end();
        });

        // Handle client disconnect
        req.on('close', () => {
            console.log('⚠ Client disconnected, destroying stream');
            response.data.destroy();
        });

    } catch (error) {
        console.error('❌ Error proxying to Python API:', error.message);
        if (error.response) {
            console.error('Python API response status:', error.response.status);
            console.error('Python API response data:', error.response.data);
        }
        
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

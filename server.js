require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const tmImage = require('@teachablemachine/image');

const app = express();
app.use(cors());
app.use(express.json());

// Configure multer for in-memory file upload
const upload = multer({ storage: multer.memoryStorage() });

let model;

// Load the Teachable Machine model
async function loadModel() {
  try {
    const modelUrl = process.env.MODEL_URL || 'https://teachablemachine.withgoogle.com/models/rtPCnZfSI/';
    model = await tmImage.load(`${modelUrl}model.json`, `${modelUrl}metadata.json`);
    console.log(' Model loaded successfully');
  } catch (err) {
    console.error(' Error loading model:', err);
  }
}
loadModel();

// Health check endpoint
app.get('/', (req, res) => {
  res.status(200).json({ status: 'Alive', model: model ? 'Loaded' : 'Loading' });
});

// Prediction endpoint
app.post('/api/predict', upload.single('image'), async (req, res) => {
  try {
    console.log(' POST /api/predict triggered');

    if (!req.file) {
      console.log('No image uploaded');
      return res.status(400).json({ error: 'No image uploaded' });
    }

    if (!model) {
      console.log(' Model not loaded');
      return res.status(503).json({ error: 'Model not loaded yet' });
    }

    let image;
    try {
      image = tf.node.decodeImage(req.file.buffer);
      console.log(' Image decoded');
    } catch (decodeErr) {
      console.error('❌ Failed to decode image:', decodeErr);
      return res.status(400).json({ error: 'Invalid image format' });
    }

    try {
      const predictions = await model.predict(image);
      tf.dispose(image);

      const result = predictions
        .map(p => ({
          className: p.className,
          probability: p.probability.toFixed(4)
        }))
        .sort((a, b) => b.probability - a.probability);

      console.log(' Prediction done:', result[0]);

      res.json({
        success: true,
        topPrediction: result[0],
        allPredictions: result
      });
    } catch (predictErr) {
      console.error(' Prediction error:', predictErr);
      return res.status(500).json({ error: 'Failed during prediction' });
    }
  } catch (err) {
    console.error(' Unexpected error:', err);
    return res.status(500).json({ error: err.message });
  }
});

// Start the server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(` Server running on port ${PORT}`);
});

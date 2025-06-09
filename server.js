require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const tmImage = require('@teachablemachine/image');

const app = express();
app.use(cors());
app.use(express.json());

// Configure upload
const upload = multer({ storage: multer.memoryStorage() });

// Load Teachable Machine model
let model;
async function loadModel() {
  const modelUrl = process.env.MODEL_URL || 'https://teachablemachine.withgoogle.com/models/rtPCnZfSI/';
  model = await tmImage.load(
    `${modelUrl}model.json`,
    `${modelUrl}metadata.json`
  );
  console.log(' Model loaded successfully');
}
loadModel();

// Health check endpoint
app.get('/', (req, res) => {
  res.status(200).json({ status: 'Alive', model: model ? 'Loaded' : 'Loading' });
});

// Prediction endpoint
app.post('/api/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded' });
    }

    console.log('📸 Processing image...');
    const image = tf.node.decodeImage(req.file.buffer);
    const predictions = await model.predict(image);
    tf.dispose(image); // Clean up TensorFlow memory

    // Format response
    const result = predictions.map(p => ({
      className: p.className,
      probability: p.probability.toFixed(4)
    })).sort((a, b) => b.probability - a.probability);

    res.json({
      success: true,
      topPrediction: result[0],
      allPredictions: result
    });

  } catch (error) {
    console.error(' Prediction error:', error);
    res.status(500).json({ 
      success: false,
      error: error.message 
    });
  }
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(` API running on port ${PORT}`);
});
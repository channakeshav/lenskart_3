require('dotenv').config();
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const tmImage = require('@teachablemachine/image');
const { createCanvas, Image } = require('canvas');

const app = express();
app.use(cors());
app.use(express.json());

// Multer config (upload to memory)
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 } // 5MB max
});

// Load model
let model;
async function loadModel() {
  try {
    const modelUrl = process.env.MODEL_URL || 'https://teachablemachine.withgoogle.com/models/rtPCnZfSI/';
    model = await tmImage.load(`${modelUrl}model.json`, `${modelUrl}metadata.json`);
    console.log('✅ Model loaded successfully');
  } catch (err) {
    console.error('❌ Error loading model:', err.message);
  }
}
loadModel();

// Health check
app.get('/', (req, res) => {
  res.status(200).json({ status: 'Alive', model: model ? 'Loaded' : 'Loading' });
});

// Prediction route
app.post('/api/predict', upload.single('image'), async (req, res) => {
  console.log('📥 POST /api/predict received');

  if (!req.file) {
    return res.status(400).json({ error: 'No image uploaded' });
  }

  if (!model) {
    return res.status(503).json({ error: 'Model not loaded yet' });
  }

  try {
    // Resize image using canvas
    const width = 224, height = 224; // TM input size
    const canvas = createCanvas(width, height);
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = req.file.buffer;
    ctx.drawImage(img, 0, 0, width, height);
    const imageData = ctx.getImageData(0, 0, width, height);

    // Convert to tensor
    const tensor = tf.browser.fromPixels(imageData);

    // Predict
    const predictions = await model.predict(tensor);

    const result = predictions.map(p => ({
      className: p.className,
      probability: p.probability.toFixed(4)
    })).sort((a, b) => b.probability - a.probability);

    res.json({
      success: true,
      topPrediction: result[0],
      allPredictions: result
    });

  } catch (err) {
    console.error('❌ Prediction error:', err.message);
    console.error(err.stack);
    res.status(500).json({ success: false, error: err.message, stack: err.stack });
  }
});

// Start server
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`🚀 Server running on port ${PORT}`);
});

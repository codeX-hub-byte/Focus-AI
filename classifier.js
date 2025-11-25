let classifier = null;
// Must match the expected video resolution used for training/normalization
const VIDEO_WIDTH = 640;
const VIDEO_HEIGHT = 480;

/**
 * Loads the pre-trained Keras/TensorFlow model for state prediction.
 * Requires 'model.json' and 'weights.bin' to be available.
 * * The model expects an input vector of 62 elements:
 * (30 keypoints from FaceMesh * 2 coords) + (2 gaze coords) = 62
 */
async function loadClassifier() {
    try {
        // Assume the model is saved via tf.js 'save' function:
        classifier = await tf.loadLayersModel('model.json');
        
        // Example sequential model creation (if loading from scratch):
        // classifier = tf.sequential();
        // classifier.add(tf.layers.dense({units: 32, inputShape: [62], activation: "relu"}));
        // classifier.add(tf.layers.dense({units: 16, activation: "relu"}));
        // classifier.add(tf.layers.dense({units: 4, activation: "softmax"}));
        // await classifier.load(await modelJson.json(), weights);


    } catch (e) {
        console.error("Failed to load classifier model. Ensure model.json and weights.bin are present.", e);
        throw new Error("Classifier model failed to load.");
    }
}

/**
 * Predicts the student's current state (Focused, Writing, Looking Away, Sleeping)
 * @param {Array} keypoints - The face landmark keypoints (all 468 from FaceMesh).
 * @param {object} gaze - The {dx, dy} gaze vector.
 * @returns {string} The predicted state label.
 */
async function predictState(keypoints, gaze) {
    // 1. Prepare input array (Select a subset of 30 keypoints + gaze vector)
    const arr = [];
    
    // Select 30 representative keypoints (e.g., forehead, cheeks, chin, corners of eyes/mouth, etc.)
    const indicesToUse = [
        10, 33, 61, 78, 81, 82, 84, 87, 88, 91, 146, 178, 181, 185, 248, 273, 291, 308, 311, 314, 317, 318, 320, 345, 402, 405, 412, 420, 422, 424
    ];

    indicesToUse.forEach(i => {
        const k = keypoints[i];
        // Normalize keypoints by video size
        arr.push(k.x / VIDEO_WIDTH);
        arr.push(k.y / VIDEO_HEIGHT);
    });

    // Add gaze vector (normalized in relation to face size if necessary, but using raw here is simpler)
    arr.push(gaze.dx);
    arr.push(gaze.dy);

    // 2. Create Tensor and Predict
    const t = tf.tensor([arr]);
    const out = classifier.predict(t);

    // 3. Get the class index with the highest probability
    const idx = out.argMax(-1).dataSync()[0];
    
    // 4. Map index to state label (Order must match training data classes)
    return ["Focused", "Writing", "Looking Away", "Sleeping"][idx];
}
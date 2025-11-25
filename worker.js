// Load necessary scripts and libraries within the worker context
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow/tfjs");
importScripts("https://cdn.jsdelivr.net/npm/@tensorflow-models/face-landmarks-detection");
importScripts("https://cdn.jsdelivr.net/npm/face-api.js");

importScripts("gaze.js");
importScripts("classifier.js");
importScripts("face-recognition.js");

let facemeshModel;
let recognitionLoaded = false;

self.onmessage = async (msg) => {
    // 1. Initialization: Load all ML models
    if (msg.data.type === "init") {
        try {
            self.postMessage({type: "status", message: "Loading FaceMesh model..."});
            // Use MediaPipeFacemesh for fast detection and iris landmarks
            facemeshModel = await faceLandmarksDetection.load(
                faceLandmarksDetection.SupportedPackages.mediapipeFacemesh
            );
            
            self.postMessage({type: "status", message: "Loading State Classifier..."});
            await loadClassifier();
            
            self.postMessage({type: "status", message: "Loading Identity Recognizer..."});
            await loadFaceRecognition();
            recognitionLoaded = true; // Set flag after identity is ready
            
            self.postMessage({type: "ready"}); 
        } catch (error) {
             console.error("Worker initialization failed:", error);
             self.postMessage({type: "error", message: error.message || "Failed to load ML models."});
        }
        return;
    }

    // 2. Frame Processing: Run detection on incoming ImageBitmap
    if (msg.data.type === "frame" && recognitionLoaded) {
        const image = msg.data.bitmap;
        const result = [];

        // Use tf.tidy for memory management of intermediate tensors
        tf.tidy(async () => {
            // Estimate faces (landmarks)
            const faces = await facemeshModel.estimateFaces({input: image});
            
            for (let f of faces) {
                const kp = f.keypoints;

                // Extract landmarks needed for Gaze and State prediction
                const iris = kp.find(k => k.name === "irisCenter");
                const leftEye = kp.find(k => k.name === "leftEyeCenter");
                const rightEye = kp.find(k => k.name === "rightEyeCenter");
                
                if (!iris || !leftEye || !rightEye || kp.length < 468) continue; // Basic sanity check

                // 1. Gaze Tracking
                const gaze = computeGaze(iris, leftEye, rightEye);
                
                // 2. State Classification (Neural Net)
                const state = await predictState(kp, gaze);
                
                // 3. True Identity Recognition
                // NOTE: Passing the full ImageBitmap to identifyFace() inside the loop is CPU intensive.
                // In a real production app, this would be sampled every N frames.
                const name = await identifyFace(image); 

                // Collect results for transfer back to main thread
                result.push({
                    x: f.box.xMin + f.box.width/2, // Center X
                    y: f.box.yMin + f.box.height/2, // Center Y
                    box: f.box, // Bounding box for drawing
                    state,
                    name 
                });
            }
        });
        
        // Send the final, processed detection results back
        self.postMessage({type: "data", detections: result});
        image.close(); // Clean up ImageBitmap memory
    }
};
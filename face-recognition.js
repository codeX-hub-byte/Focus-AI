let faceMatcher = null;
let recognitionLoaded = false;

/**
 * Loads Face-API.js models and creates a FaceMatcher database from student images.
 * * NOTE: This function needs access to the /models/ directory containing
 * the tinyFaceDetector, faceRecognitionNet, and faceLandmark68Net weights.
 */
async function loadFaceRecognition() {
    try {
        await faceapi.nets.tinyFaceDetector.load('/models/');
        await faceapi.nets.faceRecognitionNet.load('/models/');
        await faceapi.nets.faceLandmark68Net.load('/models/');

        const students = await (await fetch("students.json")).json();
        let descriptors = [];

        for (let stu of students) {
            const img = await faceapi.fetchImage(stu.url);
            const det = await faceapi
                .detectSingleFace(img, new faceapi.TinyFaceDetectorOptions())
                .withFaceLandmarks()
                .withFaceDescriptor();

            if (det) {
                 // Store the descriptor with the student's name
                 descriptors.push(new faceapi.LabeledFaceDescriptors(stu.name, [det.descriptor]));
            } else {
                 console.warn(`Could not detect face in reference image for ${stu.name}`);
            }
        }

        // Initialize FaceMatcher (0.45 is the tolerance threshold for matching)
        faceMatcher = new faceapi.FaceMatcher(descriptors, 0.45);
        recognitionLoaded = true;

    } catch (e) {
        console.error("Failed to load Face Recognition models/data. Check /models/ and students.json.", e);
        throw new Error("Face recognition failed to initialize.");
    }
}

/**
 * Identifies the face in the given image/bitmap.
 * @param {ImageBitmap} input - The image frame from the video.
 * @returns {string} The recognized student name or "Unknown".
 */
async function identifyFace(input) {
    if (!recognitionLoaded || !faceMatcher) return "Unknown (Loading)";

    // Detect and extract descriptor
    const det = await faceapi
        .detectSingleFace(input, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks()
        .withFaceDescriptor();

    if (!det) return "Unknown";

    // Find the best match
    const match = faceMatcher.findBestMatch(det.descriptor);
    return match.label;
}
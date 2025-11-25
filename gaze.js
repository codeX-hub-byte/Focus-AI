/**
 * Calculates the gaze vector (dx, dy) by comparing the iris center
 * to the midpoint between the left and right eye centers.
 * This is used as input for the State Classifier model.
 * * NOTE: This file is only imported by worker.js
 * * @param {object} iris - Keypoint for the iris center (from FaceMesh).
 * @param {object} leftEye - Keypoint for the left eye center.
 * @param {object} rightEye - Keypoint for the right eye center.
 * @returns {object} {dx, dy} - The gaze deviation vector.
 */
function computeGaze(iris, leftEye, rightEye) {
    // 1. Calculate the center point between the two eye centers
    const cx = (leftEye.x + rightEye.x) / 2;
    const cy = (leftEye.y + rightEye.y) / 2;

    // 2. Gaze deviation is the offset of the iris center relative to the eye midpoint
    // This value is highly sensitive to head position/scale, so it is normalized
    // inside the classifier input preparation.
    return {
        dx: iris.x - cx,
        dy: iris.y - cy
    };
}
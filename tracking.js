// --- Predictive Tracking (Kalman Filter Implementation) ---
class Kalman {
    constructor() {
        // State vector [x, y, dx, dy] - Position and Velocity (as a Tensor)
        this.x = tf.tensor([[0],[0],[0],[0]]);
        // Covariance matrix P (High initial uncertainty)
        this.P = tf.eye(4).mul(100); 

        // State transition matrix F (constant velocity model, dt=1 frame)
        this.F = tf.tensor([[1,0,1,0],
                            [0,1,0,1],
                            [0,0,1,0],
                            [0,0,0,1]]);

        // Observation matrix H (we only observe x and y positions)
        this.H = tf.tensor([[1,0,0,0],
                            [0,1,0,0]]);

        // Measurement noise R (Tuned for typical webcam noise)
        this.R = tf.eye(2).mul(0.1);
        // Process noise Q (Tuned for slight acceleration/deceleration uncertainty)
        this.Q = tf.eye(4).mul(0.01);
    }

    /** Predicts the next state (time update). */
    predict() {
        tf.tidy(() => {
            // Predict next state: x_k = F * x_{k-1}
            this.x = this.F.matMul(this.x);
            // Predict next covariance: P_k = F * P_{k-1} * F^T + Q
            this.P = this.F.matMul(this.P).matMul(this.F.transpose()).add(this.Q);
        });
    }
    
    /** Updates the state with a new measurement z = [x, y] (measurement update). */
    update(z) {
        tf.tidy(() => {
            // Measurement residual y = z - H * x_k
            const y = z.sub(this.H.matMul(this.x));
            // Residual covariance S = H * P_k * H^T + R
            const S = this.H.matMul(this.P).matMul(this.H.transpose()).add(this.R);
            // Kalman Gain K = P_k * H^T * S^{-1}
            const K = this.P.matMul(this.H.transpose()).matMul(tf.linalg.inv(S));

            // Updated state vector x_k' = x_k + K * y
            this.x = this.x.add(K.matMul(y));

            // Updated covariance matrix P_k' = (I - K * H) * P_k
            const I = tf.eye(4);
            this.P = I.sub(K.matMul(this.H)).matMul(this.P);
        });
    }

    /** Gets the smoothed/predicted x and y position. */
    get position() {
        // Must be done outside tf.tidy() to return a concrete value
        const pos = this.x.arraySync();
        return {
            x: pos[0][0],
            y: pos[1][0]
        };
    }
}

/** Manages the state and persistence of a single student. */
class StudentTracker {
    constructor(id, x, y, name="Unknown") {
        this.id = id;
        this.name = name; 
        this.kalman = new Kalman();
        // Initialize Kalman position
        this.kalman.update(tf.tensor([[x],[y]])); 

        this.state = "Unknown"; 
        this.lost = 0; // Frames since last detection
        this.box = null; // Last known bounding box
    }

    update(x, y, state, name, box) {
        this.kalman.update(tf.tensor([[x],[y]])); // Update filter
        
        // Persist ML results
        this.state = state;
        this.name = (name && name !== "Unknown") ? name : this.name; // Keep known name if new one is "Unknown"
        this.box = box;
        this.lost = 0; // Reset lost counter
    }
    
    isFocused() {
        // Consider "Focused" and "Writing" as generally focused states
        return this.state === "Focused" || this.state === "Writing";
    }
}

const trackers = [];
let nextID = 1;

/**
 * Associates new detection measurements with existing trackers.
 * @param {Array} dets - Array of new detection objects from the worker.
 */
function matchStudents(dets) {
    // 1. Predict next location for all existing trackers
    trackers.forEach(t => t.kalman.predict()); 
    const assigned = new Set();

    for (let d of dets) {
        let bestTracker = null;
        let bestDist = 200; // Max distance for association in pixels

        for (let t of trackers) {
            // Compare new detection location (d.x, d.y) to the tracker's PREDICTED position
            const p = t.kalman.position;
            const dist = Math.hypot(d.x - p.x, d.y - p.y);

            // Match if close enough
            if (dist < bestDist && !assigned.has(t.id)) {
                bestDist = dist;
                bestTracker = t;
            }
        }

        if (bestTracker) {
            // Found a match: Update the existing tracker
            bestTracker.update(d.x, d.y, d.state, d.name, d.box);
            assigned.add(bestTracker.id);
        } else {
            // No match: Create a new tracker
            trackers.push(
                new StudentTracker(nextID++, d.x, d.y, d.name)
            );
        }
    }

    // 2. Manage lost trackers
    for (let i = trackers.length - 1; i >= 0; i--) {
        if (!assigned.has(trackers[i].id)) {
            trackers[i].lost++;
            // Remove if lost for more than 30 frames (approx 1 second)
            if (trackers[i].lost > 30) {
                // IMPORTANT: Dispose of Tensors managed by the tracker's Kalman filter
                tf.dispose([trackers[i].kalman.x, trackers[i].kalman.P]); 
                trackers.splice(i, 1);
            }
        }
    }
}
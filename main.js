const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

const loader = document.getElementById("loader");
const countDisplay = document.getElementById("countDisplay");
const focusDisplay = document.getElementById("focusDisplay");
const sensitivitySlider = document.getElementById("sensitivity");
const senseVal = document.getElementById("senseVal");
const initBtn = document.getElementById("initBtn");

// Create the Web Worker
const worker = new Worker("worker.js");

// UI setup for sensitivity slider
senseVal.innerText = sensitivitySlider.value + '%';
sensitivitySlider.oninput = (e) => {
    senseVal.innerText = e.target.value + '%';
};

// State flags
let systemInitialized = false;
let modelsLoaded = false;
let loopActive = false;

// --- Worker Message Handler ---
worker.onmessage = (e) => {
    if (e.data.type === "status") {
        document.querySelector('#loader p').innerText = e.data.message;
        return;
    }
    if (e.data.type === "ready") {
        modelsLoaded = true;
        loader.classList.add('hidden');
        initBtn.disabled = false;
        initBtn.textContent = 'START STREAM';
        console.log("ML Models and Face Recognition loaded in Worker.");
        return;
    }
    
    if (e.data.type === "error") {
        loader.innerHTML = `<h2 class="text-xl font-bold danger-text">CRITICAL ERROR</h2><p class="text-sm mt-2">${e.data.message || 'Check console for details.'}</p>`;
        return;
    }

    if (e.data.type === "data" && loopActive) {
        const dets = e.data.detections;
        matchStudents(dets); // Kalman Filter association and update
        drawUI();
        updateHUD();
    }
};

// --- System Initialization (Called by START SYSTEM button) ---
window.startSystem = async function() {
    if (systemInitialized) return;
    systemInitialized = true;
    initBtn.disabled = true;
    initBtn.textContent = 'Loading Camera...';
    loader.classList.remove('hidden');

    try {
        // Request video stream at target resolution
        const stream = await navigator.mediaDevices.getUserMedia({video: { width: 640, height: 480 }});
        video.srcObject = stream;

        await new Promise(resolve => video.onloadedmetadata = resolve);

        // Set canvas dimensions
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Start worker initialization
        initBtn.textContent = 'Initializing ML Models...';
        worker.postMessage({type: "init"});
        
        // Start the continuous frame processing loop
        loopActive = true;
        renderLoop();

    } catch (err) {
        console.error("Camera or initialization error:", err);
        loader.innerHTML = `<h2 class="text-xl font-bold danger-text">CAMERA ERROR</h2><p class="text-sm mt-2">Please ensure you allow webcam access and refresh.</p>`;
        initBtn.textContent = 'RETRY';
        initBtn.disabled = false;
        systemInitialized = false;
    }
};

// --- Main Loop: Draw frame & Send frame to Worker ---
function renderLoop() {
    if (!loopActive) return;

    // 1. Draw the current video frame onto the canvas
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // 2. Performance Optimization: Send frame to worker as ImageBitmap
    if (modelsLoaded) {
        createImageBitmap(video).then(bitmap => {
            // Transfer ownership of the bitmap to the worker (zero-copy)
            worker.postMessage({type: "frame", bitmap}, [bitmap]);
        }).catch(err => {
             console.warn("Error creating ImageBitmap:", err);
        });
    }

    requestAnimationFrame(renderLoop);
}

// --- UI Drawing Function ---
function drawUI() {
    // Only draw the detection overlays, the video feed is drawn in renderLoop()
    trackers.forEach(t => {
        const box = t.box;
        
        let color = '#ccc'; // Default
        if (t.state === "Focused") color = '#059669'; // Green
        else if (t.state === "Writing" || t.state === "Looking Away") color = '#f59e0b'; // Amber/Orange
        else if (t.state === "Sleeping") color = '#dc2626'; // Red
        
        // Use the tracked name and ID for stable display
        const labelText = `${t.name} (ID:${t.id}) - ${t.state}`; 

        // 1. Draw Bounding Box (Uses the last detected box from the worker)
        if (box) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.lineJoin = 'round';
            ctx.strokeRect(box.x, box.y, box.width, box.height);
        }

        // 2. Draw Label Background
        ctx.fillStyle = color;
        ctx.font = 'bold 16px Inter, sans-serif';
        const textWidth = ctx.measureText(labelText).width;
        // Position the label above the bounding box
        if (box) ctx.fillRect(box.x, box.y - 30, textWidth + 10, 30);

        // 3. Draw Label Text
        ctx.fillStyle = 'white';
        if (box) ctx.fillText(labelText, box.x + 5, box.y - 10);
        
        // 4. Draw Smoothed Position (Kalman Prediction)
        const p = t.kalman.position;
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(p.x, p.y, 5, 0, Math.PI * 2);
        ctx.fill();
    });
}

// --- HUD Update Function ---
function updateHUD() {
    const totalStudents = trackers.length;
    let focusedCount = 0;
    
    trackers.forEach(t => {
        if (t.isFocused()) {
            focusedCount++;
        }
    });

    const focusedPct = totalStudents > 0 ? Math.round((focusedCount / totalStudents) * 100) : 0;
    
    // Update displays
    countDisplay.innerText = totalStudents.toString();
    focusDisplay.innerText = focusedPct + '%';
    
    // Update color based on focus level and sensitivity
    const strictness = parseInt(sensitivitySlider.value, 10);
    const focusThreshold = 100 - strictness; 

    focusDisplay.className = 'text-2xl font-bold leading-none ';
    
    if (focusedPct >= focusThreshold) {
        focusDisplay.classList.add('focused-text'); 
    } else if (focusedPct >= focusThreshold / 2) {
        focusDisplay.classList.add('warning-text'); 
    } else {
        focusDisplay.classList.add('danger-text'); 
    }
}
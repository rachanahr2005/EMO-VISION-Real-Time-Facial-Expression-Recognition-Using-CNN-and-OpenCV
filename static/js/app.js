let cameraStream = null;
let currentImageData = null;
let isCameraRunning = false;
let lastPrediction = null;

const CLASS_NAMES = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
];

document.addEventListener("DOMContentLoaded", () => {

    initializeDashboard();

});


/* =========================================================
   INITIALIZE DASHBOARD
========================================================= */

function initializeDashboard() {

    setupNavigation();

    setupCameraButtons();

    setupImageUpload();

    setupHistory();

    startClock();

}


/* =========================================================
   DOM ELEMENTS
========================================================= */

function getElements() {

    return {

        video: document.getElementById("video"),

        canvas: document.getElementById("canvas"),

        cameraPlaceholder:
            document.getElementById("cameraPlaceholder"),

        startCameraBtn:
            document.getElementById("startCameraBtn"),

        startBtn:
            document.getElementById("startBtn"),

        stopBtn:
            document.getElementById("stopBtn"),

        imageInput:
            document.getElementById("imageInput"),

        uploadImageBtn:
            document.getElementById("uploadImageBtn"),

        uploadFileName:
            document.getElementById("uploadFileName"),

        clearHistoryBtn:
            document.getElementById("clearHistoryBtn"),

        faceCount:
            document.getElementById("faceCount"),

        dominantExpression:
            document.getElementById("dominantExpression"),

        confidenceValue:
            document.getElementById("confidenceValue"),

        confidenceBar:
            document.getElementById("confidenceBar"),

        historyList:
            document.getElementById("historyList"),

    };

}


/* =========================================================
   NAVIGATION
========================================================= */

function setupNavigation() {

    const navLinks = document.querySelectorAll(".nav-link");

    navLinks.forEach(link => {

        link.addEventListener("click", () => {

            navLinks.forEach(item => {
                item.classList.remove("active");
            });

            link.classList.add("active");

        });

    });

}


/* =========================================================
   CAMERA BUTTONS
========================================================= */

function setupCameraButtons() {

    const elements = getElements();

    if (elements.startCameraBtn) {

        elements.startCameraBtn.addEventListener(
            "click",
            startCamera
        );

    }


    if (elements.startBtn) {

        elements.startBtn.addEventListener(
            "click",
            analyzeCurrentFrame
        );

    }


    if (elements.stopBtn) {

        elements.stopBtn.addEventListener(
            "click",
            stopCamera
        );

    }

}


/* =========================================================
   IMAGE UPLOAD
========================================================= */

function setupImageUpload() {

    const elements = getElements();


    /* Choose Image button */

    if (elements.uploadImageBtn) {

        elements.uploadImageBtn.addEventListener(
            "click",
            () => {

                elements.imageInput.click();

            }
        );

    }


    /* File selection */

    if (elements.imageInput) {

        elements.imageInput.addEventListener(
            "change",
            handleImageUpload
        );

    }

}


/* =========================================================
   HANDLE IMAGE UPLOAD
========================================================= */

function handleImageUpload(event) {

    const elements = getElements();

    const file = event.target.files[0];

    if (!file) {
        return;
    }


    /* Validate image */

    if (!file.type.startsWith("image/")) {

        alert("Please select a valid image file.");

        return;

    }


    /* Stop webcam if running */

    stopCamera();


    /* Display filename */

    if (elements.uploadFileName) {

        elements.uploadFileName.textContent =
            file.name;

    }


    /* Read image */

    const reader = new FileReader();


    reader.onload = function(e) {

        currentImageData = e.target.result;

        displayUploadedImage(currentImageData);

    };


    reader.readAsDataURL(file);

}


/* =========================================================
   DISPLAY UPLOADED IMAGE
========================================================= */

function displayUploadedImage(imageData) {

    const elements = getElements();

    const canvas = elements.canvas;

    const ctx = canvas.getContext("2d");

    const image = new Image();


    image.onload = function() {

        /*
         * IMPORTANT:
         * Set canvas size to the actual image size.
         */

        canvas.width = image.naturalWidth;
        canvas.height = image.naturalHeight;


        /*
         * Draw uploaded image.
         */

        ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );


        ctx.drawImage(
            image,
            0,
            0,
            canvas.width,
            canvas.height
        );


        /*
         * Show canvas.
         */

        canvas.style.display = "block";

        elements.video.style.display = "none";

        elements.cameraPlaceholder.style.display = "none";


        /*
         * Make sure the canvas fits inside
         * the viewfinder.
         */

        canvas.style.maxWidth = "100%";
        canvas.style.maxHeight = "620px";

    };


    image.onerror = function() {

        console.error("Unable to display uploaded image.");

        alert("Unable to display this image.");

    };


    image.src = imageData;

}


/* =========================================================
   START CAMERA
========================================================= */

async function startCamera() {

    const elements = getElements();


    try {

        /* Clear uploaded image */

        currentImageData = null;

        if (elements.imageInput) {
            elements.imageInput.value = "";
        }

        if (elements.uploadFileName) {
            elements.uploadFileName.textContent =
                "No image selected";
        }


        /* Request webcam */

        cameraStream =
            await navigator.mediaDevices.getUserMedia({
                video: true,
                audio: false
            });


        elements.video.srcObject = cameraStream;


        await elements.video.play();


        isCameraRunning = true;


        /* Display video */

        elements.video.style.display = "block";

        elements.canvas.style.display = "none";

        elements.cameraPlaceholder.style.display = "none";


        console.log("Camera started successfully.");

    }

    catch (error) {

        console.error(
            "Camera error:",
            error
        );

        alert(
            "Unable to access the camera. " +
            "Please allow camera permission."
        );

    }

}


/* =========================================================
   STOP CAMERA
========================================================= */

function stopCamera() {

    const elements = getElements();


    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(track => track.stop());

        cameraStream = null;

    }


    if (elements.video) {

        elements.video.pause();

        elements.video.srcObject = null;

        elements.video.style.display = "none";

    }


    isCameraRunning = false;


    /*
     * IMPORTANT:
     * Do NOT clear the canvas here if an uploaded image
     * has already been displayed.
     */

    if (!currentImageData) {

        elements.canvas.style.display = "none";

        elements.cameraPlaceholder.style.display = "flex";

    }


}


/* =========================================================
   ANALYZE CURRENT FRAME
========================================================= */

async function analyzeCurrentFrame() {

    const elements = getElements();

    let imageData = null;


    /*
     * CASE 1:
     * Uploaded image
     */

    if (currentImageData) {

        imageData = currentImageData;

    }


    /*
     * CASE 2:
     * Webcam
     */

    else if (
        isCameraRunning &&
        elements.video.videoWidth > 0
    ) {

        const canvas = elements.canvas;

        const ctx = canvas.getContext("2d");


        canvas.width =
            elements.video.videoWidth;

        canvas.height =
            elements.video.videoHeight;


        ctx.drawImage(
            elements.video,
            0,
            0,
            canvas.width,
            canvas.height
        );


        imageData =
            canvas.toDataURL("image/jpeg");


        /*
         * Keep canvas visible after capture.
         */

        canvas.style.display = "block";

        elements.video.style.display = "none";

    }


    else {

        alert(
            "Please start the webcam or upload an image first."
        );

        return;

    }


    /*
     * Send image to Flask
     */

    await sendPrediction(imageData);

}


/* =========================================================
   SEND PREDICTION
========================================================= */

async function sendPrediction(imageData) {

    try {

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({
                    image: imageData
                })
            }
        );


        const result = await response.json();


        console.log(
            "Prediction result:",
            result
        );


        if (!result.success) {

            alert(
                result.message ||
                "Prediction failed."
            );

            return;

        }


        processPrediction(
            result,
            imageData
        );

    }

    catch (error) {

        console.error(
            "Prediction error:",
            error
        );

        alert(
            "Unable to connect to the prediction server."
        );

    }

}


/* =========================================================
   PROCESS PREDICTION
========================================================= */

function processPrediction(
    result,
    imageData
) {

    const elements = getElements();


    /*
     * Update face count
     */

    elements.faceCount.textContent =
        `Faces Detected: ${result.face_count}`;


    /*
     * No face
     */

    if (
        !result.faces ||
        result.faces.length === 0
    ) {

        elements.dominantExpression.textContent =
            "NO FACE";

        elements.confidenceValue.textContent =
            "0%";

        elements.confidenceBar.style.width =
            "0%";

        return;

    }


    /*
     * Use first detected face
     */

    const face = result.faces[0];


    lastPrediction = face;


    /*
     * Update analytics
     */

    updateAnalytics(face);


    /*
     * IMPORTANT:
     *
     * First redraw the uploaded image.
     * Then draw the detection box on top.
     */

    redrawOriginalImage(
        imageData,
        result.faces
    );


    /*
     * Add history
     */

    addHistory(face);

}


/* =========================================================
   REDRAW ORIGINAL IMAGE + FACE BOXES
========================================================= */

function redrawOriginalImage(
    imageData,
    faces
) {

    const elements = getElements();

    const canvas = elements.canvas;

    const ctx = canvas.getContext("2d");

    const image = new Image();


    image.onload = function() {

        /*
         * Keep original image dimensions.
         */

        canvas.width =
            image.naturalWidth;

        canvas.height =
            image.naturalHeight;


        /*
         * Draw original image.
         */

        ctx.clearRect(
            0,
            0,
            canvas.width,
            canvas.height
        );


        ctx.drawImage(
            image,
            0,
            0,
            canvas.width,
            canvas.height
        );


        /*
         * Draw face boxes.
         */

        drawFaceBoxes(
            ctx,
            faces
        );


        /*
         * Show canvas.
         */

        canvas.style.display = "block";

        elements.video.style.display = "none";

        elements.cameraPlaceholder.style.display =
            "none";

    };


    image.src = imageData;

}


/* =========================================================
   DRAW FACE BOXES
========================================================= */

function drawFaceBoxes(
    ctx,
    faces
) {

    faces.forEach(face => {

        const x = face.x;
        const y = face.y;

        const width = face.width;
        const height = face.height;


        /*
         * Face rectangle
         */

        ctx.strokeStyle = "#25d8ff";

        ctx.lineWidth = 3;

        ctx.strokeRect(
            x,
            y,
            width,
            height
        );


        /*
         * Label background
         */

        const label =
            `${capitalize(face.expression)} ${face.confidence.toFixed(1)}%`;


        ctx.font =
            "bold 16px Arial";


        const textWidth =
            ctx.measureText(label).width;


        ctx.fillStyle =
            "rgba(5, 10, 15, 0.85)";


        ctx.fillRect(
            x,
            Math.max(
                0,
                y - 30
            ),
            textWidth + 16,
            28
        );


        /*
         * Label text
         */

        ctx.fillStyle =
            "#25d8ff";


        ctx.fillText(
            label,
            x + 8,
            Math.max(
                19,
                y - 10
            )
        );

    });

}


/* =========================================================
   UPDATE ANALYTICS
========================================================= */

function updateAnalytics(face) {

    const elements = getElements();


    /*
     * Dominant expression
     */

    elements.dominantExpression.textContent =
        capitalize(face.expression);


    /*
     * Confidence
     */

    const confidence =
        Number(face.confidence) || 0;


    elements.confidenceValue.textContent =
        `${confidence.toFixed(1)}%`;


    elements.confidenceBar.style.width =
        `${Math.min(confidence, 100)}%`;


    /*
     * Probabilities
     */

    if (face.probabilities) {

        CLASS_NAMES.forEach(className => {

            const value =
                Number(
                    face.probabilities[className] || 0
                );


            const bar =
                document.getElementById(
                    `bar-${className}`
                );


            const valueElement =
                document.getElementById(
                    `value-${className}`
                );


            if (bar) {

                bar.style.width =
                    `${Math.min(value, 100)}%`;

            }


            if (valueElement) {

                valueElement.textContent =
                    `${value.toFixed(1)}%`;

            }

        });

    }

}


/* =========================================================
   HISTORY
========================================================= */

function setupHistory() {

    const elements = getElements();


    if (elements.clearHistoryBtn) {

        elements.clearHistoryBtn.addEventListener(
            "click",
            clearHistory
        );

    }

}


/* =========================================================
   ADD HISTORY
========================================================= */

function addHistory(face) {

    const elements = getElements();


    if (!elements.historyList) {
        return;
    }


    /*
     * Remove empty state
     */

    const empty =
        elements.historyList.querySelector(
            ".history-empty"
        );


    if (empty) {
        empty.remove();
    }


    const row =
        document.createElement("div");


    row.className =
        "history-row";


    const time =
        new Date().toLocaleTimeString(
            [],
            {
                hour: "2-digit",
                minute: "2-digit",
                second: "2-digit"
            }
        );


    row.innerHTML = `

        <div class="history-expression">
            ${escapeHTML(
                capitalize(face.expression)
            )}
        </div>

        <div class="history-confidence">
            ${Number(face.confidence).toFixed(1)}%
        </div>

        <div class="history-source">
            ${currentImageData ? "Image" : "Webcam"}
        </div>

        <div class="history-time">
            ${time}
        </div>

    `;


    elements.historyList.prepend(row);

}


/* =========================================================
   CLEAR HISTORY
========================================================= */

function clearHistory() {

    const elements = getElements();


    elements.historyList.innerHTML = `

        <div class="history-empty">

            <div class="empty-icon">
                ◷
            </div>

            <h3>No analysis yet</h3>

            <p>
                Start the camera or upload an image
                to begin analysis.
            </p>

        </div>

    `;

}


/* =========================================================
   CLOCK
========================================================= */

function startClock() {

    const clock =
        document.getElementById("clock");


    if (!clock) {
        return;
    }


    function updateClock() {

        const now =
            new Date();


        clock.textContent =
            now.toLocaleTimeString();

    }


    updateClock();


    setInterval(
        updateClock,
        1000
    );

}


/* =========================================================
   HELPERS
========================================================= */

function capitalize(value) {

    if (!value) {
        return "";
    }


    return value.charAt(0).toUpperCase()
        + value.slice(1);

}


function escapeHTML(value) {

    return String(value)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");

}


/* =========================================================
   CLEANUP
========================================================= */

window.addEventListener(
    "beforeunload",
    () => {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );

        }

    }
);
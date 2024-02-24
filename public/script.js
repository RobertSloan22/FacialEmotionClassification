let model;
const startButton = document.getElementById('startButton');
const stopButton = document.getElementById('stopButton');
const webcamContainer = document.getElementById('webcam-container');
const labelContainer = document.getElementById('label-container');
let video;
let intervalId; // Define a variable to hold the interval ID

async function loadModel() {
    console.log("Loading model...");
    model = await tf.loadLayersModel('facialemotions/model.json').catch(e => console.error(e));
    console.log("Model loaded.");
    startButton.disabled = false; // Enable the webcam button only after the model is loaded
}

async function enableWebcam() {
    if (!model) {
        console.error("Model not loaded, cannot enable webcam");
        return;
    }

    if (navigator.mediaDevices.getUserMedia) {
        video = document.createElement('video');
        video.setAttribute('autoplay', '');
        video.setAttribute('playsinline', '');
        video.style.width = '640px';
        video.style.height = '480px';
        webcamContainer.appendChild(video);

        navigator.mediaDevices.getUserMedia({ video: true })
            .then(stream => {
                video.srcObject = stream;
                video.addEventListener('loadeddata', () => {
                    if (intervalId) clearInterval(intervalId);
                    intervalId = setInterval(predictEmotion, 2000); // Update to predict every second
                });
            })
            .catch(error => {
                console.error("Error accessing webcam", error);
            });
    }
}

async function predictEmotion() {
    if (!model) {
        console.error("Model is not loaded yet.");
        return;
    }

    const face = tf.tidy(() => {
        return tf.image.resizeBilinear(
            tf.browser.fromPixels(video),
            [48, 48]
        ).expandDims(0).toFloat().div(tf.scalar(255.0));
    });

    const prediction = await model.predict(face).data();
    displayPrediction(prediction);

    tf.dispose(face);
}

function displayPrediction(prediction) {
    const emotions = ['happy', 'sad', 'surprise', 'neutral'];
    const maxIndex = prediction.indexOf(Math.max(...prediction));
    labelContainer.innerText = `Emotion: ${emotions[maxIndex]}`;
}

function disableWebcam() {
    if (video && video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
    if (intervalId) clearInterval(intervalId);
    labelContainer.innerText = "Webcam disabled";
}

startButton.addEventListener('click', enableWebcam);
stopButton.addEventListener('click', disableWebcam);

// Initially disable the start button until the model is loaded
startButton.disabled = true;
loadModel();

const imageInput = document.getElementById('image-input');
const predImage = document.getElementById('pred-image');
const inpaintButton = document.getElementById('inpaint-button');
const srcCanvas = document.getElementById('src-canvas');
const canvasCtx = srcCanvas.getContext('2d');
const backendURL = 'http://127.0.0.1:5000';
var drawing = false;

srcCanvas.height = 256;
srcCanvas.width = 256;
canvasCtx.lineWidth = 12;
canvasCtx.lineCap = 'round';
canvasCtx.strokeStyle = 'rgb(255, 255, 255)';

const image = new Image();

imageInput.addEventListener('change', (event) => {
    const reader = new FileReader();
    reader.onload = (event) => {
        image.src = event.target.result;
        image.onload = () => {
            canvasCtx.drawImage(image, 0, 0, 256, 256);
        }
    }
    reader.readAsDataURL(event.target.files[0]);
});

srcCanvas.addEventListener('mousedown', (event) => {
    drawing = true;
    canvasCtx.beginPath();
    canvasCtx.moveTo(event.offsetX, event.offsetY);
})

srcCanvas.addEventListener('mousemove', (event) => {
    if (drawing) {
        canvasCtx.lineTo(event.offsetX, event.offsetY);
        canvasCtx.stroke();
    }
})

srcCanvas.addEventListener('mouseup', (event) => {
    drawing = false;
})


inpaintButton.addEventListener('click', (event) => {
    const dataURL = srcCanvas.toDataURL('image/jpeg', 1.0);
    const base64 = dataURL.replace(/^data:image\/(png|jpg);base64,/, '');
    const pureBase64 = base64.split(',')[1];
    fetch(`${backendURL}/predict`, {
        method: 'POST',
        headers: {
            'Content-Type': 'image/jpeg+base64'
        },
        body: pureBase64
    }).then((response) => {
        return response.blob();
    }).then((data) => {
        const imageObjectURL = URL.createObjectURL(data);
        predImage.src = imageObjectURL;
    });
})
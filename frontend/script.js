const imageInput = document.getElementById('image-input');
const predImage = document.getElementById('pred-image');
const inpaintButton = document.getElementById('inpaint-button');
const clearMaskButton = document.getElementById('clear-mask-button');
const refreshModelsButton = document.getElementById('refresh-models-button');
const brushSizeInput = document.getElementById('brush-size-input');
const modelSelect = document.getElementById('model-select');
const srcImage = document.getElementById('src-image');
const srcCanvas = document.getElementById('src-canvas');
const canvasCtx = srcCanvas.getContext('2d');
const backendURL = 'http://127.0.0.1:5000';
var drawing = false;

srcCanvas.height = 256;
srcCanvas.width = 256;
canvasCtx.lineWidth = 12;
canvasCtx.lineCap = 'round';
canvasCtx.strokeStyle = 'rgb(255, 0, 0)';

clearMaskButton.addEventListener('click', (event) => {
    canvasCtx.clearRect(0, 0, 256, 256);
})

refreshModelsButton.addEventListener('click', (event) => {
    fetch(`${backendURL}/models`).then((response) => {
        return response.json();
    }).then((data) => {
        modelSelect.innerHTML = '';
        for (let model of data) {
            const option = document.createElement('option');
            option.value = model;
            option.text = model;
            modelSelect.appendChild(option);
        }
    });
})
refreshModelsButton.click();

brushSizeInput.addEventListener('change', (event) => {
    canvasCtx.lineWidth = event.target.value;
})

imageInput.addEventListener('change', (event) => {
    const reader = new FileReader();
    reader.onload = (event) => {
        srcImage.src = event.target.result;
        canvasCtx.clearRect(0, 0, 256, 256);
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

srcCanvas.addEventListener('mouseleave', (event) => {
    drawing = false;
})


inpaintButton.addEventListener('click', (event) => {
    // serialize canvas to base64
    const dataURL = srcCanvas.toDataURL('image/png');
    const maskBase64 = dataURL.replace(/^data:image\/(png|jpg);base64,/, '');

    // serialize srcImage to base64
    const srcDataURL = srcImage.src;
    const srcBase64 = srcDataURL.replace(/^data:image\/(png|jpeg);base64,/, '');

    const selectedModel = modelSelect.options[modelSelect.selectedIndex].value;
    fetch(`${backendURL}/predict`, {
        method: 'POST',
        headers: {  
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            image: srcBase64,
            mask: maskBase64,
            model: selectedModel,
        })
    }).then((response) => {
        return response.blob();
    }).then((data) => {
        const imageObjectURL = URL.createObjectURL(data);
        predImage.src = imageObjectURL;
    });
})
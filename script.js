const content = document.querySelector('#content');
const fileInput = document.querySelector('#file-input');

async function loadTranningData(){
    const labels = ['Đeo kính', 'Không đeo kính'];
    const faceDescriptors = [];
    
    for(const label of labels){
        const descriptors = [];
        
        for(let i = 1; i <= 20; i++){
            const image = await faceapi.fetchImage(`/data/${label}/${i}.jpg`);
            const detection = await faceapi.detectSingleFace(image).withFaceLandmarks().withFaceDescriptor();
            descriptors.push(detection.descriptor);
        }

        faceDescriptors.push(new faceapi.LabeledFaceDescriptors(label, descriptors));

        Toastify({
            text: `Training xong dữ liệu của ${label}`
        }).showToast();
    }

    return faceDescriptors;
}
let faceMatcher;
async function init(){
    await Promise.all([
        faceapi.nets.ssdMobilenetv1.loadFromUri('/models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('/models')
    ])

    const trainingData = await loadTranningData();
    faceMatcher = new faceapi.FaceMatcher(trainingData, 1);
    Toastify({text: 'Tải xong mô hình'}).showToast();
}
init()

document.addEventListener('DOMContentLoaded', (event) => {
    const content = document.querySelector('#content');
    const fileInput = document.querySelector('#file-input');

    fileInput.addEventListener('change', async (e) => {
        const file = fileInput.files[0];
        const image = await faceapi.bufferToImage(file);
        const canvas = faceapi.createCanvasFromMedia(image);

        content.innerHTML = '';
        content.appendChild(image);
        content.appendChild(canvas);
        const size = {
            width: image.width,
            height: image.height
        }
        faceapi.matchDimensions(canvas, size);
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors();
        const resizedDetections = faceapi.resizeResults(detections, size);
        for (const detection of resizedDetections) {
            const drawBox = new faceapi.draw.DrawBox(detection.detection.box, {
              label: faceMatcher.findBestMatch(detection.descriptor).toString(),
            });
            drawBox.draw(canvas);
          }
    });
});

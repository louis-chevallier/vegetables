
let camera, renderer, cameraControls;
let mouseX = 0, mouseY = 0;
let windowHalfX = window.innerWidth / 2;
let windowHalfY = window.innerHeight / 2;
let object;

const session = new onnx.InferenceSession()
//const session = new onnx.InferenceSession({ backendHint: 'webgl' });

function preprocess(data, width, height) {
    const dataFromImage = ndarray(new Float32Array(data), [width, height, 4]);
    const dataProcessed = ndarray(new Float32Array(width * height * 3), [1, 3, height, width]);
    
    EKOX(dataFromImage.get(1,1,0))
    EKOX(dataFromImage.get(1,1,1))
    EKOX(dataFromImage.get(1,1,2))    
    EKOX(dataFromImage.get(1,1,3))    
    
    
    // Normalize 0-255 to (-1)-1
    ndarray.ops.divseq(dataFromImage.pick(2, null, null), 255);
    ndarray.ops.divseq(dataFromImage.pick(1, null, null), 255);
    ndarray.ops.divseq(dataFromImage.pick(0, null, null), 255);

    EKOX(dataFromImage.get(1,1,0))
    EKOX(dataFromImage.get(1,1,1))
    EKOX(dataFromImage.get(1,1,2))    
    EKOX(dataFromImage.get(1,1,3))    
    
    ndarray.ops.subseq(dataFromImage.pick(2, null, null), 0.485);
    ndarray.ops.subseq(dataFromImage.pick(1, null, null), 0.456);
    ndarray.ops.subseq(dataFromImage.pick(0, null, null), 0.406);

    EKOX(dataFromImage.get(1,1,0))
    EKOX(dataFromImage.get(1,1,1))
    EKOX(dataFromImage.get(1,1,2))    
    EKOX(dataFromImage.get(1,1,3))    
    
    ndarray.ops.divseq(dataFromImage.pick(2, null, null), 0.229);
    ndarray.ops.divseq(dataFromImage.pick(1, null, null), 0.224);
    ndarray.ops.divseq(dataFromImage.pick(0, null, null), 0.225);
    EKOX(dataFromImage.get(1,1,0))
    EKOX(dataFromImage.get(1,1,1))
    EKOX(dataFromImage.get(1,1,2))    
    EKOX(dataFromImage.get(1,1,3))    
    
    // Realign imageData from [224*224*4] to the correct dimension [1*3*224*224].
    ndarray.ops.assign(dataProcessed.pick(0, 0, null, null), dataFromImage.pick(null, null, 2));
    ndarray.ops.assign(dataProcessed.pick(0, 1, null, null), dataFromImage.pick(null, null, 1));
    ndarray.ops.assign(dataProcessed.pick(0, 2, null, null), dataFromImage.pick(null, null, 0));
    
  return dataProcessed.data;
}

async function load_model(){
    EKOX("loading model")
    await session.loadModel('get_model')
    EKOX("model loaded");
    load_model_button.innerHTML = "model loaded locally"   
}

async function local_predict() {
    // load image.
    const imageSize = 224;
    const imageLoader = new ImageLoader(imageSize, imageSize);


    const width = 224; //imageSize;
    const height = 224; //imageSize;

    let resizedCanvas = document.createElement('canvasrz');
    resizedCanvas.width = width;
    resizedCanvas.height = height;
    let imageData = 0
    if (1>0) { 
        const imageLoader = new ImageLoader(imageSize, imageSize);
        imageData = await imageLoader.getImageData('./blue.png');
        //let image_data_url = imageLoader.canvas.toDataURL('image/jpeg');
    } else {
        canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
        //let image_data_url = canvas.toDataURL('image/jpeg');
        imageData = canvas.toDataURL('image/jpeg');
    }


    
    //let resizedCtx = resizedCanvas.getContext('2d');
    //resizedCtx.drawImage(canvas, 0, 0, width, height);
    //const imageData = resizedCtx.getImageData(0, 0, width, height).data;
    //canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    
    //const imageData = canvas.getContext('2d').getImageData(0, 0, width, height).data;    
    //const imageData = await imageLoader.getImageData('./tomato.jpg');


    // preprocess the image data to match input dimension requirement, which is 1*3*224*224
    const preprocessedData = preprocess(imageData.data, width, height);
    
    const inputTensor = new onnx.Tensor(preprocessedData, 'float32', [1, 3, width, height]);
    // Run model with Tensor inputs and get the result.
    const outputMap = await session.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;

    function softmax(arr) {
        return arr.map(function(value,index) { 
            return Math.exp(value) / arr.map( function(y /*value*/){ return Math.exp(y) } ).reduce( function(a,b){ return a+b })
        })
    }
    EKOX(softmax(outputData))
    var arr = outputData
    var smarr = softmax(arr)
    var imax = arr.indexOf(Math.max.apply(Math, arr));
    EKOX(imax)
    
    // Render the output result in html.
    EKOX(outputData);

    dataurl.value = "result:" + classes_array[imax] + " p=" + smarr[imax];
}

async function remote_predict() {

    const imageSize = 224;
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    let image_data_url = canvas.toDataURL('image/jpeg');

    
    let data = JSON.stringify({image: image_data_url});
    const chunk = data.split(',').pop()
    EKOX("fetching");
    const response = await fetch('chunk', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            chunk: chunk
        })
    })
    EKOX('fetched');
    const json = await response.json();
    EKOX(json.status)
    /*
    var httpPost = new XMLHttpRequest()
    //httpPost.setHeader('Content-Type', 'application/json');
    httpPost.open("POST",  "/get_photo", true);
    httpPost.send(data);
    EKOX("sent");
    */
    if (json.status == "ok")  {
        dataurl.value = json.name + " p=" + json.probability + ", request number=" + json.requests;
    }
    //dataurl.value = image_data_url;
    dataurl_container.style.display = 'block';

}


function date() {
    var today = new Date();
    var date = today.getFullYear()+'-'+(today.getMonth()+1)+'-'+today.getDate();
    var time = today.getHours() + ":" + today.getMinutes() + ":" + today.getSeconds();
    var dateTime = date+' '+time;
    return dateTime
}    

function onWindowResize() {

    windowHalfX = window.innerWidth / 2;
    windowHalfY = window.innerHeight / 2;
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize( window.innerWidth, window.innerHeight );
}

function onDocumentMouseMove( event ) {
    mouseX = ( event.clientX - windowHalfX ) / 2;
    mouseY = ( event.clientY - windowHalfY ) / 2;
}

function trace(txt) {
    let t = document.getElementById("json").innerHTML;    
    if (txt == "clear") {
        t = "";
    }
    if (t.length > 2000) {
        t = t.slice(0, 2000);
    }
    document.getElementById("json").innerHTML  =  date() + txt + "<br>" + t;        
}




var selectPhoto = document.getElementById('photoselect');
var formPhoto   = document.getElementById('uploadPhoto')
console.log('init');


// trick pour rendre ces fct accessibles depuis le html ..
//window.doReset = doReset;
//window.doGo = doGo;
//window.activate = activate;



let camera_button = document.querySelector("#start-camera");
let video = document.querySelector("#video");
let click_button = document.querySelector("#click-photo");
let canvas = document.querySelector("#canvas");
let dataurl = document.querySelector("#dataurl");
let dataurl_container = document.querySelector("#dataurl-container");
canvas.hidden = true;    
camera_button.addEventListener('click', start_cam)


let classes_area = document.querySelector("#classes");
const response = await fetch('get_classes', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({})
})
EKOX('fetched');
let classes_text = await response.text()
EKOX(classes_text);
let classes_array = classes_text.split(" ");
EKOX(classes_array);

classes_area.innerText = classes_array;


let load_model_button = document.querySelector("#load-model");
load_model_button.addEventListener('click', load_model)

let local_predict_button = document.querySelector("#local_predict");
local_predict_button.addEventListener('click', local_predict)

let remote_predict_button = document.querySelector("#remote_predict");
remote_predict_button.addEventListener('click', remote_predict)


async function start_cam() {
    EKOX("starting camera");
    let stream = null;
    try {
    	stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: { exact : 'environment' }},
                                                             audio: false });
    }
    catch(error) {
        try {
    	    stream = await navigator.mediaDevices.getUserMedia({ video: true,
                                                                 audio: false });
        }  catch (error1) {
    	    alert(error1.message);
    	    return;
        }
    }

    video.srcObject = stream;

    video.style.display = 'block';
    camera_button.style.display = 'none';
    click_button.style.display = 'block';
    EKOX("camera started");
};

async function send_photo() {
    EKOX("read image ");
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    let image_data_url = canvas.toDataURL('image/jpeg');
    let data = JSON.stringify({image: image_data_url});
    const chunk = data.split(',').pop()
    EKOX("fetching");
    const response = await fetch('chunk', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            chunk: chunk
        })
    })
    EKOX('fetched');
    const json = await response.json();
    EKOX(json.status)
    /*
    var httpPost = new XMLHttpRequest()
    //httpPost.setHeader('Content-Type', 'application/json');
    httpPost.open("POST",  "/get_photo", true);
    httpPost.send(data);
    EKOX("sent");
    */
    if (json.status == "ok")  {
        dataurl.value = json.name + " p=" + json.probability;
    }
    //dataurl.value = image_data_url;
    dataurl_container.style.display = 'block';

};
EKOX("starting");
click_button.addEventListener('click', send_photo)



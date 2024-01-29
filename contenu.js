
let camera, renderer, cameraControls;
let mouseX = 0, mouseY = 0;
let windowHalfX = window.innerWidth / 2;
let windowHalfY = window.innerHeight / 2;
let object;


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




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

function doGo() {
    let murl = "/process";
    //console.log("fetching");
    fetch(murl).then(function(response) {
        console.log("response");
        let d = response.json();
        console.log(d);
        return d;
    }).then(function(data) {
        trace(" /runOnPhoto response=" + JSON.stringify(data))
    })
}




function activate(b)
{
    console.log('b', b);
    const req1 = new XMLHttpRequest();
    req1.open("POST", "/config", true);
    req1.setRequestHeader('X-active', b);
    console.log('sending', b);
    req1.send();
    trace(" /config active= " + b)

}

function uploadPhoto(file)
{
    
    var xhr = new XMLHttpRequest();
    console.log("uploading photo");
    var photoselect = document.getElementById("photoselect")
    photoselect.disabled = true
    console.log('uploading photo ..');


    let formData = new FormData(); // creates an object, optionally fill from <form>
    formData.append("ufile", file.name); // appends a field
    
    xhr.upload.addEventListener('progress', function(event) 
                                {
                                    console.log('progress uploading', file.name, event.loaded, event.total);
                                });
    xhr.addEventListener('readystatechange', function(event) 
                         {
                             console.log(
                                 'ready state', 
                                 file.name, 
                                 xhr.readyState, 
                                 xhr.readyState == 4 && xhr.status
                             );
                             photoselect.disabled = false                             
                         });
    console.log("posting ...")
    xhr.open('POST', '/uploadPhoto', true);
    xhr.setRequestHeader('X_Filename', file.name);
    xhr.setRequestHeader("Content-Type", "multipart/form-data")
    xhr.send(formData)    
    console.log('sending', file.name, file);
    //xhr.send(file)
    trace(" /uploadPhoto filename=" + file.name)
    
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

camera_button.addEventListener('click', async function() {
   	let stream = null;
    try {
    	stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    }
    catch(error) {
    	alert(error.message);
    	return;
    }

    video.srcObject = stream;

    video.style.display = 'block';
    camera_button.style.display = 'none';
    click_button.style.display = 'block';
    EKOX("camera started");
});

async function send_photo() {
    EKOX("sending");
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
    let image_data_url = canvas.toDataURL('image/jpeg');
    let data = JSON.stringify({image: image_data_url});
    const chunk = data.split(',').pop()
    const response = await fetch('chunk', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            chunk: chunk
        })
    })
    EKOX('fetching');
    const json = await response
    EKOX(json)
    const j = json.json();
    EKOX(j)
    /*
    var httpPost = new XMLHttpRequest()
    //httpPost.setHeader('Content-Type', 'application/json');
    httpPost.open("POST",  "/get_photo", true);
    httpPost.send(data);
    EKOX("sent");
    */
    dataurl.value = image_data_url;
    dataurl_container.style.display = 'block';
};
EKOX("starting");
click_button.addEventListener('click', send_photo)



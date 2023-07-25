var record = false;
var emotion;

function RecordButton(){
    if(record == false)
    {
        record = true;
        stopVideoCapture();
    }
    else
    {
        record = false;
        startVideoCapture();
    }
}
     

var socket = io.connect('http://localhost:5000');

socket.on('connect', function(){
    console.log("Connected...!", socket.connected)
});

const video = document.querySelector("#videoElement");

video.width = 266; 
video.height = 200; 

var videoStream = null;

window.addEventListener("load", startVideoCapture);


//Video Capture
function startVideoCapture()
{
    if (navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ video: true })
        .then(function (stream) {
            videoStream = stream;
            video.srcObject = stream;
            video.play();
        })
        .catch(function (err0r) {
            console.log(err0r)
            console.log("Something went wrong!");
        });
    }
}

function stopVideoCapture() {
    if (videoStream) {
      const tracks = videoStream.getTracks();
      tracks.forEach(track => track.stop());
      videoStream = null;
    }
}




const canvas = document.querySelector("#canvasOutput");
canvas.width = video.width;
canvas.height = video.height;
const context = canvas.getContext('2d', { willReadFrequently: true });

let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let cap = new cv.VideoCapture(video);



const FPS = 15;
var timeout;

function processVideo() {
    let begin = Date.now();
    cap.read(src);
    src.copyTo(dst);
    if(!record)
    {
        cv.imshow("canvasOutput", dst);
    
        // schedule next one.
        let delay = 1000/30 - (Date.now() - begin);
        setTimeout(processVideo, delay);
    }
    else
    {
        setTimeout(processVideo, 0);

    }
}



// schedule the first one.
setTimeout(processVideo, 0);

//Send to Server
var a = setInterval(() => {
    if(!record){
        cap.read(src);
        var type = "image/png"
        var data = document.getElementById("canvasOutput").toDataURL(type, 0.1);
        data = data.replace('data:' + type + ';base64,', ''); //split off junk 
        socket.emit('image', data);
    }
}, 10000/FPS);



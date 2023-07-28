var record = false;

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

//Stop Video Capture
function stopVideoCapture() {
    if (videoStream) {
      const tracks = videoStream.getTracks();
      tracks.forEach(track => track.stop());
      videoStream = null;
    }
}


//Control webcam capture and sending quality and aspect ratio
video.width = 533; 
video.height = 400; 

//FPS, rate at which the video frames are sent
const FPS = 60;



//The canvas where the video feed from the webcam is shown directly
const canvas = document.querySelector("#canvasOutput");
canvas.width = video.width;
canvas.height = video.height;
const context = canvas.getContext('2d', { willReadFrequently: true });

let src = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let dst = new cv.Mat(video.height, video.width, cv.CV_8UC4);
let cap = new cv.VideoCapture(video);


var timeout;

//For showing webcam feed on the Canvas
function processVideo() {
    let begin = Date.now();
    cap.read(src);
    src.copyTo(dst);
   

    // Draw a rectangle on the destination image (dst)  
    const rectStart = new cv.Point(video.width - 100, 100);
    const rectEnd = new cv.Point(video.width - (100 + 160), (100 + 160));
    const rectColor = new cv.Scalar(0, 255, 255); 
    const rectThickness = 2;
    cv.rectangle(dst, rectStart, rectEnd, rectColor, rectThickness, cv.LINE_8, 0);

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


//Send frame to Server
var a = setInterval(() => {
    if(!record){
        cap.read(src);
        var type = "image/png"
        var data = document.getElementById("canvasOutput").toDataURL(type, 0.1);
        data = data.replace('data:' + type + ';base64,', ''); //split off junk 
        socket.emit('image', data);
    }
}, 10000/FPS);

var preview_container = document.getElementById("preview2-container");
var img = document.getElementById("preview")
var live_letter = document.getElementById("live-letter");
var confidence = document.getElementById("confidence");
var interpreted_text = document.getElementById("interpreted-text");

var letter_counter = 0;
var previous_letter = '';

// Listen and receive data from Flask Server
socket.on('processed_frame', function(data) {

    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');

    img.width = 300;
     
    // Decode the base64 encoded frame data into an image
    img.src = 'data:image/jpeg;base64,' + data.frame;

    var score = (parseFloat(data.prediction_score)*100).toFixed(2);


    if(score > 90)
    {
        confidence.style.color = 'green';
        if(previous_letter == data.letter)
        {
            letter_counter++;
        }
        else
        {
            letter_counter=0;
        }
        
        //change the consecutive occuring of the letter
        if(letter_counter > 5)
        {
            if (previous_letter == "space")
                interpreted_text.innerHTML += " ";
            else if (previous_letter == "del")
                interpreted_text.innerHTML = interpreted_text.innerHTML.slice(0, -1);
            else
                interpreted_text.innerHTML += previous_letter;
            
                letter_counter = 0;
        }

        previous_letter = data.letter;
    }
    else
    {
        confidence.style.color = 'black';
    }

    live_letter.innerHTML = data.letter;
    confidence.innerHTML =  score + '%';

  });



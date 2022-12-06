const http = require('http');
const url = require('url');
const qs = require('querystring');
const fs = require('fs');

const express = require('express');
const app = express();

app.use(express.static('public'));

app.get('/', function (req, res) {
  res.sendFile('index.html');
});

app.post('/post_test', function (req, res) {
  var body = '';
  req.on('data', function (data) {
    body = body + data;
  });
  req.on('end', function () {
    var post = qs.parse(body);

    var title = post.email;
    var url = 'URL';
    console.log(url + ':' + title);

    // 1. child-process모듈의 spawn 취득
    const spawn = require('child_process').spawn;

    // 2. spawn을 통해 "python 파이썬파일.py" 명령어 실행

    const result_03 = spawn('python', ['deep_url_classifier.py', title]); //post or tittle
    result_03.stdout.on('data', (result) => {
      console.log(result.toString());
    });

    setTimeout(function () {
      res.sendFile(__dirname + '/public/result.html');
    }, 15000);
  });
});

app.listen(3000);

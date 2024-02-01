#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://usefulangle.com/post/352/javascript-capture-image-from-camera

import os, gc, sys, glob
import os, json, base64
import shutil
import re
from utillc import *
import cherrypy
import threading
import queue
import json, pickle
import time
import time as _time
from time import gmtime, strftime
from datetime import timedelta
import PIL
from PIL import Image
import os
from urllib.parse import urlparse

import train

fileDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(fileDir, '.')
EKOX(rootDir)

port = 8080
if "PORT" in os.environ :
    port = int(os.environ["PORT"])

OK="OK"
FAILED="FAILED"
STATUS="status"
REASON="reason"
no_path = "no_path"

config = {
  '/' : {
      'tools.staticdir.on': True,
      'tools.staticdir.dir': rootDir,
#      'tools.staticdir.dir': '/mnt/hd2/users/louis/dev/git/three.js/examples/test',

    },
  'global' : {
      'server.ssl_module' : 'builtin',
      'server.ssl_certificate' : "cert.pem",
      'server.ssl_private_key' : "privkey.pem",
      
      'server.socket_host' : '0.0.0.0', #192.168.1.5', #'127.0.0.1',
      'server.socket_port' : port,
      'server.thread_pool' : 8,
      'log.screen': False,
      'log.error_file': './error.log',
      'log.access_file': './access.log'
  },
}

dirphotos = "/tmp/photos_%s" % os.environ["USER"]
tmpphotos = "/tmp/photos_tmp_%s" % os.environ["USER"]


class App:
    """
    the Webserver
    """
    def __init__(self, gd, train_dir=None) :
        EKOT("app init")
        self.no_image = 0
        self.gd = gd
        v = self.vegetable = train.Vegetable(gd, use_gpu=True, model_name="resnet50", train_dir=train_dir)
        self.model = model = v.test(measure=False, disp=False)
        model.eval()
        v.predict(model, Image.open('brocoli.jpg'))
        EKOX(self.get_next_image_num())
        os.makedirs( os.path.join(gd, "tests"), exist_ok=True)

        
    def info(self) :
        def read(gi) :
            i = os.environ[gi] if gi in os.environ else ""
            return gi + "=" + i
        return read('GITINFO') + ", " + read("HOST") + ", " + read("DATE")
            
    @cherrypy.expose
    def index(self):
        """ main 
        """
        EKOT("REQ main")
        with open('./main.html', 'r') as file:
            EKOT("main")
            data = file.read()
            data = data.replace("INFO", self.info())
            return data

    @cherrypy.expose
    def get_model(self, number=60):
        EKOT("REQ model")
        fn = os.path.join(self.gd, "models", "vegetables_mobilenet_v2_%03d.onnx" % number)
        EKOX(fn)
        with open(fn, 'rb') as file:
            data = file.read()
            return data

    @cherrypy.expose
    def log(self, data=None) :
        #EKO()
        p = urlparse(data);
        rp = os.path.relpath(p.path, start = "/")
        print(rp)
   
    def get_next_image_num(self) :
        l = glob.glob("test/test_*.jpg")
        EKOX(l)
        l = sorted(l)
        EKOX(l[-1])
        result = re.search(r"test_([0-9]+).jpg", l[-1])
        EKOX(result.group(1))
        return int(result.group(1))+1
        
    @cherrypy.expose
    def chunk(self, data=None) :
        EKOT("received chunk")
        #EKOX(data);
        try :
            fn = "tests/test_%04d.jpg" % self.get_next_image_num()
            self.no_image += 1
            self.out = open(fn, "wb")
            body = cherrypy.request.body.read()
            EKOX(len(body));
            c = json.loads(body)
            d = c["chunk"]
            EKOX(len(d))
            e = base64.b64decode(d)
            EKOX(len(e))
            self.out.write(e)
            EKOT("written")
            EKOT("predicting")
            label, _, prob = self.vegetable.predict(self.model, Image.open(fn))
            EKOX(self.vegetable.idx_to_class[label])
            ans =  json.dumps({"status" : "ok",
                               "probability" : float(prob.cpu().numpy()),
                               "label" :  int(label), "name" : self.vegetable.idx_to_class[label]})
            EKO()
        except Exception as e :
            EKOX(e)
            ans =  json.dumps({"status" : e})
        EKOX(ans)
        return ans
    
    @cherrypy.expose
    def exit(self):
        EKOT("REQ exit")
        self.queue.put({ 'exit' : True } )
        #sys.exit(0)
        cherrypy.engine.exit()


        
config2 = {
    "dry" : (False, " true : will not run the reconstructor"),
    "gitinfo" : "info"
}

def go(gd = "/content/gdrive/MyDrive/data", train_dir=None) :
    app = App(gd, train_dir)
    cherrypy.log.error_log.propagate = False
    cherrypy.log.access_log.propagate = False
    EKO()
    cherrypy.quickstart(app, '/', config)
    EKOT("end server", n=LOG)

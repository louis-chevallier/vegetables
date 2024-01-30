#!/usr/bin/env python
# -*- coding: utf-8 -*-

# https://usefulangle.com/post/352/javascript-capture-image-from-camera

import os, gc, sys
import os, json, base64
import shutil
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

class Reconstructor(threading.Thread):
    """
    the thread which runs the reconstructor, 
    comunication with the web server using 2 queues
    """
    def __init__(self, server, args) :
        super().__init__()        
        pass
        self.queue = queue.Queue(maxsize=50)
        self.event = threading.Event()
        self.server = server
        self.progress = []
        EKO()
        self.args = args
        device_name = args.dev
        dataroot = args.dataroot
        EKO()
        args.showloss=False
        self.fitting = None
        self.vegetable = train.Vegetable()
        EKO()
        self.total=1
        self.running = False
        self.computationTime = "time : nothing yet"
        self.radic = 0
        self.last_objpath = no_path
        
    def run(self) :
        EKOT("http://localhost:%d/main.html" % config["global"]["server.socket_port"])        
        while True :
            EKOT("waiting for request to process")
            try :
                EKO();
                go_on = self.process();
                if not go_on :
                    EKO()
                    break
            except Exception as e :
                EKOX(e, n=WARNING)
        EKO()

        
        
    def process(self) -> bool :
        self.running = False
        EKOT("waiting for something")
        EKOX(self.server.queue.qsize())                    
        req = self.server.queue.get()
        EKOX(self.server.queue.qsize())                 
        EKOT("got something")
        if "exit" in req :
            EKO()
            return False
            
        self.progress = []
        d = self.vegetable.predict(req['image'])
        EKOX(d)
        self.queue.put(d)            
        EKOT("mess sent")
        return True

class App:
    """
    the Webserver
    """
    def __init__(self, gd) :
        EKOT("app init")
        self.no_image = 0
        self.gd = gd
        v = self.vegetable = train.Vegetable(gd, use_gpu=True)
        self.model = model = v.test(measure=False, disp=False)
        model.eval()
        v.predict(model, Image.open('brocoli.jpg'))
        
    def info(self) :
        return "xxx"
        
    @cherrypy.expose
    def index(self):
        EKOT("REQ main")
        with open('./main.html', 'r') as file:
            EKOT("main")
            data = file.read()
            data = data.replace("INFO", self.info())
            return data

    @cherrypy.expose
    def get_model(self, number=40):
        EKOT("REQ model")
        fn = os.path.join(self.gd, "vegetables_%03d.onnx" % number)
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
   

    @cherrypy.expose
    def chunk(self, data=None) :
        EKOT("received chunk")
        #EKOX(data);
        try :
            fn = "test_%04d.jpg" % self.no_image
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

def go(gd = "/content/gdrive/MyDrive/data") :
    app = App(gd)
    cherrypy.log.error_log.propagate = False
    cherrypy.log.access_log.propagate = False
    EKO()
    cherrypy.quickstart(app, '/', config)
    EKOT("end server", n=LOG)

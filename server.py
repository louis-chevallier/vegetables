#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os, gc, sys
import shutil
from utillc import *
import cherrypy
import threading
import queue
import json, pickle
from iterative_fit import OBJPATH
import time
import time as _time
from time import gmtime, strftime
from datetime import timedelta
import PIL
from PIL import Image
import os

fileDir = os.path.dirname(os.path.abspath(__file__))
rootDir = os.path.join(fileDir, 'www/test')
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
      'server.socket_host' : '0.0.0.0', #192.168.1.5', #'127.0.0.1',
      'server.socket_port' : port,
      'server.thread_pool' : 8,
      'log.screen': False, #True,
      'log.error_file': './error.log',
      'log.access_file': './access.log'
  }
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
    def __init__(self, args) :
        EKOT("app init")
        self.queue = queue.Queue(maxsize=50)
        self.reconstructor = Reconstructor(self, args)
        EKO()
        self.reconstructor.start()
        EKO()
        self.no_image = 0
        self.resetImageCounter()
        self.fitting_active = False
        self.args = args
        EKO()
        
    def info(self) :
        return self.args.gitinfo
        
    @cherrypy.expose
    def main(self):
        EKOT("REQ main")
        with open('www/test/main.html', 'r') as file:
            data = file.read()
            data = data.replace("INFO", self.info())
            return data
        
    @cherrypy.expose
    def config(self):
        """
        - to configure the reconstruction
        - POST 
        - params : 
          - quality  : float 0. - 1.
          - active : 1/0 
        - return : { 'status' : OK/FAILED }
        """
        EKOT("REQ config")
        try :
            quality = 1.
            try :
                EKOX(cherrypy.request.headers.keys())
                active = os.path.basename(cherrypy.request.headers['x-active'])
                EKOX(active)
                squality = os.path.basename(cherrypy.request.headers['x-quality'])
                quality = float(squality)
            except Exception as e:
                EKOX(e)
                pass
            if self.reconstructor.fitting is not None :
                self.reconstructor.fitting.args.cycles = quality
            EKOX(quality)
            self.queue.put({
                'config' : {
                    "active" : active,
                    'quality' : quality,
                }} )

            rep = "set to %f" % quality
            #EKOX(rep)
            rep = { STATUS : OK,
                    "quality" : quality }
        except Exception as e :
            rep = { STATUS : FAILED}
            EKOX(e)
        EKOX(rep)
        return json.dumps(rep)
        
    @cherrypy.expose
    def progress(self):
        """
        - to get info on on going calculation
        - GET 
        - params : 
          - 
        - return dict :
          - 'status' : OK/FAILED,
          - 'message' : info
          - 'radic' : radical of result video
          - 'uploaded' : number of uploaded images so far
          - 'active' : 0/1 : indicate reconstruction status ( real/fake )
        }
        """
        
        #EKOT("REQ progress")
        #EKOX(cherrypy.request.remote.ip)
        d = {}
        try :
            #EKOX( self.reconstructor.queue.empty())
            ln = int(100 * len(self.reconstructor.progress) / self.reconstructor.total)
            d = { "progress" : ln }
            #EKOX(d)
            #EKOX(self.queue.qsize())            
            d["message"] = ""
            #d["bar"] = str("=") * ln + str("_") * (100 - ln)
            if not self.reconstructor.queue.empty() :
                EKO()
                rep = self.reconstructor.queue.get()
                op = OBJPATH
                cjf = "computation_just_finished"
                if cjf in rep :
                    d[cjf] = 1                
                if op in rep :
                    d[op] = rep[op]
                if STATUS in rep :
                    a = rep[STATUS] == "active"
                    self.fitting_active = d["active"] = a
                EKOX(d)
                
            d[OBJPATH] = self.reconstructor.last_objpath
                
            d["images_uploaded"] = self.no_image
            d["running"] = self.reconstructor.running

            #fitting_active = self.reconstructor.fitting is not None 
            
            d["message"] = self.reconstructor.computationTime
            d["message"] += (", images uploaded : %d " % self.no_image)
            d["message"] += ", real reconstructor active=%d" % self.fitting_active
            d["message"] += ", radic=%d" % self.reconstructor.radic
            d["radic"] = self.reconstructor.radic
            d["active"] = self.fitting_active
            d["uploaded"] = self.no_image
            #EKOX(d)

        except Exception as e  :
            EKOX(e)
            d = { STATUS : FAILED }
        #EKO()
        s = json.dumps(d)
        #EKOX(s)
        return s
    
    #@cherrypy.expose
    def indexXX(self):
        EKO()
        with open('www/test/page.html', 'r') as file:
            data = file.read()
            return data
        
    @cherrypy.expose
    def runOnPhoto(self):
        ''' launch the reconstruction (from photo since, when uploading a video, the reconstruction is launched automatically)
        '''
        EKOT("REQ runPhotos")

        if self.reconstructor.running :
            EKOT("RECONSTRUCTION STILL ON GOING!!!", n=WARNING)
        
        self.queue.put({ 'run_photos' : True } )
        EKOX(self.queue.qsize())
        rep = { STATUS : OK }
        EKOX(rep)
        return json.dumps(rep)

    @cherrypy.expose
    def resetImageCounter(self):
        ''' 
        '''
        self.no_image = 0
        EKOT("REQ resetImageCounter")
        try :
            shutil.rmtree(dirphotos)
        except :
            pass
        os.makedirs(dirphotos, exist_ok=True)
        os.makedirs(tmpphotos, exist_ok=True)
        rep = { STATUS : OK }
        EKOX(rep)
        return json.dumps(rep)
        
    @cherrypy.expose
    def uploadPhoto(self, ufile):
        '''receive a picture
        '''
        EKOT("REQ uploadPhoto")
        try :
            #filename  = os.path.basename(cherrypy.request.headers['x-filename'])
            #EKOX(dir(cherrypy.request))
            #EKOX(cherrypy.request.body)
            body = cherrypy.request.body
            EKOX(body)
            filename = ufile.filename
            EKOX(filename)
            destination = os.path.join(tmpphotos, filename)
            EKOX(destination)
            ext = os.path.splitext(filename)[1]
            size = 0
            with open(destination, 'wb') as out:
                while True:
                    data = ufile.file.read(8192)
                    #EKOX(len(data))
                    if not data:
                        break
                    out.write(data)
                    size += len(data)
            EKOX("done %d" % size)
            """
            with open(destination, 'wb') as f:
                EKOX(f)
                ff = body
                shutil.copyfileobj(ff, f)
                #EKO()
            """
            EKOX(ext)
            dd = os.path.join(dirphotos, "image_%04d%s" % (self.no_image, ext))
            EKOX(dd)
            os.makedirs(dirphotos, exist_ok=True)
            shutil.copyfile(destination, dd)

            
            EKOX(self.no_image)
            self.no_image += 1
            rep = { STATUS : OK }
            try :
                image = Image.open(dd)
                EKOT("good image")
            except Exception as e:
                EKOX(e)
                rep = { STATUS : FAILED }
            
        except Exception as e :
            EKOX(e)
            rep = { STATUS : FAILED }
        EKOX(rep)
        return json.dumps(rep)
        
    @cherrypy.expose
    def exit(self):
        EKOT("REQ exit")
        self.queue.put({ 'exit' : True } )
        #sys.exit(0)
        cherrypy.engine.exit()

        
    @cherrypy.expose
    def upload(self):
        '''Handle non-multipart upload'''
        EKOT("REQ uploading")
        def findnewfile() :
            for i in range(10000) :
                rt = self.reconstructor.args.dataroot
                pp = os.path.join(rt, "sequences/video_%05d.mp4" % i)
                if not os.path.exists(pp) :
                    return pp
        
        filename  = os.path.basename(cherrypy.request.headers['X_Filename'])
        destination = os.path.join(tmpphotos, filename)
        EKOX(destination)
        EKOX(cherrypy.request.body)
        EKOX(findnewfile())
        with open(destination, 'wb') as f:
            shutil.copyfileobj(cherrypy.request.body, f)
        #EKO()
        shutil.copyfile(destination, findnewfile())
        EKOT("done, file copied locally")
        self.queue.put({ 'file' : destination} )
        EKO()
        rep = { STATUS : OK }        
        return json.dumps(rep)
        
config2 = {
    "dry" : (False, " true : will not run the reconstructor"),
    "gitinfo" : "info"
}

if __name__ == '__main__':

    args = process.parse()
    args = process.parse(vars(args), config2)
    args = process.parse(vars(args), iterative_fit.config(args), end=True)
    app = App(args)
    cherrypy.log.error_log.propagate = False
    cherrypy.log.access_log.propagate = False
    EKO()
    cherrypy.quickstart(app, '/', config)
    EKOT("end server", n=LOG)

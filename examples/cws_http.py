#!/usr/bin/python3
import cherrypy
from jinja2 import Template
import pickle
import gzip
import sys
import urllib.parse

class CWS (object) :
    exposed = True
    def GET(self, s = None):
        output = segger.predict([s.strip()])[0]
        output = ' '.join(output)
        return output
    

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print("""oneseg Chinese Word Segmentor
    example: %s model.gz"""%(sys.argv[0]))
        exit()


    model_filename = sys.argv[1]
    segger = pickle.load(gzip.open(model_filename))

    cherrypy.config["tools.encode.on"] = True
    cherrypy.config["tools.encode.encoding"] = "utf8"

    #cherrypy.config.update({"server.socket_host":"10.0.2.15",
    #    "server.socket_port":8080})
    cherrypy.config.update("server.config")
    cherrypy.tree.mount(CWS(),'/cws',
            {'/':{'request.dispatch': cherrypy.dispatch.MethodDispatcher()}},)
    cherrypy.engine.start()
    cherrypy.engine.block()

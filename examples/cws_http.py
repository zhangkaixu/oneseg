#!/usr/bin/python3
import cherrypy
from jinja2 import Template
import pickle
import gzip
import sys
import urllib.parse

class CWS (object) :
    def seg(self, s = None):
        index_file = open('examples/index.html').read()
        if s is not None :
            output = segger.predict([s.strip()])[0]
            output = ' '.join(output)
        else :
            s = '深圳，我来啦！'
            output = ''
        index = Template(index_file).render({
            'input' : s,
            'output' : output,
            })
        return index
    exposed = True
    def GET(self, raw = None):
        return self.seg(raw)
    exposed = True
    def POST(self, raw = None):
        return self.seg(raw)

    

if __name__ == '__main__':
    if len(sys.argv) < 2 :
        print("""oneseg Chinese Word Segmentor
    example: %s model.gz"""%(sys.argv[0]))
        exit()

    model_filename = sys.argv[1]
    segger = pickle.load(gzip.open(model_filename))

    cherrypy.config["tools.encode.on"] = True
    cherrypy.config["tools.encode.encoding"] = "utf8"

    cherrypy.config.update("server.config")
    cherrypy.tree.mount(CWS(),'/cws',
            {'/':{'request.dispatch': cherrypy.dispatch.MethodDispatcher()}},)
    cherrypy.engine.start()
    cherrypy.engine.block()

#!/usr/bin/python3
import cherrypy
from jinja2 import Template
import sys
import json

def save_anno(filename, corpus):
    f = open(filename,'w')
    for rec in corpus :
        json.dump(rec, f, ensure_ascii = False)
        f.write('\n')
    f.close()

def load_anno(filename):
    corpus = []
    for line in open(filename):
        rec = json.loads(line)
        corpus.append(rec)
    return corpus

class Annotator :
    def __init__(self, filename):
        self.corpus = load_anno(filename)
        self.filename = filename
    def find_ind(self, index):
        return index % len(self.corpus)
    exposed = True
    def GET(self, index, anno = None):
        index = int(index)
        if anno :
            if anno == 'save' :
                save_anno(self.filename, self.corpus)
                pass
            else :
                self.corpus[index]['anno'] = list(anno)
                index = self.find_ind(index + 1)
        index = self.find_ind(index)
        rec = self.corpus[index]
        raw = json.dumps(list(rec['raw']), ensure_ascii = False)
        anno = json.dumps(list(rec['anno']))
        margin = rec.get('margin',0)
        margins = rec.get('margins',[])
        margins = json.dumps(margins)
        predicted = json.dumps(list(rec.get('predicted',[])))
        index_file = open('oneseg/annotate/index.html').read()
        index = Template(index_file).render({
            'sentence_id' : index,
            'raw' : raw,
            'anno' : anno,
            'margin' : margin,
            'margins' : margins,
            'predicted' : predicted,
            })
        return index

if __name__ == '__main__':
    if len(sys.argv) == 1 :
        exit()
    filename = sys.argv[1]

    #cherrypy.config["tools.encode.on"] = True
    #cherrypy.config["tools.encode.encoding"] = "utf8"

    cherrypy.config.update("server.config")
    cherrypy.tree.mount(Annotator(filename),'/annotate',
            {'/':{'request.dispatch': cherrypy.dispatch.MethodDispatcher()}},)
    cherrypy.engine.start()
    cherrypy.engine.block()

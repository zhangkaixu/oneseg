#!/usr/bin/python3
import re
import os
import sys


es_map = { 
        '&mdash;' : '—',
        "&alpha;" : 'α',
        "&bull;" : '•',
        "&hellip;" : '…',
        "&le;" : '<',
        "&lsquo;" : '‘',
        "&mu;" : 'μ',
        "&rsquo;" : '’',
        '&ldquo;' : '“',
        '&rdquo;' : '”',
        '&quot;':'"',
        '&apos;':"'",
        '&amp;':'&',
        '&lt;':'<',
        '&gt;':'>',
        '&nbsp;':' ',
        '&iexcl;':'¡',
        '&cent;':'¢',
        '&pound;':'£',
        '&curren;':'¤',
        '&yen;':'¥',
        '&brvbar;':'¦',
        '&sect;':'§',
        '&uml;':'¨',
        '&copy;':'©',
        '&ordf;':'ª',
        '&laquo;':'«',
        '&not;':'¬',
        #'&shy;':'',
        '&reg;':'®',
        '&macr;':'¯',
        '&deg;':'°',
        '&plusmn;':'±',
        '&sup2;':'²',
        '&sup3;':'³',
        '&acute;':'´',
        '&micro;':'µ',
        '&para;':'¶',
        '&middot;':'·',
        '&cedil;':'¸',
        '&sup1;':'¹',
        '&ordm;':'º',
        '&raquo;':'»',
        '&frac14;':'¼',
        '&frac12;':'½',
        '&frac34;':'¾',
        '&iquest;':'¿',
        '&times;':'×',
        '&divide;':'÷',
        '&Agrave;':'À',
        '&Aacute;':'Á',
        '&Acirc;':'Â',
        '&Atilde;':'Ã',
        '&Auml;':'Ä',
        '&Aring;':'Å',
        '&AElig;':'Æ',
        '&Ccedil;':'Ç',
        '&Egrave;':'È',
        '&Eacute;':'É',
        '&Ecirc;':'Ê',
        '&Euml;':'Ë',
        '&Igrave;':'Ì',
        '&Iacute;':'Í',
        '&Icirc;':'Î',
        '&Iuml;':'Ï',
        '&ETH;':'Ð',
        '&Ntilde;':'Ñ',
        '&Ograve;':'Ò',
        '&Oacute;':'Ó',
        '&Ocirc;':'Ô',
        '&Otilde;':'Õ',
        '&Ouml;':'Ö',
        '&Oslash;':'Ø',
        '&Ugrave;':'Ù',
        '&Uacute;':'Ú',
        '&Ucirc;':'Û',
        '&Uuml;':'Ü',
        '&Yacute;':'Ý',
        '&THORN;':'Þ',
        '&szlig;':'ß',
        '&agrave;':'à',
        '&aacute;':'á',
        '&acirc;':'â',
        '&atilde;':'ã',
        '&auml;':'ä',
        '&aring;':'å',
        '&aelig;':'æ',
        '&ccedil;':'ç',
        '&egrave;':'è',
        '&eacute;':'é',
        '&ecirc;':'ê',
        '&euml;':'ë',
        '&igrave;':'ì',
        '&iacute;':'í',
        '&icirc;':'î',
        '&iuml;':'ï',
        '&eth;':'ð',
        '&ntilde;':'ñ',
        '&ograve;':'ò',
        '&oacute;':'ó',
        '&ocirc;':'ô',
        '&otilde;':'õ',
        '&ouml;':'ö',
        '&oslash;':'ø',
        '&ugrave;':'ù',
        '&uacute;':'ú',
        '&ucirc;':'û',
        '&uuml;':'ü',
        '&yacute;':'ý',
        '&thorn;':'þ',
        '&yuml;':'ÿ',

        }
def esf(s):
    s =s.group()
    #print(s)
    return es_map.get(s, s)

if __name__ == '__main__':
    files = os.listdir('data')
    files = [f for f in files if int(f)<95001 and int(f) > 90000]
    #print(files)
    
    #doc_ids = ['99133']
    doc_ids = list(sorted(files))

    for doc_id in doc_ids :
        print(doc_id, file = sys.stderr)
        doc = open('data/'+doc_id).read()
        content = re.search('<section id="articleContent">(.*?)</section>',doc, flags= re.DOTALL)
        if not content : continue
        content = content.group(1)
        #titleArea = re.search('<section class="titleArea">(.*?)</section>',doc, flags= re.DOTALL).group(1)
        #tagArea = re.search('<section class="tag">(.*?)</section>',doc, flags= re.DOTALL).group(1)

        content = re.sub('<div[^>]*>.*?</div>','', content,flags = re.DOTALL)
        content = re.sub('</?(b|a|q|s|u|f|i)[^\>]*>','', content,flags = re.DOTALL)
        content = re.sub('\&[^\;]+\;', esf, content,flags = re.DOTALL)

        for i, x in enumerate(re.findall('<p>(.*?)</p>',content, flags= re.DOTALL)):
            x = re.sub('</?(!|[a-z])[^\>]*>','', x,flags = re.DOTALL)
            x = x.strip()
            sentences = []
            for j,s in enumerate(re.split('([！？。]+”?)', x)):
                if j%2 == 0 :
                    sentences.append(s)
                else :
                    sentences[-1]+=(s)
            sentences = [s for s in sentences if s]
            for j, s in enumerate(sentences):
                pass
                print('%s:%s:%s'%(doc_id,i+1,j+1),s)


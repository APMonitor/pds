import requests
import pandas as pd
import os
from lxml.html import fromstring
from bs4 import BeautifulSoup
import json

url_base = 'https://www.apmonitor.com/pds/index.php/Main/'
pages = pd.read_csv('pages.csv')
print(pages.head())

def get_file(page_name):
    url = url_base + page_name
    r = requests.get(url, allow_redirects=True)
    tree = fromstring(r.content)
    title = tree.findtext('.//title')
    soup = BeautifulSoup(r.text,features='lxml')
    metas = soup.find_all('meta')
    desc = [meta.attrs['content'] for meta in metas if 'name' in meta.attrs and meta.attrs['name'] == 'description' ][0]
    i = 1
    get_pages = True
    while True:
        url = url_base + page_name + '?action=sourceblock&num=' + str(i)
        r = requests.get(url, allow_redirects=True)
        pn = page_name+'_'+str(i)+'.py'
        open(pn, 'wb').write(r.content)
        sz = os.path.getsize(pn)
        with open(pn) as f:
            # detect no additional source blocks
            if 'gtag.js' in f.read():
                f.close()
                os.remove(pn)
                get_pages=False
        if i>=100 or get_pages==False:
            n = i-1
            return title, desc, n
        i+=1

for p in pages['Title']:
    try:
        os.mkdir(p)
    except:
        pass
    os.chdir(p)

    title,desc,n = get_file(p)
    url = url_base + p

    f = open('README.md','w')
    f.write('### Machine Learning for Engineers: [' + p + ']('+(url)+')\n')
    f.write('- ['+title+']('+url+')\n')
    f.write(' - Source Blocks: ' + str(n)+'\n')
    f.write(' - Description: '+desc+'\n')
    f.write('- [Course Overview](https://apmonitor.com/pds)\n')
    f.write('- [Course Schedule](https://apmonitor.com/pds/index.php/Main/CourseSchedule)\n')
    f.close()
    
    if n>=1:
        nb = {'nbformat': 4, 'nbformat_minor': 2, 'cells': [], 'metadata': 
           {"kernelspec": {"display_name": "Python 3","language": \
                           "python", "name": "python3"}}}

        # write header markdown
        s = open('README.md','r')
        cell = {}
        cell['metadata'] = {}
        cell['source'] = [s.read()]
        cell['cell_type'] = 'markdown'
        nb['cells'].append(cell)

        for i in range(n):
            s = open(p+'_'+str(i+1)+'.py','r')
            cell = {}
            cell['metadata'] = {}
            cell['outputs'] = []
            cell['source'] = [s.read()]
            cell['execution_count'] = None
            cell['cell_type'] = 'code'
            nb['cells'].append(cell)

        with open(p+'.ipynb', 'w') as jnb:
            jnb.write(json.dumps(nb))
    
    os.chdir('../')
    


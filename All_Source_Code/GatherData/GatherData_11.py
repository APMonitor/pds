import requests
url = 'http://apmonitor.com/pds/index.php/Main/GatherData?action=print'
page = requests.get(url)
import urllib.request

# download image
url = 'http://apmonitor.com/pds/uploads/Main/students_walking.jpg'
urllib.request.urlretrieve(url, 'students_walking.jpg')
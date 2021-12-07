url = 'http://apmonitor.com/pds/index.php/Main/GatherData'
table = pd.read_html(url)
print(table)
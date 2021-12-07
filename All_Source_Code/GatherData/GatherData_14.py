for link in soup.find_all('a'):
    print('Link Text: {}'.format(link.text))
    print('href: {}'.format(link.get('href')))
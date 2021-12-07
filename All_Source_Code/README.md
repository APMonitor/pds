### Source Code: Machine Learning for Engineers
- [Course Overview](https://apmonitor.com/pds)
- [Course Schedule](https://apmonitor.com/pds/index.php/Main/CourseSchedule)

The `get_source.py` script uses BeautifulSoup4 to web scrape the source code for each page listed in `pages.csv`. The source code scripts are saved as individual text `.py` files and combined as a Jupyter Notebook `.ipynb`. The files are saved to a directory with the title address of the scraped webpage. A `README.md` summary is created for each folder. The source files in this archive may be outdated if the webpage content is recently changed. The files can be updated by running `get_source.py` to refresh the content.
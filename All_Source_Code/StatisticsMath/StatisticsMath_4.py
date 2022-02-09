from pandas_profiling import ProfileReport

profile = ProfileReport(data, explorative=True, minimal=False)
try:
   profile.to_widgets()         # view as widget in Notebook
except:
   profile.to_notebook_iframe() # view as html iframe in Notebook
profile.to_file('data.html')    # save as html file
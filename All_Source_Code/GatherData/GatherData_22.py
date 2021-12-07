dx.append(dy)\
  .sort_values(by='Time')\
  .drop_duplicates(subset='Time')\
  .reset_index(drop=True)
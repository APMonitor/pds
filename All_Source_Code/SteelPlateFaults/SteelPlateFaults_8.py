# Display probabilities, with the most likely label
#   highlighted and the actual label displayed 
yp['Actual fault'] = actual_label.values
yp.style.highlight_max(axis=1)
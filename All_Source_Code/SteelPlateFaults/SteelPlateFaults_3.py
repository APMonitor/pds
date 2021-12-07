# keras packages
from keras.models import Sequential
from keras.layers import Dense

# scikit-learn packages
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.feature_selection import SelectKBest,chi2
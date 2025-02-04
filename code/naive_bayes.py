import pandas 
import numpy

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# each encoder for each column
weather_encoder = LabelEncoder()
temp_encoder = LabelEncoder()
play_encoder = LabelEncoder()
# model
model = MultinomialNB()

def test_encode(x, y):
    array = [x, y]
    dft = pandas.DataFrame([array], columns=['Weather', 'Temperature'])
    print(dft)
    dft['Temperature'] = temp_encoder.transform(dft['Temperature'])
    dft['Weather'] = weather_encoder.transform(dft['Weather'])
    return dft

# encode
df = pandas.read_csv('../data/weather.csv')
df['Weather'] = weather_encoder.fit_transform(df['Weather'])
df['Temperature'] = temp_encoder.fit_transform(df['Temperature'])
df['Play'] = play_encoder.fit_transform(df['Play'])

#print(df)

# feature and output 
features = df.iloc[:, 0:2]
outputs = df.iloc[:, 2:]

# train and test
X_train, X_test, y_train, y_test = train_test_split(features, outputs, train_size=0.8)

#print(weather_encoder.inverse_transform(numpy.asarray(X_test['Weather'])))
#print(temp_encoder.inverse_transform(numpy.asarray(X_test['Temperature'])))

# flatten y_train and y_test to 1d arrays because model expects
y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

# fit
model.fit(X=X_train, y=y_train)

'''
# predict
predict = model.predict(X_test)
print(play_encoder.inverse_transform(predict)) 
'''

# PREDICT RAINY AND COOL
test_df = test_encode('Rainy', 'Cool')
predict = model.predict(test_df)
print(play_encoder.inverse_transform(predict))








'''
def test_encode(x, y):
    w = weather_encoder.transform([x])
    t = temp_encoder.transform([y])
    tdata = numpy.concatenate((w, t))
    tempdf = pandas.DataFrame([tdata], columns=['Weather', 'Temperature'])
    print(tempdf)
    #return tempdf
'''

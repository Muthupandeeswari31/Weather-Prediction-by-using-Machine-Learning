import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.metrics import confusion_matrix

from tensorflow import keras
from keras import layers
import telepot
from bs4 import BeautifulSoup
import requests
headers = {
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}



token = '5834293262:AAHDRxoQE895V88ONBvKrJpIKIyadsda5lI' # telegram token
receiver_id = 5161631269 # https://api.telegram.org/bot5834293262:AAHDRxoQE895V88ONBvKrJpIKIyadsda5lI/getUpdates

bot = telepot.Bot(token)


def weather(city):
	city = city.replace(" ", "+")
	res = requests.get(
		f'https://www.google.com/search?q={city}&oq={city}&aqs=chrome.0.35i39l2j0l4j46j69i60.6128j1j7&sourceid=chrome&ie=UTF-8', headers=headers)
	print("Searching...\n")
	soup = BeautifulSoup(res.text, 'html.parser')
	location = soup.select('#wob_loc')[0].getText().strip()
	time = soup.select('#wob_dts')[0].getText().strip()
	info = soup.select('#wob_dc')[0].getText().strip()
	weather = soup.select('#wob_tm')[0].getText().strip()
	print(location)
	print(time)
	TM=time[15:17]
	print(info)
	print(weather+"°C")
	SV=weather
	return SV,TM

df = pd.read_csv('seattle-weather.csv')

#Convert date format to int format
df['date'] = pd.to_datetime(df['date'])
df['date'] = pd.to_numeric(df['date'])

df['weather'] = df['weather'].replace(['drizzle','rain','sun','snow','fog'],[0,1,2,3,4])

X = df.copy()
y = X.pop('weather')

preprocessor = make_column_transformer(
    (StandardScaler(),
     make_column_selector(dtype_include=np.number)),
    (OneHotEncoder(sparse=False),
     make_column_selector(dtype_include=np.number)),
)

X = preprocessor.fit_transform(X)
y = y

input_shape = [X.shape[1]]
print("Input shape: {}".format(input_shape))
#activation = RELU or softmax
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dense(128, activation='relu'),    
    layers.Dense(64, activation='relu'),
    layers.Dense(1),
])

#Model compilation
#optimizer = ADAM or sgd
#loss = mean_squared_error(MSE) or mae or binary_crossentropy
model.compile(
    optimizer='adam',
    loss='mse'
)

#early stopping
callback = keras.callbacks.EarlyStopping(
    monitor='loss', 
    patience=3
)
#Training
history = model.fit(
    X, y,
    batch_size=128,
    epochs=800,
    callbacks=[callback]
)

#Display loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

predictions = model.predict(X)

#reshape to display, round to convert to binary
predictions = predictions.reshape(predictions.size)
predictions = np.round_(predictions)
print("reshape to display, round to convert to binary")
bot.sendMessage(receiver_id, 'Weather Prediction') # send a activation message to telegram receiver id
bot.sendMessage(receiver_id,str(predictions)) # send a activation message to telegram receiver id

#Convert to text format
df['weather'] = df['weather'].replace([0,1,2,3,4],['drizzle','rain','sun','snow','fog'])
predictions = predictions.astype(str)
predictions = np.char.replace(predictions,'-0.0','drizzle')
predictions = np.char.replace(predictions,'0.0','drizzle')
predictions = np.char.replace(predictions,'1.0','rain')
predictions = np.char.replace(predictions,'2.0','sun')
predictions = np.char.replace(predictions,'3.0','snow')
predictions = np.char.replace(predictions,'4.0','fog')

#Real weather
plt.figure(figsize=(16,6))
plt.title("Real weather")
sns.scatterplot(x=df['date'], y=df['temp_max'],hue=df['weather'])

#Predicted weather
plt.figure(figsize=(16,6))
plt.title("Predicted weather")
sns.scatterplot(x=df['date'], y=df['temp_max'],hue=predictions)
plt.savefig('test.jpg')
bot.sendMessage(receiver_id, 'Weather Prediction') # send a activation message to telegram receiver id
bot.sendPhoto(receiver_id, photo=open('test.jpg', 'rb')) # send message to telegram


res = pd.DataFrame({
    "df":df['weather'],
    "pred":predictions
})

res['df'] = res['df'].replace(['drizzle','rain','sun','snow','fog'],[0,1,2,3,4])
res['pred'] = res['pred'].replace(['drizzle','rain','sun','snow','fog'],[0,1,2,3,4])

def plot_confusion_matrix(cm, classes,
                          normalize = False,
                          title = 'Confusion matrix"',
                          cmap = plt.cm.Blues) :
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :
        plt.text(j, i, cm[i, j],
                 horizontalalignment = 'center',
                 color = 'white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Show metrics 
def show_metrics():
    tp = cm[1,1]
    fn = cm[1,0]
    fp = cm[0,1]
    tn = cm[0,0]
    print('Accuracy  =     {:.3f}'.format((tp+tn)/(tp+tn+fp+fn)))
    print('Precision =     {:.3f}'.format(tp/(tp+fp)))
    print('Recall    =     {:.3f}'.format(tp/(tp+fn)))
    print('F1_score  =     {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/
                                                 ((tp/(tp+fp))+(tp/(tp+fn))))))

cm = confusion_matrix(res['df'], res['pred'])
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cm, 
                      classes=class_names, 
                      title='Logistic Confusion matrix')

plt.show()

show_metrics()

print('yesterday ')

##bot.sendMessage(receiver_id,'yesterday  ---->') # send a activation message to telegram receiver id
##
##bot.sendMessage(receiver_id,str(predictions[0])) # send a activation message to telegram receiver id
##
##bot.sendMessage(receiver_id,'today ----->') # send a activation message to telegram receiver id
##
##bot.sendMessage(receiver_id,str(predictions[1])) # send a activation message to telegram receiver id
##
##bot.sendMessage(receiver_id,'tomorrow------>') # send a activation message to telegram receiver id
##
##bot.sendMessage(receiver_id,str(predictions[2])) # send a activation message to telegram receiver id
##
##


city = "Madurai"
city = city+" weather"
Music_1,Music_2=weather(city)


print("Have a Nice Day:)")

# This code is contributed by adityatri
print("Weather")

print(Music_1)
print("Weather")
bot.sendMessage(receiver_id,'Current Weather') # send a activation message to telegram receiver id

bot.sendMessage(receiver_id,str(Music_1)) # send a activation message to telegram receiver id



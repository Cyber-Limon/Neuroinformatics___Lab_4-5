import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import LSTM, Dense
from tkinter import *
from PIL import Image, ImageDraw

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat = keras.utils.to_categorical(y_test, 10)

model = keras.Sequential([
    LSTM(500, input_shape=(28, 28)),
    Dense(10, activation='softmax')])

print(model.summary())

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

log = model.fit(x_train, y_train_cat, batch_size=32, epochs=10, validation_split=0.2)

model.evaluate(x_test, y_test_cat)


# Paint #


def press(event):
    global arr
    arr = [(event.x, event.y)]


def draw(event):
    arr.append((event.x, event.y))

    canvas.create_line(arr[-2], arr[-1], fill=color, width=brush_size, smooth=True)
    draw_img.line((arr[-2], arr[-1]), fill=color, width=brush_size)

    cord = arr[-1]
    x1, y1 = cord[0] - brush_size / 2, cord[1] - brush_size / 2
    x2, y2 = cord[0] + brush_size / 2, cord[1] + brush_size / 2

    canvas.create_oval(x1, y1, x2, y2, fill=color, width=0)
    draw_img.ellipse((x1, y1, x2, y2), fill=color, width=0)

    image = image1.resize((28, 28))
    image.save('image_number.png')
    image = cv2.imread('image_number.png', cv2.IMREAD_GRAYSCALE)
    image = np.asarray(image)
    image = image / 255

    rec_img = np.expand_dims(image, axis=0)
    rec_res = model.predict(rec_img)
    rec_max = max(rec_res[0])

    for i in range(10):

        if rec_res[0][i] == rec_max:
            label[i].config(fg='red')
        else:
            label[i].config(fg='black')

        label[i].config(text=str(i) + ': ' + str(np.round(rec_res[0][i] * 100, 1)) + '%')


def clear_canvas():
    canvas.delete('all')
    canvas['bg'] = 'black'
    draw_img.rectangle((0, 0, 500, 500), fill='black', width=0)

    for i in range(10):
        label[i].config(text=str(i) + ': 0%')


root = Tk()
root.title('Paint')
root.geometry('500x500')
root.resizable(False, False)

brush_size = 30
color = 'white'

root.columnconfigure(6, weight=1)
root.rowconfigure(3, weight=1)

canvas = Canvas(root, bg='black')
canvas.grid(row=1, column=0, rowspan=3, columnspan=6, padx=5, pady=5, sticky=W+E+N+S)

canvas.bind('<Button-1>', press)
canvas.bind('<B1-Motion>', draw)

image1 = Image.new('RGB', (500, 500), 'black')
draw_img = ImageDraw.Draw(image1)

Button(root, text='Очистить', width=10, height=4, command=clear_canvas).grid(row=4, column=0, rowspan=2)

label = []

for i in range(10):
    if i < 5:
        r = 4
        c = i + 1
    else:
        r = 5
        c = i - r + 1

    label.append(Label(root, text=str(i) + ': 0%', width=7, font=('Times New Roman Bold', 15)))
    label[i].grid(row=r, column=c)

root.mainloop()

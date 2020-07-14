from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow и tf.keras
import tensorflow as tf
from tensorflow import keras
import os

# Вспомогательные библиотеки
import numpy as np
import matplotlib.pyplot as plt

#функция загрузки меток из файлов
def load_labels(labels,pth):
    with open(pth) as f:
        text=f.readlines()
        for i in text[1:]:
            s=i.split(' ')
            s=s[1]
            labels.append(int(s[:len(s)-1]))

#упорядоченный список файлов
def list_file(flist,spisok2,pth):
    spisok=[]
    for i in flist:
        spisok.append(int(i[:len(i)-4]))
    spisok.sort()
    for i in spisok:
        spisok2.append(str(i)+'.txt')

#
def load_data(data,pth,flist):
    n=0
    for j in flist:
        with open(pth+j) as f:
            slist = []
            data.append([])
            text = f.readlines()
            for i in text:
                s = i.split('	')
                s = s[1]
                slist.append(float(s[:len(s) - 1]))
            data[n].append(slist)
        n = n + 1



print(tf.__version__)

class_names = ['Noize','Event']

#метки для тренировочного массива
train_labels=[]
pth='y:\\Work\Мурыськин_Алексей\\NN\\Katav\\train\\Event.txt'
pth2='y:\\Work\Мурыськин_Алексей\\NN\\Katav\\train\\Noize.txt'
load_labels(train_labels,pth)
load_labels(train_labels,pth2)

#метки для тест-массива
test_labels=[]
pth='y:\\Work\Мурыськин_Алексей\\NN\\Katav\\test\\Event.txt'
pth2='y:\\Work\Мурыськин_Алексей\\NN\\Katav\\test\\Noize.txt'
load_labels(test_labels,pth)
load_labels(test_labels,pth2)

#получаем упорядоченный список файлов
flist=[]
pth='y:\\Work\\Мурыськин_Алексей\\NN\\Katav\\train\\Event\\'
flist=os.listdir(pth)
spisok=[]
list_file(flist, spisok,pth)

flist_n=[]
pth_n='y:\\Work\\Мурыськин_Алексей\\NN\\Katav\\train\\Noize\\'
flist_n=os.listdir(pth_n)
spisok_n=[]
list_file(flist_n, spisok_n,pth_n)

#аналогично тест-массив
flist_t=[]
pth_t='y:\\Work\\Мурыськин_Алексей\\NN\\Katav\\test\\Event\\'
flist_t=os.listdir(pth_t)
spisok_t=[]
list_file(flist_t, spisok_t,pth_t)

flist_tn=[]
pth_tn='y:\\Work\\Мурыськин_Алексей\\NN\\Katav\\test\\Noize\\'
flist_tn=os.listdir(pth_tn)
spisok_tn=[]
list_file(flist_tn, spisok_tn,pth_tn)

#загружаем данные
train_envelop=[]
load_data(train_envelop,pth,spisok)
load_data(train_envelop,pth_n,spisok_n)
test_envelop=[]
load_data(test_envelop,pth_t,spisok_t)
load_data(test_envelop,pth_tn,spisok_tn)



with open('y:\\Work\\Мурыськин_Алексей\\NN\\Katav\\train\\res.txt','w') as f:
    for i in train_envelop:
        for j in i:
            for k in j:
                f.write(str(k)+' ')
            f.write('\n')

with open('y:\\Work\\Мурыськин_Алексей\\NN\\Katav\\train\\test.txt','w') as f:
    for i in test_envelop:
        for j in i:
            for k in j:
                f.write(str(k)+' ')
            f.write('\n')


train2=np.loadtxt('y:\\Work\\Мурыськин_Алексей\\NN\\Katav\\train\\res.txt')
test2=np.loadtxt('y:\\Work\\Мурыськин_Алексей\\NN\\Katav\\train\\test.txt')
print(train2.shape)
print(test2.shape)
train_labels=np.array(train_labels)
test_labels=np.array(test_labels)
print(test_labels.shape)

#настройка модели
model = keras.Sequential([
    keras.layers.Dense(128,input_dim=3000, activation='sigmoid'),
    keras.layers.Dense(64,input_dim=3000, activation='sigmoid'),
    keras.layers.Dense(2, activation='softmax')
])
#компиляция модели
model.compile(optimizer='RMSprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#тренировка модели
model.fit(train2, train_labels, epochs=10)

#точность
test_loss, test_acc = model.evaluate(test2,  test_labels, verbose=2)
print('\nТочность на проверочных данных:', test_acc)
print('\nПотери:', test_loss)

#предиктивное моделирование, пытаемся угадать вещи на картинках
predictions = model.predict(test2)
j=0
with open('y:\\Work\\Мурыськин_Алексей\\NN\\Katav\\res_NN_mod1.txt','w') as f:
    f.write('N'+' '+'test'+' '+'NN'+'\n')
    for i in test_labels:
        f.write(str(j)+' '+str(i)+' '+str(predictions[j])+'\n')
        j=j+1


mass_env=np.concatenate((train2,test2),0)
print(mass_env.shape)
mass_labels=np.concatenate((train_labels,test_labels),0)
print(mass_labels.shape)




#настройка модели
model2 = keras.Sequential([
    keras.layers.Dense(128,input_dim=3000, activation='sigmoid'),
    keras.layers.Dense(64,input_dim=3000, activation='sigmoid'),
    keras.layers.Dense(2, activation='softmax')
])

#компиляция модели
model2.compile(optimizer='RMSprop',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#тренировка модели
model2.fit(mass_env, mass_labels, epochs=2)

#точность
test_loss, test_acc = model2.evaluate(train2,  train_labels, verbose=2)
print('\nТочность на проверочных данных (модель 2):', test_acc)
print('\nПотери (модель 2):', test_loss)

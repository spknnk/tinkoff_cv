#Tinkoff CV Assignment

## Задание
Дано: архив с фотографиями и файл, в котором написано, есть ли Олег на фото или нет.
Напишите питоновский скрипт, который берет на вход имя архива с фотографиями и возвращает список имен файлов с вероятностями, что на фото есть Олег. 

**В ходе работы были использованы следующие средства:**
- Catalyst (train/infer cycle, Ralamb optimizer, OneCycleLRWithWarmup scheduler)
- Tensorboard (for training process visualization)
- Albumentations (for image transforms)
- MTCNN (for head crop)
- Facenet (for pretrained face embeddings)

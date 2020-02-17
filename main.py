from classes import *
from pathlib import Path

import catalyst
from catalyst import utils
from catalyst.utils import imread

from catalyst.dl import SupervisedRunner
from catalyst.dl.callbacks import EarlyStoppingCallback, LRFinder, CheckpointCallback, InferCallback, F1ScoreCallback, ConfusionMatrixCallback, TensorboardLogger
from catalyst.dl.callbacks.metrics.ppv_tpr_f1 import  PrecisionRecallF1ScoreCallback

from catalyst.contrib.nn import OneCycleLRWithWarmup
from catalyst.contrib.nn.optimizers.ralamb import Ralamb 

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "" - CPU, "0" - 1 GPU, "0,1" - MultiGPU

SEED = 42
num_workers = 0
batch_size = 20
num_epochs = 100 # early stopping 

utils.set_global_seed(SEED)
utils.prepare_cudnn(deterministic=True)

IMGDIR = "../train"
LOGDIR = "./logs/classification"

train = pd.read_csv("../train_labels.csv")
train.drop(index=[13, 14, 52, 131], inplace=True) #bad mtcnn prep

train.to_csv("../train_labels_mod.csv", index=False) 
train = pd.read_csv("../train_labels_mod.csv") #index reset

train_ids, valid_ids = train_test_split(train.index, random_state=SEED, stratify=train['label'], test_size=0.25)

detector = MTCNN()
bounding_box_shift = 15

train_dataset = TinkoffDataset(csv_file='../train_labels_mod.csv', root_dir=f"{IMGDIR}/", 
                            img_ids=train_ids, mtcnn_model=detector, sh=bounding_box_shift, transform=train_val_transfroms()[0])
valid_dataset = TinkoffDataset(csv_file='../train_labels_mod.csv', root_dir=f"{IMGDIR}/", 
                            img_ids=valid_ids, mtcnn_model=detector, sh=bounding_box_shift, transform=train_val_transfroms()[1])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

loaders = {"train": train_loader, "valid": valid_loader}

model = FNInceptionResnetV1()

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = Ralamb(model.parameters())
scheduler = OneCycleLRWithWarmup(
    optimizer, 
    num_steps=25, 
    lr_range=(0.001, 0.0001),
    warmup_steps=1
)

runner = SupervisedRunner(device = device)

runner.train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    loaders=loaders,
    callbacks=[EarlyStoppingCallback(patience=5), TensorboardLogger()],
    logdir=LOGDIR,
    num_epochs=num_epochs,
    verbose = True,
)

gc.collect()

model.load_state_dict(torch.load(f"{LOGDIR}/checkpoints/best.pth")['model_state_dict'])
model.to(device)
model.eval()

with zipfile.ZipFile('../test.zip', 'r') as f:
    for name in f.namelist():
        if ".jpg" not in name: continue
        data = f.read(name)
        name = name[5:] # w/o "test/"
        img = cv2.imdecode(np.frombuffer(data, np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        result = detector.detect_faces(img)
        num_faces = len(result)

        if not num_faces:
            print(f"{name},0")   
            continue
            
        logits = np.zeros(num_faces)
        for face_id in range(num_faces):
                bb = np.array(result[face_id]['box']).clip(min=0) # bounding boxes
                face = img[max(0, bb[1]-bounding_box_shift):bb[1]+bb[3]+bounding_box_shift, 
                           max(0, bb[0]-bounding_box_shift):bb[0]+bb[2]+bounding_box_shift, :]
                
                aug = train_val_transfroms()[1](image=face)
                face = aug['image'].unsqueeze(0)
                
                logits[face_id] = model(face.to(device))
        logits = sigmoid(logits)
        print(f"{name},{np.max(logits)}")        
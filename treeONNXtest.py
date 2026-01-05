#simple testing script to see if ONNX model can work, will also eventually become the FastAPI endpoint for the simple app

import numpy as np
import onnxruntime as ort
from PIL import Image
import cv2

#just reuse our transforms:

IMG_SIZE = 320
#CLAHE (Contrast-Limited Adaptive Histogram Equalization) transform is really good on tress since trees contrast greatly with the background, this "normalizes" the contrast making it more visible for the model

class CLAHETransform:
    """ Enhance image contrast with CLAHE"""

    def __init__(self, clip_limit = 2.0, tile_grid_size = (8,8)):
        self.clip_limit = clip_limit
        self.title_grid_szie = tile_grid_size

    #actual call we will use to apply the transforms
    def __call__(self, img):
        img_np = np.array(img)

        #create a LAB image since RGB does not work with CLAHE
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        #get luminance and a,b channels
        l,a,b = cv2.split(lab)

        #apply clahe on the luminance channel and merge it back
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.title_grid_szie)
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))

        finalImg = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        return Image.fromarray(finalImg)

#create a letterbox resize to ensure aspect ratio is maintained
class LetterBoxResize:
    """ Resize an image with letterboxing to keep aspect ratio"""

    def __init__(self, size = IMG_SIZE):
        self.size = size 
    
    def __call__(self, img):
        img.thumbnail((self.size, self.size), Image.Resampling.LANCZOS)

        #create a new bg, black for now
        new_img = Image.new("RGB", (self.size, self.size), (0,0,0))

        #paste the resized image into the center
        new_img.paste(img, ((self.size - img.size[0]) // 2,
                            (self.size - img.size[1]) // 2))
    
        return new_img


class TreeClassifier:
    def __init__(self, model_path, class_names):
        #init ONNX runtime session
        #use CUDA if avail, fallback to CPU otherwise
        self.session = ort.InferenceSession(model_path, providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.class_names = class_names
        #get the name of input layer (which we called input in the train)
        self.input_name = self.session.get_inputs()[0].name

        self.Clahe = CLAHETransform()
        self.letterbox = LetterBoxResize()
        #normalization vals used in training must also be applied:
        self.mean = np.array([0.485, 0.456, 0.406]).astype(np.float32)
        self.std = np.array([0.229, 0.224, 0.225]).astype(np.float32)


    def preprocess(self, imgPth):
        
        #TODO: for now load relies on a path, for future use we want to use FASTAPI uploadfile
        img = Image.open(imgPth).convert('RGB')
        
        img = self.letterbox(img)

        #transform to np arr since CLAHE expects that
        img = self.Clahe(img)

        #normalize by dividing by 255 (scale 0 - 1 then subtract mean/ std)
        imgNp = np.array(img).astype(np.float32) / 255.0
        imgNp = (imgNp - self.mean) / self.std

        #final prep is to covert HCW (clahe did this) back to CHW and add batch dims
        imgNp = imgNp.transpose(2, 0, 1)  # (3, 320, 320)
        imgNp = np.expand_dims(imgNp, axis=0)  # (1, 3, 320, 320)

        return imgNp

    def predict(self, imgPth):
        inputData = self.preprocess(imgPth)
        
        #run the model
        #outputs are returned as a list of numpy arrs
        outputs = self.session.run(None, {self.input_name: inputData})

        #softmax to get probs
        #use only first row since that's all were using (1 output)
        logits = outputs[0]
        expLogits = np.exp(logits - np.max(logits))
        probs = expLogits / expLogits.sum()

        classIdx = np.argmax(probs)
        confidence = np.max(probs)

        return self.class_names[classIdx], confidence 
    

#EXEC
classes = ['Acer palmatum', 'Cedrus deodara', 'Celtis sinensis', 'Cinnamomum camphora (Linn) Presl', 'Elaeocarpus decipiens', 'Flowering cherry', 'Ginkgo biloba', 'Koelreuteria paniculata', 'Lagerstroemia indica', 'Liquidambar formosana', 'Liriodendron chinense', 'Magnolia grandiflora L', 'Magnolia liliflora Desr', 'Michelia chapensis', 'Osmanthus fragrans', 'Photinia serratifolia', 'Platanus', 'Prunus cerasifera f. atropurpurea', 'Salix babylonica', 'Sapindus saponaria', 'Styphnolobium japonicum', 'Triadica sebifera', 'Zelkova serrata']
tester = TreeClassifier("treeModel.onnx", classes)
#pick out an image from data
label, confidence = tester.predict(r"data\tree\test\Michelia chapensis\Michelia chapensis_tree_1 (72).JPG")

print(f"Pred: {label} | Conf: {confidence:.2%}")
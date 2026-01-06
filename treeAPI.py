from fastapi import FastAPI, File, UploadFile
import uvicorn
from treeONNXtest import TreeClassifier
#first try doing fastapi
#so FastAPI import itself is the main calss and creates an app instance, handles routing and automatically generates a SwaggerUI
#File is a class that allows it to treat specific parameter in a function be  treated as form upload instead of just plain data
#UploadFile allows me to not have to rely on sending byte strings and instead handle sending metadata and do mem management for memet


#unicorn starts the server, fastapi is simply the code, but it cannot listen to anything unless hooked up to some sort of server
CLASSES = [
    'Acer palmatum', 'Cedrus deodara', 'Celtis sinensis', 'Cinnamomum camphora (Linn) Presl', 
    'Elaeocarpus decipiens', 'Flowering cherry', 'Ginkgo biloba', 'Koelreuteria paniculata', 
    'Lagerstroemia indica', 'Liquidambar formosana', 'Liriodendron chinense', 'Magnolia grandiflora L', 
    'Magnolia liliflora Desr', 'Michelia chapensis', 'Osmanthus fragrans', 'Photinia serratifolia', 
    'Platanus', 'Prunus cerasifera f. atropurpurea', 'Salix babylonica', 'Sapindus saponaria', 
    'Styphnolobium japonicum', 'Triadica sebifera', 'Zelkova serrata'
]
#start a fastapi app:
app = FastAPI(title = "Tree Classifier")
model = TreeClassifier("treeModel.onnx", CLASSES)

#app.post is the route decorator, web protocols have verbs, web is for grabbing, post is for sending data to be processed. the path beside it is the endpoint, so the URL of something like this will end on /predict
@app.post('/predict')
#async func so it can handle multiple instances
#UploadFiile = File(...) simplt means that ... is required if a user tries to access the endpoint no file, it will send back error 422 Unprocessable Entity
async def predictTree(file: UploadFile = File(...)):

#await file.read creates a bytes bject, it awaits till it reads all of the contents
    contents = await file.read()
    label, conf = model.predict(contents)
    return {"prediction": label, "confidence": f"{conf:.2%}"}

#start a server here
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



import torch



weights_file = "final_model_ASL.pt"
filename = "asl_detector_model.pt"




model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_file, force_reload=True)
torch.save(model, filename)
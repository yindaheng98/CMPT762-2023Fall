# -*- coding: utf-8 -*-
from lab3_dataset import *

"""### Set Configs"""

'''
# Set the configs for the detection part in here.
# TODO: approx 15 lines
'''
cfg = get_cfg()
# model settings
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# training settings
cfg.DATASETS.TRAIN = ("data_detection_train",)
cfg.DATASETS.TEST = ()
cfg.SOLVER.MAX_ITER = 500
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.00025
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") # pretrain model

"""### Training"""

'''
# Create a DefaultTrainer using the above config and train the model
# TODO: approx 5 lines
'''
# trainer = DefaultTrainer(cfg)
# trainer.resume_or_load(resume=False)
# trainer.train()

"""### Evaluation and Visualization"""

'''
# After training the model, you need to update cfg.MODEL.WEIGHTS
# Define a DefaultPredictor
'''
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
predictor = DefaultPredictor(cfg)

'''
# Visualize the output for 3 random test samples
# TODO: approx 10 lines
'''
test_set = get_detection_data("test")
for i in range(3):
    idx = random.randrange(0, len(test_set))
    data = test_set[idx]
    img = cv2.imread(data["file_name"])
    result = predictor(img)
    visualizer = Visualizer(img[:, :, ::-1], metadata=MetadataCatalog.get("data_detection_test"), scale=1.2, instance_mode=ColorMode.IMAGE_BW)
    result = visualizer.draw_instance_predictions(result["instances"].to("cpu"))
    img = result.get_image()[:, :, ::-1]
    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR, f"test_set_{idx}.png"), img)

'''
# Use COCOEvaluator and build_detection_train_loader
# You can save the output predictions using inference_on_dataset
# TODO: approx 5 lines
'''
evaluator = COCOEvaluator("data_detection_val", tasks=("segm",), output_dir=cfg.OUTPUT_DIR)
test_loader = build_detection_test_loader(cfg, "data_detection_val")
print(inference_on_dataset(predictor.model, test_loader, evaluator))
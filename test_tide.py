from tidecv import TIDE, datasets

tide = TIDE()
tide.evaluate(datasets.COCO(),
              datasets.COCOResult('/home/chenzhen/code/detection/datasets/coco100/annotations/instances_val2017.json'),
              mode=TIDE.BOX) # Use TIDE.MASK for masks
tide.summarize()  # Summarize the results as tables in the console
tide.plot()       # Show a summary figure. Specify a folder and it'll output a png to that folder.
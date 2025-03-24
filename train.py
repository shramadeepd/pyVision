from pipe import FullPipeline

model = 'yolo11m.pt'
ls_id = 309

# # Ensure LS_ID is a list
# if not isinstance(ls_id, list):
#     ls_id = [ls_id]

pipeline = FullPipeline(model=model, LS=ls_id, epochs=1000, batch_size=-1)
pipeline.run()
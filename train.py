from pipe import FullPipeline

model = 'yolo11n-seg.pt'
ls_id = 309

# # Ensure LS_ID is a list
# if not isinstance(ls_id, list):
#     ls_id = [ls_id]
#model=model, LS=ls_id, epochs=1000, 
pipeline = FullPipeline(LS=ls_id,model='yolo11m.ptf')
pipeline.run()
from pipe import FullPipeline

model = 'yolo11l.pt'
ls_id = [193]

# # Ensure LS_ID is a list
# if not isinstance(ls_id, list):
#     ls_id = [ls_id]
#model=model, LS=ls_id, epochs=1000, 
pipeline = FullPipeline(LS=ls_id,model='yolo11l.pt',epochs=20)
pipeline.run()
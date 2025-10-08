from utils.prepare import load_datadict, load_video_loaders
csv_file = '../Video_Summarization/preprocessing/video_and_keyframe_path.csv'
loaders = load_datadict(csv_file, batch_size=4)
videoloader = load_video_loaders(csv_file, 4, mode="train", num_frames = 10)
# Test loading one batch
print("////////// KEY FRAME TENSOR ////////")
batch = next(iter(loaders['train']))
images = batch['images']   # tensor [B, num_frames, C, H, W] hoặc [B, C, H, W]
print(images[0].shape)
print("Số video trong batch:", len(images))
#for i, vid in enumerate(images):
#    print(f"Video {i} shape:", vid)  # (T, C, H, W)
print("/////// VIDEO TENSOR ///////////")
videobatch = next(iter(videoloader['train']))
videotensor = videobatch['video_tensor']
print(videotensor[0].shape)
print("Số video trong batch:", len(videotensor))
#for i, vid in enumerate(videotensor):
#    print(f"Video {i} shape:", vid)  # (T, C, H, W)
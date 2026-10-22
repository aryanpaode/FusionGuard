from fusionguard_multimodal import warmup_arcface

print("Preparing FusionGuard ArcFace model...")
print("On the first run DeepFace may download the ArcFace weights. Internet access is required once.")
warmup_arcface()
print("ArcFace is ready.")

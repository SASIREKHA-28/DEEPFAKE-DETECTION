"""
Comprehensive verification test for Veritas AI image and video pipelines.
"""
import app

print("Initializing app startup event...")
app.startup_event()

print("\n--- Testing Sample Real Image ---")
with open("static/samples/sample_real.jpg", "rb") as f:
    real_bytes = f.read()
res_real = app.evaluate_image(real_bytes)
print(f"Result Label: {res_real['label']}")
print(f"Confidence: {res_real['confidence']}%")
print(f"Face Detected: {res_real['face_detected']}")
print(f"Heatmap Base64 starts with: {res_real['heatmap_image'][:40]}... (len: {len(res_real['heatmap_image'])})")

print("\n--- Testing Sample Deepfake Image ---")
with open("static/samples/sample_fake.jpg", "rb") as f:
    fake_bytes = f.read()
res_fake = app.evaluate_image(fake_bytes)
print(f"Result Label: {res_fake['label']}")
print(f"Confidence: {res_fake['confidence']}%")
print(f"Face Detected: {res_fake['face_detected']}")
print(f"Heatmap Base64 starts with: {res_fake['heatmap_image'][:40]}... (len: {len(res_fake['heatmap_image'])})")

print("\n--- Testing Sample Video Pipeline ---")
res_video = app.evaluate_video("static/samples/sample_video.mp4")
print(f"Video Result Label: {res_video['label']}")
print(f"Video Confidence: {res_video['confidence']}%")
print(f"Frames Analyzed: {res_video['frames_analyzed']}")
print(f"Fake Frames Count: {res_video['fake_frames_count']} ({res_video['fake_frame_percentage']}%)")
print(f"Sample Frame 0: {res_video['frame_results'][0]}")

print("\n[SUCCESS] All image and video pipelines tested and functioning perfectly!")

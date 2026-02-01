"""Test TRELLIS.2 Image-to-3D via API."""
import requests
import time
import sys

# Test image path
image_path = "reference/test_input.jpg"

# Enable UTF-8 output
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Send multipart form request
print("Submitting Image-to-3D job...")
with open(image_path, "rb") as f:
    files = {"file": ("test_input.jpg", f, "image/jpeg")}
    data = {
        "seed": 42,
        "resolution": 256,
        "sparse_steps": 12,
        "sparse_guidance": 7.5,
        "sparse_rescale": 0.7,
        "shape_steps": 12,
        "shape_guidance": 3.0,
        "shape_rescale": 0.5,
        "tex_steps": 12,
        "tex_rescale": 0.0,
        "simplify": 0.95,
        "texture_size": 1024
    }
    resp = requests.post("http://127.0.0.1:8000/api/generate/image", files=files, data=data)

print(f"Response status: {resp.status_code}")
if resp.status_code != 200:
    print(f"Response text: {resp.text[:1000]}")
    exit(1)

result = resp.json()
job_id = result.get("job_id")
print(f"Job ID: {job_id}")

if job_id:
    print(f"\nMonitoring job {job_id}...")
    while True:
        status_resp = requests.get(f"http://127.0.0.1:8000/api/jobs/{job_id}")
        status = status_resp.json()
        stage = status.get("stage", "")
        progress = status.get("progress", 0)
        print(f"Status: {status.get('status', 'unknown')} - {stage} ({progress}%)")
        if status.get("status") == "completed":
            print(f"\nCOMPLETED!")
            print(f"Output files: {status.get('files', {})}")
            break
        elif status.get("status") == "failed":
            print(f"\nFAILED: {status.get('error', 'Unknown error')}")
            break
        time.sleep(5)

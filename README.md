# XRayVision

A Python-based DICOM storage and relay system with a real-time dashboard for processing and reviewing X-ray images using OpenAI APIs.

## Features

* ✅ Asynchronous TCP DICOM server (storescp-compatible)
* ✅ Real-time queue processing and OpenAI API integration
* ✅ Automatic DICOM-to-PNG conversion (OpenCV)
* ✅ Persistent processing history with file-based storage
* ✅ WebSocket-powered live dashboard updates
* ✅ Manual flagging/unflagging of processed items
* ✅ Manual DICOM QueryRetrieve trigger with configurable time span
* ✅ Automatic hourly QueryRetrieve of CR modality studies
* ✅ Fully responsive PicoCSS dashboard with lightbox image previews
* ✅ Logging with timestamps to both console and file

---

## Requirements

* Python 3.8+
* DICOM peer system for QueryRetrieve (pynetdicom-compatible)
* OpenAI API endpoint (can be local or remote)

### Python Packages

```bash
pip install aiohttp pydicom pynetdicom opencv-python
```

---

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/yourusername/xrayvision-dashboard.git
cd xrayvision-dashboard
```

2. Update the configuration in `xrayvision.py`:

```python
OPENAI_API_URL = 'http://127.0.0.1:8080/v1/chat/completions'
OPENAI_API_KEY = 'sk-your-api-key'
REMOTE_AE_IP = '127.0.0.1'
REMOTE_AE_PORT = 104
REMOTE_AE_TITLE = 'REMOTE_AE'
```

3. Run the server:

```bash
python xrayvision.py
```

4. Open the dashboard:

```
http://localhost:8000
```

---

## Project Structure

```text
├── images/               # Storage for received and processed files
├── xrayvision.py         # Main server and processing logic
├── dashboard.html        # WebSocket-powered dashboard UI
├── history.json          # Persistent history log
├── xrayvision.log        # Server logs with timestamps
├── README.md             # Project documentation
```

---

## Dashboard Features

* Live processing statistics (queue, current file, success, failure)
* Last 20 processed files with thumbnails
* Real-time flag/unflag functionality
* Manual QueryRetrieve trigger (select time span: 1, 3, 6, 12, 24 hours)
* Lightbox image preview with flagged and positive highlighting

---

## Logging

All events are logged to:

* `xrayvision.log` file
* Console output

Timestamps, info, warnings, and errors are all captured.

---

## Future Improvements

* SQLite history backend
* Pagination and filtering for large histories
* Live logs displayed in the dashboard
* Docker support

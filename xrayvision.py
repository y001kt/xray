#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 
# XRayVision - A Python-based DICOM storage and relay system with a real-time 
#              dashboard for processing and reviewing X-ray images using 
#              OpenAI APIs.
# Copyright (C) 2025 Costin Stroie <costinstroie@eridu.eu.org>
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.

import asyncio
import os
import uuid
import base64
import aiohttp
import cv2
import numpy as np
import math
import json
from aiohttp import web
from pydicom import dcmread
from pydicom.dataset import Dataset
from pynetdicom import AE, evt, StoragePresentationContexts, QueryRetrievePresentationContexts
from pynetdicom.sop_class import ComputedRadiographyImageStorage, DigitalXRayImageStorageForPresentation, PatientRootQueryRetrieveInformationModelFind, PatientRootQueryRetrieveInformationModelMove
from datetime import datetime, timedelta
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)8s | %(message)s',
    handlers=[
        #logging.FileHandler("xrayvision.log"),
        logging.StreamHandler()
    ]
)

# Configuration
OPENAI_API_URL = 'http://127.0.0.1:8080/v1/chat/completions'
#OPENAI_API_URL = 'http://192.168.3.239:8080/v1/chat/completions'
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-your-api-key')
DASHBOARD_PORT = 8000
AE_TITLE = 'XRAYVISION'
AE_PORT  = 4010
REMOTE_AE_TITLE = '3DNETCLOUD'
REMOTE_AE_IP = '192.168.3.50'
REMOTE_AE_PORT = 104
IMAGES_DIR = 'images'
HISTORY_FILE = os.path.join(IMAGES_DIR, 'history.json')

SYS_PROMPT = "You are a smart radiologist working in ER. Respond in plaintext. Start with yes or no, then provide just one line description like a radiologist. Do not assume, stick to the facts, but look again if you are in doubt."
USR_PROMPT = "{} in this {} xray of a {}? Are there any other lesions?"

os.makedirs(IMAGES_DIR, exist_ok = True)

main_loop = None  # Global variable to hold the main event loop
data_queue = asyncio.Queue()
websocket_clients = set()

# Dashboard state
dashboard_state = {
    'queue_size': 0,
    'processing_file': None,
    'success_count': 0,
    'failure_count': 0,
    'history': []
}

MAX_HISTORY = 100

async def query_retrieve_loop():
    while True:
        await query_and_retrieve()
        await asyncio.sleep(3600)

async def query_and_retrieve(hours = 1):
    ae = AE(ae_title = AE_TITLE)
    ae.requested_contexts = QueryRetrievePresentationContexts
    ae.connection_timeout = 30

    assoc = ae.associate(REMOTE_AE_IP, REMOTE_AE_PORT, ae_title = REMOTE_AE_TITLE)
    if assoc.is_established:
        logging.info(f"QueryRetrieve association established. Asking for studies in the last {hours} hours.")
        current_time = datetime.now()
        past_time = current_time - timedelta(hours = hours)
        time_range = f"{past_time.strftime('%H%M%S')}-{current_time.strftime('%H%M%S')}"
        date_today = current_time.strftime('%Y%m%d')

        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.StudyDate = date_today
        ds.StudyTime = time_range
        ds.Modality = "CR"

        responses = assoc.send_c_find(ds, PatientRootQueryRetrieveInformationModelFind)

        for (status, identifier) in responses:
            if status and status.Status in (0xFF00, 0xFF01):
                study_instance_uid = identifier.StudyInstanceUID
                logging.info(f"Found Study {study_instance_uid}")
                await send_c_move(ae, study_instance_uid)

        assoc.release()
    else:
        logging.error("Could not establish QueryRetrieve association.")

async def send_c_move(ae, study_instance_uid):
    assoc = ae.associate(REMOTE_AE_IP, REMOTE_AE_PORT, ae_title = REMOTE_AE_TITLE)
    if assoc.is_established:
        ds = Dataset()
        ds.QueryRetrieveLevel = "STUDY"
        ds.StudyInstanceUID = study_instance_uid

        responses = assoc.send_c_move(ds, AE_TITLE, PatientRootQueryRetrieveInformationModelMove)

        assoc.release()
    else:
        logging.error("Could not establish C-MOVE association.")

async def handle_manual_query(request):
    try:
        data = await request.json()
        hours = int(data.get('hours', 3))
        logging.info(f"Manual QueryRetrieve triggered for the last {hours} hours.")
        await query_and_retrieve(hours)
        return web.json_response({'status': 'success', 'message': f'Query triggered for the last {hours} hours.'})
    except Exception as e:
        logging.error(f"Error processing manual query: {e}")
        return web.json_response({'status': 'error', 'message': str(e)})

# Load existing .dcm files into queue
def preload_dicom_files():
    for filename in os.listdir(IMAGES_DIR):
        if filename.lower().endswith('.dcm'):
            dicom_file = os.path.join(IMAGES_DIR, filename)
            asyncio.create_task(data_queue.put(dicom_file))
            logging.info(f"Preloading {dicom_file} into processing queue...")
    dashboard_state['queue_size'] = data_queue.qsize()
    broadcast_dashboard_update()

def adjust_gamma(image, gamma = 1.2):
    # If gamma is None, compute it
    if gamma is None:
        if len(image.shape) > 2: # or image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute gamma = log(mid*255)/log(mean)
        mid = 0.5
        mean = np.median(image)
        gamma = math.log(mid * 255) / math.log(mean)
        logging.debug(f"Calculated gamma is {gamma:.2f}")
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def dicom_to_png(dicom_file, max_size = 800):
    """Convert DICOM to PNG and return PNG filename."""
    # Get the dataset
    ds = dcmread(dicom_file)
    # Check for PixelData
    if 'PixelData' not in ds:
        raise ValueError(f"DICOM file {dicom_file} has no pixel data!")
    # Normalize image to 0-255
    image = ds.pixel_array.astype(np.float32)
    image -= image.min()
    if image.max() != 0:
        image /= image.max()
    image *= 255.0
    image = image.astype(np.uint8)
    # Adjust gamma
    image = adjust_gamma(image, None)
    # Resize while maintaining aspect ratio
    height, width = image.shape[:2]
    if max(height, width) > max_size:
        if height > width:
            new_height = max_size
            new_width = int(width * (max_size / height))
        else:
            new_width = max_size
            new_height = int(height * (max_size / width))
        image = cv2.resize(image, (new_width, new_height), interpolation = cv2.INTER_AREA)
    # Save the PNG file
    base_name = os.path.splitext(os.path.basename(dicom_file))[0]
    png_file = os.path.join(IMAGES_DIR, f"{base_name}.png")
    cv2.imwrite(png_file, image)
    logging.info(f"Converted PNG saved to {png_file}")
    # Return the PNG file name and metadata
    meta = {
        'patient': {
            'name': str(ds.PatientName),
            'id': str(ds.PatientID),
            'age': str(ds.PatientAge),
            'sex': str(ds.PatientSex),
            'bdate': str(ds.PatientBirthDate),
        },
        'series': {
            'uid': str(ds.SeriesInstanceUID),
            'desc': str(ds.SeriesDescription),
            'proto': str(ds.ProtocolName),
            'date': str(ds.SeriesDate),
            'time': str(ds.SeriesTime),
        },
        'study': {
            'uid': str(ds.StudyInstanceUID),
            'date': str(ds.StudyDate),
            'time': str(ds.StudyTime),
        }
    }
    return png_file, meta

async def toggle_flag(request):
    data = await request.json()
    file_to_toggle = data.get('file')

    for item in dashboard_state['history']:
        if item['file'] == file_to_toggle:
            item['flagged'] = not item.get('flagged', False)
            break

    with open(HISTORY_FILE, 'w') as f:
        json.dump(dashboard_state['history'], f)

    await broadcast_dashboard_update()
    return web.json_response({'status': 'success', 'flagged': item['flagged']})

async def broadcast_dashboard_update(client = None):
    if not websocket_clients:
        return
    update = {
        'queue_size': dashboard_state['queue_size'],
        'processing_file': dashboard_state['processing_file'],
        'success_count': dashboard_state['success_count'],
        'failure_count': dashboard_state['failure_count'],
        'history': dashboard_state['history']
    }
    if client:
        try:
            await client.send_json(update)
        except Exception as e:
            logging.error(f"Error sending update to WebSocket client: {e}")
            websocket_clients.remove(client)
    else:
        for ws in websocket_clients.copy():
            try:
                await ws.send_json(update)
            except Exception as e:
                logging.error(f"Error sending update to WebSocket client: {e}")
                websocket_clients.remove(ws)

async def websocket_handler(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    websocket_clients.add(ws)
    await broadcast_dashboard_update(ws)
    logging.info(f"Dashboard connected via WebSocket from {request.remote}")
    try:
        async for msg in ws:
            pass
    finally:
        websocket_clients.remove(ws)
        logging.info("Dashboard WebSocket disconnected.")
    return ws

def handle_store(event):
    """Callback for receiving a DICOM file."""
    # Get the dataset
    ds = event.dataset
    ds.file_meta = event.file_meta
    # Check the Modality
    if ds.Modality == "CR":
        # Save the DICOM file
        dicom_file = os.path.join(IMAGES_DIR, f"{ds.SOPInstanceUID}.dcm")
        ds.save_as(dicom_file, write_like_original = False)
        logging.info(f"DICOM file saved to {dicom_file}")
        # Schedule queue put on the main event loop
        main_loop.call_soon_threadsafe(data_queue.put_nowait, dicom_file)
        dashboard_state['queue_size'] = data_queue.qsize()
        asyncio.run_coroutine_threadsafe(broadcast_dashboard_update(), main_loop)
    # Return success
    return 0x0000

def check_any(string, *words):
    """ Check if any of the words are present in string """
    return any(i in string for i in words)

async def send_image_to_openai(png_file, meta, max_retries = 3):
    """Send PNG to OpenAI API with retries and save response to text file."""
    # Read the PNG file
    with open(png_file, 'rb') as f:
        image_bytes = f.read()
    # Prepare the prompt
    question = "Is there anything abnormal"
    region = ""
    subject = ""
    anatomy = ""
    projection = ""
    gender = "child"
    age = ""

    # Identify anatomy
    desc = meta["series"]["desc"].lower()
    if check_any(desc, 'torace', 'pulmon'):
        anatomy = 'chest'
        question = "Are there any lung consolidations, hyperlucencies, infitrates, nodules, mediastinal shift, pleural effusion or pneumothorax"
    elif check_any(desc, 'grilaj', 'coaste'):
        anatomy = 'chest'
        question = "Are there any ribs or clavicles fractures"
    elif 'stern' in desc:
        anatomy = 'sternum'
        question = "Are there any fractures"
    elif 'abdomen' in desc:
        anatomy = 'abdominal'
        question = "Are there any hydroaeric levels or pneumoperitoneum"
    elif check_any(desc, 'cap', 'craniu', 'occiput'):
        anatomy = 'skull'
        question = "Are there any fractures"
    elif 'mandibula' in desc:
        anatomy = 'mandible'
        question = "Are there any fractures"
    elif 'nazal' in desc:
        anatomy = 'nasal bones'
        question = "Are there any fractures"
    elif 'sinus' in desc:
        anatomy = 'maxilar and frontal sinus'
        question = "Are there any changes in transparency of the sinuses"
    elif 'col.' in desc:
        anatomy = 'spine'
        question = "Are there any fractures or dislocations"
    elif 'bazin' in desc:
        anatomy = 'pelvis'
        question = "Are there any fractures"
    elif 'clavicul' in desc:
        anatomy = 'clavicle'
        question = "Are there any fractures"
    elif check_any(desc, 'humerus', 'antebrat'):
        anatomy = 'upper limb'
        question = "Are there any fractures, dislocations or bone tumors"
    elif check_any(desc, 'pumn', 'mana', 'deget'):
        anatomy = 'hand'
        question = "Are there any fractures, dislocations or bone tumors"
    elif 'umar' in desc:
        anatomy = 'shoulder'
        question = "Are there any fractures or dislocations"
    elif 'cot' in desc:
        anatomy = 'elbow'
        question = "Are there any fractures or dislocations"
    elif 'sold' in desc:
        anatomy = 'hip'
        question = "Are there any fractures or dislocations"
    elif check_any(desc, 'femur', 'tibie', 'glezna', 'picior', 'gamba', 'calcai'):
        anatomy = 'lower limb'
        question = "Are there any fractures, dislocations or bone tumors"
    elif check_any(desc, 'genunchi', 'patella'):
        anatomy = 'knee'
        question = "Are there any fractures or dislocations"
    else:
        anatomy = desc

    # Identify projection
    if check_any(desc, 'a.p.', 'p.a.', 'd.v.', 'v.d.'):
        projection = 'frontal'
    elif 'lat.' in desc:
        projection = 'lateral'
    elif 'oblic' in desc:
        projection = 'oblique'

    # Identify gender
    if 'm' in meta["patient"]["sex"].lower():
        gender = 'boy'
    elif 'f' in meta["patient"]["sex"].lower():
        gender = 'girl'

    # Identify age
    age = meta["patient"]["age"].lower().replace("y", "").strip()
    if age:
        if age == '000':
            age = 'newborn'
        else:
            age = age + " years old"
    else:
        age = ""

    subject = " ".join([age, gender])
    if anatomy:
        region = " ".join([projection, anatomy])
    prompt = USR_PROMPT.format(question, region, subject)
    logging.debug(f"Prompt: {prompt}")
    # Base64 encode the PNG to comply with OpenAI Vision API
    image_b64 = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:image/png;base64,{image_b64}"
    # Prepare the request headers
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json',
    }
    # Prepare the JSON data
    data = {
        'model': 'medgemma-4b-it',
        'messages': [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYS_PROMPT}]
            },
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': image_url}}
                ]
            }
        ]
    }
    # Up to 3 attempts with exponential backoff (2s, 4s, 8s delays).
    attempt = 0
    while attempt < max_retries:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(OPENAI_API_URL, headers = headers, json = data) as response:
                    result = await response.json()
                    text = result["choices"][0]["message"]["content"]
                    text = text.replace('\n', " ").replace("  ", " ").strip()
                    logging.info(f"OpenAI API response for {png_file}: {text}")
                    # Save the result to a text file
                    base_name = os.path.splitext(os.path.basename(png_file))[0]
                    text_file = os.path.join(IMAGES_DIR, f"{base_name}.txt")
                    with open(text_file, 'w') as f:
                        f.write(text)
                    logging.info(f"Response saved to {text_file}")
                    # Update the dashboard
                    dashboard_state['success_count'] += 1
                    # Add to history (keep only last MAX_HISTORY)
                    dashboard_state['history'].insert(0, {'file': os.path.basename(png_file), 'meta': meta, 'text': text, 'flagged': False})
                    dashboard_state['history'] = dashboard_state['history'][:MAX_HISTORY]
                    await broadcast_dashboard_update()
                    # Save as JSON-friendly structure
                    history_to_save = [item for item in dashboard_state['history']]
                    # Save history
                    with open(HISTORY_FILE, 'w') as f:
                        json.dump(history_to_save, f)
                    # Success
                    return True
        except Exception as e:
            logging.warning(f"Error uploading {png_file} (Attempt {attempt + 1}): {e}")
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
            attempt += 1
    # Failure after max_retries
    logging.error(f"Failed to upload {png_file} after {max_retries} attempts.")
    dashboard_state['failure_count'] += 1
    await broadcast_dashboard_update()
    return False

async def relay_to_openai():
    """Relay PNG files to OpenAI API with retries and dashboard update."""
    while True:
        # Get one file from queue
        dicom_file = await data_queue.get()
        # Update the dashboard
        dashboard_state['processing_file'] = os.path.basename(dicom_file)
        dashboard_state['queue_size'] = data_queue.qsize()
        await broadcast_dashboard_update()
        # Try to convert to PNG
        try:
            png_file, meta = dicom_to_png(dicom_file)
        except Exception as e:
            logging.error(f"Error converting DICOM file {dicom_file}: {e}")
        # Try to send to AI
        try:
            await send_image_to_openai(png_file, meta)
        except Exception as e:
            logging.error(f"Unhandled error processing {png_file}: {e}")
        finally:
            dashboard_state['processing_file'] = None
            dashboard_state['queue_size'] = data_queue.qsize()
            await broadcast_dashboard_update()
            data_queue.task_done()
        # Remove the DICOM file
        try:
            os.remove(dicom_file)
            logging.info(f"DICOM file {dicom_file} deleted after processing.")
        except Exception as e:
            logging.error(f"Error removing DICOM file {dicom_file}: {e}")

def start_dicom_server():
    """Start the DICOM Storage SCP."""
    ae = AE(ae_title=AE_TITLE)
    # Accept everything
    #ae.supported_contexts = StoragePresentationContexts
    # Accept only XRays
    ae.add_supported_context(ComputedRadiographyImageStorage)
    ae.add_supported_context(DigitalXRayImageStorageForPresentation)
    # C-Store handler
    handlers = [(evt.EVT_C_STORE, handle_store)]
    logging.info(f"Starting DICOM server on port {AE_PORT} with AE Title '{AE_TITLE}'...")
    ae.start_server(("0.0.0.0", AE_PORT), evt_handlers = handlers, block = True)

async def dashboard(request):
    with open('dashboard.html', 'r') as f:
        content = f.read()
    return web.Response(text = content, content_type = 'text/html')

async def start_dashboard():
    """Start the dashboard web server."""
    app = web.Application()
    app.router.add_get('/', dashboard)
    app.router.add_static('/static/', path = IMAGES_DIR, name = 'static')
    app.router.add_get('/ws', websocket_handler)
    app.router.add_post('/toggle_flag', toggle_flag)
    app.router.add_post('/trigger_query', handle_manual_query)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, '0.0.0.0', DASHBOARD_PORT)
    await site.start()
    logging.info(f"Dashboard available at http://localhost:{DASHBOARD_PORT}")

def load_history():
    """Load history on startup."""
    if os.path.exists(HISTORY_FILE):
        logging.info(f"Loading history from {HISTORY_FILE}")
        with open(HISTORY_FILE, 'r') as f:
            dashboard_state['history'] = json.load(f)
    else:
        dashboard_state['history'] = []

async def main():
    # Load history
    load_history()
    # Store main event loop here
    global main_loop
    main_loop = asyncio.get_running_loop()
    # Start the asynchronous tasks
    asyncio.create_task(relay_to_openai())
    asyncio.create_task(start_dashboard())
    asyncio.create_task(query_retrieve_loop())
    # Preload the existing dicom files
    preload_dicom_files()
    # Start the DICOM server
    await asyncio.get_running_loop().run_in_executor(None, start_dicom_server)


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("Server stopped by user. Shutting down.")

# Required Libraries
import os
import platform
import subprocess
import threading
import time
import warnings
import re
import string
import ssl
import certifi
import urllib.request
import difflib
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import torch
from itertools import zip_longest
from openpyxl import Workbook
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor

# Suppress specific warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

# SSL context setup for custom URL access
def create_ssl_context():
    context = ssl.create_default_context(cafile=certifi.where())
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context

# Function to transcribe audio batch
def transcribe_audio_batch(file_paths, progress_callback=None):
    if not file_paths:
        print("No files provided for transcription.")
        return {}

    # Use the custom SSL context
    ssl_context = create_ssl_context()
    
    # Patch urllib to use our custom SSL context
    original_urlopen = urllib.request.urlopen
    urllib.request.urlopen = lambda *args, **kwargs: original_urlopen(*args, **kwargs, context=ssl_context)
    
    try:
        # Load your specific Whisper model from Hugging Face
        model_name = "shaunliu82714/whisper-small-en-genshin"
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
        
        # Enable GPU acceleration if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        # Create pipeline for ASR
        asr_pipeline = pipeline("automatic-speech-recognition", model=model, processor=processor, device=0 if device == "cuda" else -1)
        
        transcriptions = {}
        start_time = time.time()
        for i, file_path in enumerate(file_paths):
            if not os.path.isfile(file_path):
                print(f"File not found: {file_path}")
                continue

            print(f"Transcribing file {i+1}/{len(file_paths)}: {file_path}")
            
            # Perform transcription
            try:
                result = asr_pipeline(file_path)
                transcriptions[os.path.splitext(os.path.basename(file_path))[0]] = result['text'].strip()
            except Exception as e:
                print(f"Error transcribing file {file_path}: {e}")
                continue
            
            # Progress tracking
            elapsed_time = time.time() - start_time
            avg_time_per_file = elapsed_time / (i + 1)
            remaining_time = avg_time_per_file * (len(file_paths) - (i + 1))
            if progress_callback:
                progress_callback(i + 1, len(file_paths), "Transcribing audio files...", remaining_time)
        
        print("Completed transcriptions:", transcriptions)
        return transcriptions
    
    finally:
        # Restore the original urlopen function
        urllib.request.urlopen = original_urlopen

# Function to save transcriptions to Excel
def save_transcriptions_to_excel(transcriptions, output_file='transcriptions.xlsx'):
    if not transcriptions:
        print("No transcriptions available to save.")
        return
    
    wb = Workbook()
    ws = wb.active
    ws.title = "Transcriptions"
    
    # Add headers
    ws['A1'] = "VOID (File Name)"
    ws['B1'] = "Transcribed Content"
    
    # Add data
    for row, (file_name, content) in enumerate(transcriptions.items(), start=2):
        ws.cell(row=row, column=1, value=file_name)
        ws.cell(row=row, column=2, value=content)
    
    # Save Excel file
    wb.save(output_file)
    print(f"Transcriptions saved to {output_file}")
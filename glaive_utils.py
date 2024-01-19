import os
import io
import json
from google.cloud import storage
import datetime

GLAIVE_BUCKET = "glaive-model-weights"

def load_jsonl(path, mode="r"):
    """Load a .jsonl file into a list of dictionaries."""
    with open(path, mode):
        jlist = [json.loads(line) for line in f]
    return jlist

def load_json(path, mode="r"):
    """Load a .jsonl file into a list of dictionaries."""
    with open(path, mode) as f:
        jlist = json.load(f)
    return jlist


def folder_exists_on_gcs(dir: str, bucket_name: str = GLAIVE_BUCKET):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=dir)
    for blob in blobs:
        return True

    return False

def upload_blob(source:str,
                destination:str,
                bucket_name: str = GLAIVE_BUCKET):
    """
    Uploads a file to the specified Google Cloud Storage bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_filename(source,timeout=600)
    url = blob.generate_signed_url(datetime.timedelta(seconds=864000), method='GET')
    return url

def download_file(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from Google Cloud Storage."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

def callback_completion(callback_url:str,
                        model_url:str,
                        failed:bool,
                        error:str=None):
    """
    Sends a callback to the specified URL.
    """
    payload = {'failed': failed, "url": model_url, "error": error}
    response = requests.post(callback_url, json=payload)
    return response
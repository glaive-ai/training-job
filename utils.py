import os
import io
import json
from google.cloud import storage
import datetime

GLAIVE_BUCKET = "glaive-model-weights"

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .jsonl file into a list of dictionaries."""
    f = _make_r_io_base(f, mode)
    jlist = [json.loads(line) for line in f]
    f.close()
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

def callback_completion(callback_url:str,model_url:str,failed:bool,error:str=None):
    """
    Sends a callback to the specified URL.
    """
    payload = {'failed': failed, "url": model_url, "error": error}
    response = requests.post(callback_url, json=payload)
    return response
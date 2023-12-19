import os
import io
import json
from google.cloud import storage
import datetime


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

def upload_blob(source:str,
                destination:str,
                bucket_name: str = "glaive-model-weights"):
    """
    Uploads a file to the specified Google Cloud Storage bucket.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination)
    blob.upload_from_filename(source,timeout=600)
    url = blob.generate_signed_url(datetime.timedelta(seconds=864000), method='GET')
    return url
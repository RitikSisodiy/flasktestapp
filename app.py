from flask import Flask, request, jsonify

app = Flask(__name__)
import logging
import requests
from azure.storage.blob import BlobServiceClient
import time
import json
import os
import re
# Environment variables (replace with your actual values)
blob_id_name_delemeter = "/"
TENANT_ID = "122ec050-e4e4-47e8-862a-5c4b5d574201"
CLIENT_ID = "3ebb5e30-c068-497d-9407-22a0c564a689"
CLIENT_SECRET = "tIC8Q~fV8eVTaN8dFE71cLculuFa0HRqey8Wmasb"
CLIENT_STATE = "SecretClientState"
CONTAINER_NAME = "sharepointtoblob"
CACHE_CONTAINER_NAME = "webhookcache"
STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=stairesearchpoceastus001;AccountKey=4Gfsp4hlLo7qCz+y2n02cEIDZm5udhh2PbdNvfHj1OnHy6u1yrPY0HHnb3YyizB3s6DzSsZr7td6+AStCINeBw==;EndpointSuffix=core.windows.net"
SP_CLIENT_ID = "800d0fad-ae3a-4213-a803-2bec5bdb5b8d"
SP_CLIENT_SECRET = "NWV2XThtiBdDIB5dTx0xjBrSW0LnQbUQj4vafW3SB4o="
SP_SITE_URL = "https://emiratesglobalaluminium.sharepoint.com"



blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONNECTION_STRING)

def save_to_cache(key, data):
    data = {"url":data}
    blob_client = blob_service_client.get_blob_client(CACHE_CONTAINER_NAME, key)
    blob_client.upload_blob(json.dumps(data), overwrite=True)

def get_from_cache(key):
    blob_client = blob_service_client.get_blob_client(CACHE_CONTAINER_NAME, key)
    try:
        data = blob_client.download_blob().readall()
        return json.loads(data).get("url")
    except:
        return None
def get_access_token():
    token_url = f'https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token'
    token_data = {
        'grant_type': 'client_credentials',
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'scope': 'https://graph.microsoft.com/.default'
    }
    token_r = requests.post(token_url, data=token_data)
    return token_r.json().get('access_token')

@app.route('/webhook', methods=['POST'])
def webhook():
    validation_token = request.args.get('validationtoken')
    print("Query Parameters:", request.args)

    # Print raw request data
    print("Raw Data:", request.data)

    # Print JSON data if available
    if request.is_json:
        print("JSON Data:", request.json)

    # Print form data if applicable
    print("Form Data:", request.form)
    if validation_token:
        print("returning v",validation_token)
        return validation_token, 200
    # Handle other webhook events here
    req_body = request.json
    for notification in req_body['value']:
        site_id = notification['siteUrl']
        list_id = notification['resource']
        # Use Microsoft Graph API delta query to get changes
        changes = get_changes(site_id, list_id)
        # Process changes (create, update, delete)
        process_changes(changes,list_id)
        print("changes are procssed")
    print("returning success status")
    return jsonify({'status': 'received'}), 200

def get_changes(site_name, list_id):
    key = "cache.txt"
    # Get an access token
    access_token = get_access_token()
    site_id = get_site_id(site_name,access_token=access_token)
    # Make a delta query to Microsoft Graph API
    delta_url = get_from_cache(key)
    if not delta_url:
        delta_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items/delta"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(delta_url, headers=headers)
    response_json = response.json()
    save_to_cache(key,response_json["@odata.deltaLink"])
    return response.json().get("value")

def process_changes(changes,list_id):
    print("changes",changes)
    for change in changes:
        item_id =  change['id']
        if len(change.get('fields',{}))==0:
            delete_from_blob(item_id)
        # Handle delete
        elif change['@odata.etag']:
            # Get the item content and metadata, then upload to Azure Blob Storage
            content, metadata = get_item_content_and_metadata(change['parentReference']['siteId'], list_id, item_id)
            upload_to_blob(content, metadata, item_id)
def get_item_content_and_metadata(site_id, list_id, item_id):
    # Get an access token

    access_token = get_access_token()

    # Get the item content from SharePoint
    item_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items/{item_id}/driveItem/content"
    headers = {"Authorization": f"Bearer {access_token}"}
    content_response = requests.get(item_url, headers=headers)
    content = content_response.content

    # Get the item metadata from SharePoint
    metadata_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items/{item_id}"
    metadata_response = requests.get(metadata_url, headers=headers)
    metadata = metadata_response.json().get('fields', {})

    return content, metadata

def upload_to_blob(content, metadata, item_id):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=f"{item_id}{blob_id_name_delemeter}{metadata['FileLeafRef']}")
    blob_client.upload_blob(content, overwrite=True)




def set_metadata(self, blob_service_client: BlobServiceClient, container_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob="sample-blob.txt")

    # Retrieve existing metadata, if desired
    blob_metadata = blob_client.get_blob_properties().metadata

    more_blob_metadata = {'docType': 'text', 'docCategory': 'reference'}
    blob_metadata.update(more_blob_metadata)

    # Set metadata on the blob
    blob_client.set_blob_metadata(metadata=blob_metadata)


def delete_from_blob(item_id):
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blobs_to_delete = [blob.name for blob in container_client.list_blobs(name_starts_with=f"{item_id}{blob_id_name_delemeter}")]
    if blobs_to_delete:
        container_client.delete_blobs(*blobs_to_delete)

def upload_file_to_root(site_id, file_path, access_token):
    import os

    file_name = os.path.basename(file_path)
    url = f"https://graph.microsoft.com/v1.0/drives/b!MRP5k8bOLU2eLj_LhsRR3ISbxhpCDTlOhn0nJaHFR12S3kv4ARJQRp-yoG22qlnQ/root:/BC.png:/content"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/octet-stream"
    }

    with open(file_path, "rb") as file:
        response = requests.put(url, headers=headers, data=file)
    return response
def get_site_id(SITE_NAME_IS,access_token=None):
    site_url = f'https://graph.microsoft.com/v1.0/sites/root:{SITE_NAME_IS}'
    if not access_token:

        access_token = get_access_token()
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    site_r = requests.get(site_url, headers=headers)
    return site_r.json().get('id')



def chunk_text(text, chunk_size=1000, overlap_size=100):
    """
    Splits text into overlapping chunks.

    Args:
        text (str): The input text to split.
        chunk_size (int): The size of each chunk.
        overlap_size (int): The number of characters to overlap between chunks.

    Returns:
        list: A list of text chunks.
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])

        if end == text_length:
            break  # Stop if we've reached the end
        start += chunk_size - overlap_size

    return chunks

@app.route("/custom-split", methods=["POST"])
def custom_split():
    data = request.json
    print("Query Parameters:", request.args)

    # Print raw request data
    print("Raw Data:", request.data)

    # Print JSON data if available
    if request.is_json:
        print("JSON Data:", data)
    values = data.get("values", [])
    results = []

    for item in values:
        record_id = item.get("recordId", "1")
        text = item.get("data", {}).get("text", "")
        # Split text by sentence (custom rule: split at periods, but keep them)
        chunks = chunk_text(text)

        # Ensure response format is correct for Azure AI Search
        results.append({
            "recordId": record_id,
            "data": {"split_text": chunks}  # Wrap chunks in a dictionary
        })

    return jsonify({"values": results})

from document_intelligence import analyze_layout
@app.route("/process_doc", methods=["POST"])
def process_doc():
    data = request.json
    print("Query Parameters:", request.args)

    # Print raw request data
    print("Raw Data:", request.data)

    # Print JSON data if available
    if request.is_json:
        print("JSON Data:", data)
    values = data.get("values", [])
    results = []
    for item in values:
        file_data = item.get("data", {}).get("file_data", "")
        file_name = item.get("data", {}).get("file_name", "uploaded_file")

        if not file_data:
            continue

        # Save file locally

        # Extract text
        extracted_text = analyze_layout(file_data["data"],file_name)

        # Create response
        results.append({
            "recordId": item["recordId"],
            "data": {"content": extracted_text},  # Limiting text for preview
            "errors": None,
            "warnings": None
        })
    print(json.dumps({"values": results},indent=4))
    return jsonify({"values": results})


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

def get_changes(site_name, list_id):
    key = "cache.txt"
    # Get an access token
    access_token = get_access_token()
    site_id = get_site_id(site_name,access_token=access_token)
    # Make a delta query to Microsoft Graph API
    delta_url = get_from_cache(key)
    if not delta_url:
        delta_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items/delta"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(delta_url, headers=headers)
    response_json = response.json()
    save_to_cache(key,response_json["@odata.deltaLink"])
    return response.json().get("value")

def process_changes(changes,list_id):
    print("changes",changes)
    for change in changes:
        item_id =  change['id']
        if len(change.get('fields',{}))==0:
            delete_from_blob(item_id)
        # Handle delete
        elif change['@odata.etag']:
            # Get the item content and metadata, then upload to Azure Blob Storage
            content, metadata = get_item_content_and_metadata(change['parentReference']['siteId'], list_id, item_id)
            upload_to_blob(content, metadata, item_id)
def get_item_content_and_metadata(site_id, list_id, item_id):
    # Get an access token

    access_token = get_access_token()

    # Get the item content from SharePoint
    item_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items/{item_id}/driveItem/content"
    headers = {"Authorization": f"Bearer {access_token}"}
    content_response = requests.get(item_url, headers=headers)
    content = content_response.content

    # Get the item metadata from SharePoint
    metadata_url = f"https://graph.microsoft.com/v1.0/sites/{site_id}/lists/{list_id}/items/{item_id}"
    metadata_response = requests.get(metadata_url, headers=headers)
    metadata = metadata_response.json().get('fields', {})

    return content, metadata

def upload_to_blob(content, metadata, item_id):
    blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=f"{item_id}{blob_id_name_delemeter}{metadata['FileLeafRef']}")
    blob_client.upload_blob(content, overwrite=True)




def set_metadata(self, blob_service_client: BlobServiceClient, container_name):
    blob_client = blob_service_client.get_blob_client(container=container_name, blob="sample-blob.txt")

    # Retrieve existing metadata, if desired
    blob_metadata = blob_client.get_blob_properties().metadata

    more_blob_metadata = {'docType': 'text', 'docCategory': 'reference'}
    blob_metadata.update(more_blob_metadata)

    # Set metadata on the blob
    blob_client.set_blob_metadata(metadata=blob_metadata)


def delete_from_blob(item_id):
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    blobs_to_delete = [blob.name for blob in container_client.list_blobs(name_starts_with=f"{item_id}{blob_id_name_delemeter}")]
    if blobs_to_delete:
        container_client.delete_blobs(*blobs_to_delete)

def upload_file_to_root(site_id, file_path, access_token):
    import os

    file_name = os.path.basename(file_path)
    url = f"https://graph.microsoft.com/v1.0/drives/b!MRP5k8bOLU2eLj_LhsRR3ISbxhpCDTlOhn0nJaHFR12S3kv4ARJQRp-yoG22qlnQ/root:/BC.png:/content"

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/octet-stream"
    }

    with open(file_path, "rb") as file:
        response = requests.put(url, headers=headers, data=file)
    return response
def get_site_id(SITE_NAME_IS,access_token=None):
    site_url = f'https://graph.microsoft.com/v1.0/sites/root:{SITE_NAME_IS}'
    if not access_token:

        access_token = get_access_token()
    headers = {
        'Authorization': f'Bearer {access_token}'
    }
    site_r = requests.get(site_url, headers=headers)
    return site_r.json().get('id')




if __name__ == '__main__':
    app.run(debug=True)

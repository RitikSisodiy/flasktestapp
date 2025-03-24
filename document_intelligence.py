"""
This code loads environment variables using the `dotenv` library and sets the necessary environment variables for Azure services.
The environment variables are loaded from the `.env` file in the same directory as this notebook.
"""

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from openai import AzureOpenAI
"""
This code loads environment variables using the `dotenv` library and sets the necessary environment variables for Azure services.
The environment variables are loaded from the `.env` file in the same directory as this notebook.
"""

import os
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentContentFormat
from openai import AzureOpenAI

load_dotenv()

doc_intelligence_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
doc_intelligence_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")

aoai_api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
aoai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
aoai_deployment_name="gpt-4v"
aoai_api_version="2024-02-15-preview"



from PIL import Image
import fitz  # PyMuPDF
import mimetypes

def crop_image_from_image(image_path, page_number, bounding_box):
    """
    Crops an image based on a bounding box.

    :param image_path: Path to the image file.
    :param page_number: The page number of the image to crop (for TIFF format).
    :param bounding_box: A tuple of (left, upper, right, lower) coordinates for the bounding box.
    :return: A cropped image.
    :rtype: PIL.Image.Image
    """
    with Image.open(image_path) as img:
        if img.format == "TIFF":
            # Open the TIFF image
            img.seek(page_number)
            img = img.copy()

        # The bounding box is expected to be in the format (left, upper, right, lower).
        cropped_image = img.crop(bounding_box)
        return cropped_image

def crop_image_from_pdf_page(pdf_path, page_number, bounding_box):
    """
    Crops a region from a given page in a PDF and returns it as an image.

    :param pdf_path: Path to the PDF file.
    :param page_number: The page number to crop from (0-indexed).
    :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates for the bounding box.
    :return: A PIL Image of the cropped area.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
    bbx = [x * 72 for x in bounding_box]
    rect = fitz.Rect(bbx)
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), clip=rect)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    doc.close()

    return img

def crop_image_from_file(file_path, page_number, bounding_box):
    """
    Crop an image from a file.

    Args:
        file_path (str): The path to the file.
        page_number (int): The page number (for PDF and TIFF files, 0-indexed).
        bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

    Returns:
        A PIL Image of the cropped area.
    """
    mime_type = mimetypes.guess_type(file_path)[0]

    if mime_type == "application/pdf":
        return crop_image_from_pdf_page(file_path, page_number, bounding_box)
    else:
        return crop_image_from_image(file_path, page_number, bounding_box)


import base64
from mimetypes import guess_type

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"



MAX_TOKENS = 2000

def understand_image_with_gptv(api_base, api_key, deployment_name, api_version, image_path, caption):
    """
    Generates a description for an image using the GPT-4V model.

    Parameters:
    - api_base (str): The base URL of the API.
    - api_key (str): The API key for authentication.
    - deployment_name (str): The name of the deployment.
    - api_version (str): The version of the API.
    - image_path (str): The path to the image file.
    - caption (str): The caption for the image.

    Returns:
    - img_description (str): The generated description for the image.
    """
    return "this is sample image description"
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{deployment_name}"
    )

    data_url = local_image_to_data_url(image_path)

    # We send both image caption and the image body to GPTv for better understanding
    if caption != "":
        response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": [
                        {
                            "type": "text",
                            "text": f"Describe this image (note: it has image caption: {caption}):"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ] }
                ],
                max_tokens=MAX_TOKENS
            )

    else:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [
                    {
                        "type": "text",
                        "text": "Describe this image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ] }
            ],
            max_tokens=MAX_TOKENS
        )

    img_description = response.choices[0].message.content

    return img_description



import re
import pdb
def update_figure_description(md_content, img_description, idx):
    """
    Updates the figure description in the Markdown content.

    Args:
        md_content (str): The original Markdown content.
        img_description (str): The new description for the image.
        idx (int): The index of the figure.

    Returns:
        str: The updated Markdown content with the new figure description.
    """

    # The substring you're looking for
    start_substring = f"![](figures/{idx})"
    end_substring = "</figure>"
    new_string = f"<!-- FigureContent=\"{img_description}\" -->"

    new_md_content = md_content
    # Find the start and end indices of the part to replace
    start_index = md_content.find(start_substring)
    if start_index != -1:  # if start_substring is found
        start_index += len(start_substring)  # move the index to the end of start_substring
        end_index = md_content.find(end_substring, start_index)
        if end_index != -1:  # if end_substring is found
            # Replace the old string with the new string
            new_md_content = md_content[:start_index] + new_string + md_content[end_index:]

    return new_md_content

def update_figure_description(md_content, img_description, idx):
    """
    Updates the figure description in the Markdown content.

    Args:
        md_content (str): The original Markdown content.
        img_description (str): The new description for the image.
        idx (int): The index of the figure.

    Returns:
        str: The updated Markdown content with the new figure description.
    """

    figure_pattern = r"<figure>[\s\S]*?</figure>"
    figure_matches = list(re.finditer(figure_pattern, md_content))
    end_substring = "</figure>"
    # pdb.set_trace()
    if idx < len(figure_matches):
        start_index, end_index = figure_matches[idx].span()
        new_string = f"\n<!-- FigureContent=\"{img_description}\" -->\n"
        md_content = md_content[:end_index-len(end_substring)] + new_string + md_content[end_index-len(end_substring):]

    return md_content




def analyze_layout(input_file_path, output_folder):
    """
    Analyzes the layout of a document and extracts figures along with their descriptions, then update the markdown output with the new description.

    Args:
        input_file_path (str): The path to the input document file.
        output_folder (str): The path to the output folder where the cropped images will be saved.

    Returns:
        str: The updated Markdown content with figure descriptions.

    """
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=doc_intelligence_endpoint,
        credential=AzureKeyCredential(doc_intelligence_key),
        headers={"x-ms-useragent":"sample-code-figure-understanding/1.0.0"},
    )

    with open(input_file_path, "rb") as f:
        poller = document_intelligence_client.begin_analyze_document(
            "prebuilt-layout", body=f, content_type="application/octet-stream", output_content_format=DocumentContentFormat.MARKDOWN
        )

    result = poller.result()
    md_content = result.content

    # print(md_content)
    # exit()
    if result.figures:
        print("Figures:")
        for idx, figure in enumerate(result.figures):
            figure_content = ""
            img_description = ""
            print(f"Figure #{idx} has the following spans: {figure.spans}")
            for i, span in enumerate(figure.spans):
                print(f"Span #{i}: {span}")
                figure_content += md_content[span.offset:span.offset + span.length]
            print(f"Original figure content in markdown: {figure_content}")

            # Note: figure bounding regions currently contain both the bounding region of figure caption and figure body
            if figure.caption:
                caption_region = figure.caption.bounding_regions
                print(f"\tCaption: {figure.caption.content}")
                print(f"\tCaption bounding region: {caption_region}")
                for region in figure.bounding_regions:
                    if region not in caption_region:
                        print(f"\tFigure body bounding regions: {region}")
                        # To learn more about bounding regions, see https://aka.ms/bounding-region
                        boundingbox = (
                                region.polygon[0],  # x0 (left)
                                region.polygon[1],  # y0 (top)
                                region.polygon[4],  # x1 (right)
                                region.polygon[5]   # y1 (bottom)
                            )
                        print(f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")
                        cropped_image = crop_image_from_file(input_file_path, region.page_number - 1, boundingbox) # page_number is 1-indexed

                        # Get the base name of the file
                        base_name = os.path.basename(input_file_path)
                        # Remove the file extension
                        file_name_without_extension = os.path.splitext(base_name)[0]

                        output_file = f"{file_name_without_extension}_cropped_image_{idx}.png"
                        cropped_image_filename = os.path.join(output_folder, output_file)

                        cropped_image.save(cropped_image_filename)
                        print(f"\tFigure {idx} cropped and saved as {cropped_image_filename}")
                        img_description += understand_image_with_gptv(aoai_api_base, aoai_api_key, aoai_deployment_name, aoai_api_version, cropped_image_filename, figure.caption.content)
                        print(f"\tDescription of figure {idx}: {img_description}")
            else:
                print("\tNo caption found for this figure.")
                for region in figure.bounding_regions:
                    print(f"\tFigure body bounding regions: {region}")
                    # To learn more about bounding regions, see https://aka.ms/bounding-region
                    boundingbox = (
                            region.polygon[0],  # x0 (left)
                            region.polygon[1],  # y0 (top
                            region.polygon[4],  # x1 (right)
                            region.polygon[5]   # y1 (bottom)
                        )
                    print(f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")

                    cropped_image = crop_image_from_file(input_file_path, region.page_number - 1, boundingbox) # page_number is 1-indexed

                    # Get the base name of the file
                    base_name = os.path.basename(input_file_path)
                    # Remove the file extension
                    file_name_without_extension = os.path.splitext(base_name)[0]

                    output_file = f"{file_name_without_extension}_cropped_image_{idx}.png"
                    cropped_image_filename = os.path.join(output_folder, output_file)
                    # cropped_image_filename = f"data/cropped/image_{idx}.png"
                    cropped_image.save(cropped_image_filename)
                    print(f"\tFigure {idx} cropped and saved as {cropped_image_filename}")
                    img_description += understand_image_with_gptv(aoai_api_base, aoai_api_key, aoai_deployment_name, aoai_api_version, cropped_image_filename, "")
                    print(f"\tDescription of figure {idx}: {img_description}")

            # replace_figure_description(figure_content, img_description, idx)
            # print(md_content)
            md_content = update_figure_description(md_content, cropped_image_filename, idx)

    return md_content


load_dotenv()

doc_intelligence_endpoint = os.environ.get("doc_intelligence_endpoint")
doc_intelligence_key = os.environ.get("doc_intelligence_key")

aoai_api_base = os.environ.get("aoai_api_base")
aoai_api_key = os.environ.get("aoai_api_key")
aoai_deployment_name = os.environ.get("aoai_deployment_name")
aoai_api_version = os.environ.get("aoai_api_version")



from PIL import Image
import fitz  # PyMuPDF
import mimetypes

def crop_image_from_image(image_path, page_number, bounding_box):
    """
    Crops an image based on a bounding box.

    :param image_path: Path to the image file.
    :param page_number: The page number of the image to crop (for TIFF format).
    :param bounding_box: A tuple of (left, upper, right, lower) coordinates for the bounding box.
    :return: A cropped image.
    :rtype: PIL.Image.Image
    """
    with Image.open(image_path) as img:
        if img.format == "TIFF":
            # Open the TIFF image
            img.seek(page_number)
            img = img.copy()

        # The bounding box is expected to be in the format (left, upper, right, lower).
        cropped_image = img.crop(bounding_box)
        return cropped_image

def crop_image_from_pdf_page(pdf_path, page_number, bounding_box):
    """
    Crops a region from a given page in a PDF and returns it as an image.

    :param pdf_path: Path to the PDF file.
    :param page_number: The page number to crop from (0-indexed).
    :param bounding_box: A tuple of (x0, y0, x1, y1) coordinates for the bounding box.
    :return: A PIL Image of the cropped area.
    """
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)

    # Cropping the page. The rect requires the coordinates in the format (x0, y0, x1, y1).
    bbx = [x * 72 for x in bounding_box]
    rect = fitz.Rect(bbx)
    pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72), clip=rect)

    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    doc.close()

    return img

def crop_image_from_file(file_path, page_number, bounding_box):
    """
    Crop an image from a file.

    Args:
        file_path (str): The path to the file.
        page_number (int): The page number (for PDF and TIFF files, 0-indexed).
        bounding_box (tuple): The bounding box coordinates in the format (x0, y0, x1, y1).

    Returns:
        A PIL Image of the cropped area.
    """
    mime_type = mimetypes.guess_type(file_path)[0]

    if mime_type == "application/pdf":
        return crop_image_from_pdf_page(file_path, page_number, bounding_box)
    else:
        return crop_image_from_image(file_path, page_number, bounding_box)


import base64
from mimetypes import guess_type

# Function to encode a local image into data URL
def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"



MAX_TOKENS = 2000

def understand_image_with_gptv(api_base, api_key, deployment_name, api_version, image_path, caption):
    """
    Generates a description for an image using the GPT-4V model.

    Parameters:
    - api_base (str): The base URL of the API.
    - api_key (str): The API key for authentication.
    - deployment_name (str): The name of the deployment.
    - api_version (str): The version of the API.
    - image_path (str): The path to the image file.
    - caption (str): The caption for the image.

    Returns:
    - img_description (str): The generated description for the image.
    """
    return "this is sample image description"
    client = AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        base_url=f"{api_base}/openai/deployments/{deployment_name}"
    )

    data_url = local_image_to_data_url(image_path)

    # We send both image caption and the image body to GPTv for better understanding
    if caption != "":
        response = client.chat.completions.create(
                model=deployment_name,
                messages=[
                    { "role": "system", "content": "You are a helpful assistant." },
                    { "role": "user", "content": [
                        {
                            "type": "text",
                            "text": f"Describe this image (note: it has image caption: {caption}):"
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url
                            }
                        }
                    ] }
                ],
                max_tokens=MAX_TOKENS
            )

    else:
        response = client.chat.completions.create(
            model=deployment_name,
            messages=[
                { "role": "system", "content": "You are a helpful assistant." },
                { "role": "user", "content": [
                    {
                        "type": "text",
                        "text": "Describe this image:"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url
                        }
                    }
                ] }
            ],
            max_tokens=MAX_TOKENS
        )

    img_description = response.choices[0].message.content

    return img_description



import re
import pdb
def update_figure_description(md_content, img_description, idx):
    """
    Updates the figure description in the Markdown content.

    Args:
        md_content (str): The original Markdown content.
        img_description (str): The new description for the image.
        idx (int): The index of the figure.

    Returns:
        str: The updated Markdown content with the new figure description.
    """

    # The substring you're looking for
    start_substring = f"![](figures/{idx})"
    end_substring = "</figure>"
    new_string = f"<!-- FigureContent=\"{img_description}\" -->"

    new_md_content = md_content
    # Find the start and end indices of the part to replace
    start_index = md_content.find(start_substring)
    if start_index != -1:  # if start_substring is found
        start_index += len(start_substring)  # move the index to the end of start_substring
        end_index = md_content.find(end_substring, start_index)
        if end_index != -1:  # if end_substring is found
            # Replace the old string with the new string
            new_md_content = md_content[:start_index] + new_string + md_content[end_index:]

    return new_md_content

def update_figure_description(md_content, img_description, idx):
    """
    Updates the figure description in the Markdown content.

    Args:
        md_content (str): The original Markdown content.
        img_description (str): The new description for the image.
        idx (int): The index of the figure.

    Returns:
        str: The updated Markdown content with the new figure description.
    """

    figure_pattern = r"<figure>[\s\S]*?</figure>"
    figure_matches = list(re.finditer(figure_pattern, md_content))
    end_substring = "</figure>"
    # pdb.set_trace()
    if idx < len(figure_matches):
        start_index, end_index = figure_matches[idx].span()
        new_string = f"\n<!-- FigureContent=\"{img_description}\" -->\n"
        md_content = md_content[:end_index-len(end_substring)] + new_string + md_content[end_index-len(end_substring):]

    return md_content




import tempfile


def analyze_layout(input_file_path):
    """
    Analyzes the layout of a document and extracts figures along with their descriptions, then update the markdown output with the new description.

    Args:
        input_file_path (str): The path to the input document file.
        output_folder (str): The path to the output folder where the cropped images will be saved.

    Returns:
        str: The updated Markdown content with figure descriptions.

    """
    print("key++++++++++++++++++++++",doc_intelligence_key)
    output_folder =  tempfile.gettempdir()
    document_intelligence_client = DocumentIntelligenceClient(
        endpoint=doc_intelligence_endpoint,
        credential=AzureKeyCredential(doc_intelligence_key),
        headers={"x-ms-useragent":"sample-code-figure-understanding/1.0.0"},
    )
    poller = document_intelligence_client.begin_analyze_document(
        "prebuilt-layout", body=base64.b64decode(input_file_path), content_type="application/octet-stream", output_content_format=DocumentContentFormat.MARKDOWN
    )

    result = poller.result()
    md_content = result.content

    # print(md_content)
    # exit()
    if result.figures:
        print("Figures:")
        for idx, figure in enumerate(result.figures):
            figure_content = ""
            img_description = ""
            print(f"Figure #{idx} has the following spans: {figure.spans}")
            for i, span in enumerate(figure.spans):
                print(f"Span #{i}: {span}")
                figure_content += md_content[span.offset:span.offset + span.length]
            print(f"Original figure content in markdown: {figure_content}")

            # Note: figure bounding regions currently contain both the bounding region of figure caption and figure body
            if figure.caption:
                caption_region = figure.caption.bounding_regions
                print(f"\tCaption: {figure.caption.content}")
                print(f"\tCaption bounding region: {caption_region}")
                for region in figure.bounding_regions:
                    if region not in caption_region:
                        print(f"\tFigure body bounding regions: {region}")
                        # To learn more about bounding regions, see https://aka.ms/bounding-region
                        boundingbox = (
                                region.polygon[0],  # x0 (left)
                                region.polygon[1],  # y0 (top)
                                region.polygon[4],  # x1 (right)
                                region.polygon[5]   # y1 (bottom)
                            )
                        print(f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")
                        cropped_image = crop_image_from_file(input_file_path, region.page_number - 1, boundingbox) # page_number is 1-indexed

                        # Get the base name of the file
                        base_name = os.path.basename(input_file_path)
                        # Remove the file extension
                        file_name_without_extension = os.path.splitext(base_name)[0]

                        output_file = f"{file_name_without_extension}_cropped_image_{idx}.png"
                        cropped_image_filename = os.path.join(output_folder, output_file)

                        cropped_image.save(cropped_image_filename)
                        print(f"\tFigure {idx} cropped and saved as {cropped_image_filename}")
                        img_description += understand_image_with_gptv(aoai_api_base, aoai_api_key, aoai_deployment_name, aoai_api_version, cropped_image_filename, figure.caption.content)
                        print(f"\tDescription of figure {idx}: {img_description}")
            else:
                print("\tNo caption found for this figure.")
                for region in figure.bounding_regions:
                    print(f"\tFigure body bounding regions: {region}")
                    # To learn more about bounding regions, see https://aka.ms/bounding-region
                    boundingbox = (
                            region.polygon[0],  # x0 (left)
                            region.polygon[1],  # y0 (top
                            region.polygon[4],  # x1 (right)
                            region.polygon[5]   # y1 (bottom)
                        )
                    print(f"\tFigure body bounding box in (x0, y0, x1, y1): {boundingbox}")

                    cropped_image = crop_image_from_file(input_file_path, region.page_number - 1, boundingbox) # page_number is 1-indexed

                    # Get the base name of the file
                    base_name = os.path.basename(input_file_path)
                    # Remove the file extension
                    file_name_without_extension = os.path.splitext(base_name)[0]

                    output_file = f"{file_name_without_extension}_cropped_image_{idx}.png"
                    cropped_image_filename = os.path.join(output_folder, output_file)
                    # cropped_image_filename = f"data/cropped/image_{idx}.png"
                    cropped_image.save(cropped_image_filename)
                    print(f"\tFigure {idx} cropped and saved as {cropped_image_filename}")
                    img_description += understand_image_with_gptv(aoai_api_base, aoai_api_key, aoai_deployment_name, aoai_api_version, cropped_image_filename, "")
                    print(f"\tDescription of figure {idx}: {img_description}")

            # replace_figure_description(figure_content, img_description, idx)
            # print(md_content)
            md_content = update_figure_description(md_content, cropped_image_filename, idx)

    return md_content


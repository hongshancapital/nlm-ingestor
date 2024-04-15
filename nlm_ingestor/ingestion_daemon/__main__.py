import logging
import nlm_ingestor.ingestion_daemon.config as cfg
import os
import tempfile
import traceback
from flask import Flask, request, jsonify, make_response
from werkzeug.utils import secure_filename
from nlm_ingestor.ingestor import ingestor_api
from nlm_utils.utils import file_utils
import boto3

from nlm_ingestor.ingestor_utils.utils import normalize_kangxi_radicals

app = Flask(__name__)

# initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(cfg.log_level())

s3_client = boto3.client('s3')

@app.route('/', methods=['GET'])
def health_check():
    return 'Service is running', 200

@app.route('/api/parseDocument', methods=['POST'])
def parse_document(
    file=None,
    render_format: str = "all",
):
    render_format = request.args.get('renderFormat', 'all')
    use_new_indent_parser = request.args.get('useNewIndentParser', 'no')
    apply_ocr = request.args.get('applyOcr', 'no')
    original_filename = None
    filename = None
    tmp_file = None
    try:
        parse_options = {
            "parse_and_render_only": True,
            "render_format": render_format,
            "use_new_indent_parser": use_new_indent_parser == "yes",
            "parse_pages": (),
            "apply_ocr": apply_ocr == "yes"
        }
        # save the incoming file to a temporary location
        if 'file' in request.files:
            file = request.files['file']
            original_filename = file.filename
            filename = secure_filename(original_filename)
            tmp_file = create_temp_file(filename)
            file.save(tmp_file)
        elif 'filename' in request.form and 's3url' in request.form:
            original_filename = request.form.get('filename')
            filename = secure_filename(original_filename)
            tmp_file = create_temp_file(filename)
            s3url = request.form.get('s3url')
            bucket_name, file_key = parse_s3_url(s3url)
            s3_client.download_file(bucket_name, file_key, tmp_file)
        else:
            raise Exception("No file found in request")

        # calculate the file properties
        props = file_utils.extract_file_properties(tmp_file)
        print(f"Parsing document: '{original_filename}'")
        logger.info(f"Parsing document: '{original_filename}'")
        return_dict, _ = ingestor_api.ingest_document(
            filename,
            tmp_file,
            props["mimeType"],
            parse_options=parse_options,
        )

        return_dict = return_dict or {}

        if "result" in return_dict and "blocks" in return_dict["result"]:
            return_dict["result"]["blocks"] = [
                {
                    **block,
                    "sentences": [normalize_kangxi_radicals(sentence) for sentence in block["sentences"]]
                } if "sentences" in block else block
                for block in return_dict.get("result", {}).get("blocks", [])
            ]

        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
        return make_response(
            jsonify({"status": 200, "return_dict": return_dict}),
        )

    except Exception as e:
        print(f"error uploading file '{original_filename}', stacktrace: ", traceback.format_exc())
        logger.error(
            f"error uploading file '{original_filename}', stacktrace: {traceback.format_exc()}",
            exc_info=True,
        )
        status, rc, msg = "fail", 500, str(e)

    finally:
        if tmp_file and os.path.exists(tmp_file):
            os.unlink(tmp_file)
    return make_response(jsonify({"status": status, "reason": msg}), rc)


def create_temp_file(filename):
    """
    Create a temporary file with the same extension as the input filename.
    """
    _, file_extension = os.path.splitext(filename)
    tempfile_handler, temp_file = tempfile.mkstemp(suffix=file_extension)
    os.close(tempfile_handler)
    return temp_file


def parse_s3_url(s3url):
    """
    Parse the S3 URL to get the bucket name and file key.
    e.g. s3://somebucket/path/to/file.pdf -> ('somebucket', 'path/to/file.pdf')
    """
    parts = s3url.split('/')
    bucket_name = parts[2]
    file_key = '/'.join(parts[3:])
    return bucket_name, file_key


def main():
    logger.info("Starting ingestor service..")
    app.run(host="0.0.0.0", port=5001, debug=False)

if __name__ == "__main__":
    main()



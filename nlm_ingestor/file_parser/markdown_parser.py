import json
import logging
import os

import mistune
from mistune.core import BlockState
from nlm_ingestor.ingestor_utils.utils import sent_tokenize, safe_open

import nlm_ingestor.ingestion_daemon.config as cfg
from nlm_ingestor.ingestor_utils.ing_named_tuples import LineStyle
from nlm_ingestor.ingestor.visual_ingestor import block_renderer


# initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(cfg.log_level())

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setLevel(cfg.log_level())
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


def get_token_text(token):
    """Extract plain text from a mistune 3.x token by traversing its children."""
    if 'raw' in token and isinstance(token['raw'], str) and 'children' not in token:
        return token['raw']
    if 'children' not in token:
        return ''
    texts = []
    for child in token['children']:
        if child['type'] == 'text':
            texts.append(child['raw'])
        elif child['type'] == 'softbreak':
            texts.append('\n')
        elif child['type'] == 'linebreak':
            texts.append('\n')
        elif 'children' in child or 'raw' in child:
            texts.append(get_token_text(child))
    return ''.join(texts)


def parse_markdown_to_blocks(markdown_text):
    state = BlockState()
    html_str, state = mistune.html.parse(markdown_text, state)
    mistune_tokens = state.tokens

    # Determine level from token context (heading level, list depth etc.)
    current_level = 0

    blocks = []
    cur_table_idx = 0
    for idx, mistune_token in enumerate(mistune_tokens):
        token_type = mistune_token["type"]

        if token_type == "blank_line":
            continue

        # Extract level from attrs
        if "attrs" in mistune_token and mistune_token["attrs"]:
            if token_type == "heading":
                current_level = mistune_token["attrs"].get("level", 0)
            elif token_type == "list":
                current_level = mistune_token["attrs"].get("depth", 0)

        token_handlers = {
            "paragraph": convert_mistune_to_paragraph,
            "table": convert_mistune_to_table,
            "heading": convert_mistune_to_header,
            "list": convert_mistune_to_list_item,
            "block_quote": convert_mistune_to_paragraphs,
            "block_code": convert_mistune_to_code_paragraph,
        }

        handler = token_handlers.get(token_type)
        if handler is None:
            continue

        cur_blocks = handler(mistune_token, current_level)

        for block in cur_blocks:
            block["block_idx"] = idx
            block["page_idx"] = 0
            if token_type == "table":
                block["table_idx"] = cur_table_idx
            blocks.append(block)
        if token_type == "table":
            cur_table_idx += 1

    return blocks, html_str


def convert_mistune_to_paragraph(token, level):
    return [
        {
            "block_type": "para",
            "block_text": get_token_text(token),
            "block_sents": sent_tokenize(get_token_text(token)),
            "level": level,
        },
    ]


def convert_mistune_to_code_paragraph(token, level):
    return [
        {
            "block_type": "para",
            "block_text": token["raw"],
            "block_sents": sent_tokenize(token["raw"]),
            "level": level,
        },
    ]


def convert_mistune_to_table(token, level):
    blocks = []
    for child in token["children"]:
        if child["type"] == "table_head":
            cell_values = [get_token_text(x) for x in child["children"]]
            block = {
                "block_type": "table_row",
                "is_header_group": True,
                "col_spans": [1] * len(child["children"]),
                "cell_values": cell_values,
                "block_text": " ".join(cell_values),
                "level": level,
            }
            blocks.append(block)
        elif child["type"] == "table_body":
            for row in child["children"]:
                cell_values = [get_token_text(x) for x in row["children"]]
                block = {
                    "block_type": "table_row",
                    "cell_values": cell_values,
                    "block_text": " ".join(cell_values),
                    "level": level,
                }
                blocks.append(block)

    blocks[0]["is_table_start"] = True
    blocks[-1]["is_table_end"] = True

    return blocks


def convert_mistune_to_header(token, level):
    blocks = [
        {
            "block_type": "header",
            "block_text": get_token_text(token),
            "level": level - 1,
        },
    ]
    return blocks


def convert_mistune_to_list_item(token, level):
    blocks = []
    for child in token["children"]:
        if child["type"] == "list":
            continue
        # child is a block_text token whose first child has the raw text
        block_text_token = child["children"][0]
        item_text = get_token_text(block_text_token)
        block = {
            "block_type": "list_item",
            "block_text": item_text,
            "block_sents": [item_text],
            "level": level,
        }
        blocks.append(block)
    return blocks


def convert_mistune_to_paragraphs(token, level):
    print("token is:", token)
    blocks = []
    for child in token["children"]:
        if child["type"] == "paragraph":
            block = {
                "block_type": "para",
                "block_text": get_token_text(child),
                "block_sents": sent_tokenize(get_token_text(child)),
                "level": level,
            }
            blocks.append(block)
        elif "raw" in child:
            block = {
                "block_type": "para",
                "block_text": child["raw"],
                "block_sents": sent_tokenize(child["raw"]),
                "level": level,
            }
            blocks.append(block)
    return blocks


class MarkdownDocument:
    def __init__(self, doc_location):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        markdown_text = ""
        with safe_open(doc_location) as file:
            markdown_text = file.read()
        self.blocks, self.html_str = parse_markdown_to_blocks(markdown_text)
        for block in self.blocks:
            block["block_class"] = ""
        self.line_style_classes = {}
        self.class_levels = {}

        br = block_renderer.BlockRenderer(self)
        self.json_dict = br.render_json()

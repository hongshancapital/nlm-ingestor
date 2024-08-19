import json
import re

import numpy as np
from nltk import load
from nltk import PunktSentenceTokenizer


nltk_abbs = load("tokenizers/punkt/{}.pickle".format("english"))._params.abbrev_types


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


nlm_abbs = {
    "u.s",
    "u.s.a",
    "n.w",
    "p.o",
    "po",
    "st",
    "ave",
    "blvd",
    "ctr",
    "cir",
    "ct",
    "dr",
    "mtn",
    "apt",
    "hwy",
    "esq",
    "fig",
    "no",
    "sec",
    "n.a",
    "s.a.b",
    "non-u.s",
    "cap",
    'u.s.c',
    "ste",
}

nlm_special_abbs = {
    "inc",
}

abbs = nltk_abbs | nlm_abbs

nltk_tokenzier = PunktSentenceTokenizer()

rules = []

for abb in abbs:
    # match start of the sentence
    pattern = fr"^{abb}.\s"
    replaced = f"{abb}_ "

    # case insensitive replacement for synonyms
    rule = re.compile(pattern, re.IGNORECASE)
    rules.append((rule, replaced))

    # match token in sentence
    pattern = fr"\s{abb}.\s"
    replaced = f" {abb}_ "

    # case insensitive replacement for synonyms
    rule = re.compile(pattern, re.IGNORECASE)
    rules.append((rule, replaced))

for abb in nlm_special_abbs:
    pattern = fr"{abb}\."
    replaced = f"{abb}_"
    rule = re.compile(pattern, re.IGNORECASE)
    rules.append((rule, replaced))

# match content inside brackets
# (?<=\() ==> starts with "("
# ([^)]+) ==> repeat not ")"
# (?=\))") ==> ends with ")"
bracket_rule = re.compile(r"(?<=\()([^)]+)(?=\))")
space_rule = re.compile(r"\s([.'](?:\s|$|\D))", re.IGNORECASE)  # Remove any space between punctuations (.')
quotation_pattern = re.compile(r'[”“"‘’\']')


def sent_tokenize(org_texts):
    if not org_texts:
        return org_texts

    sents = []

    # in case org_texts has \n, break it into multiple paragraph
    # edge case for html and markdown
    for org_text in org_texts.split("\n"):
        org_text = space_rule.sub(r'\1', org_text)
        modified_text = re.sub(r'^([.,?!]\s+)+', "", org_text)  # To handle bug https://github.com/nltk/nltk/issues/2925
        orig_offset = abs(len(org_text) - len(modified_text))

        # do not break bracket
        for span_group in bracket_rule.finditer(modified_text):
            start_byte, end_byte = span_group.span()
            span = modified_text[start_byte:end_byte]
            # skip this logic when span is too big? disabled for now
            # if len(span.split()) >= 10:
            #     continue
            modified_text = modified_text.replace(
                f"({span})", f"_{span.replace('.','_')}_",
            )

        for rule, replaced in rules:
            modified_text = rule.sub(replaced, modified_text)
        # Normalize all the quotation.
        modified_text = quotation_pattern.sub("\"", modified_text)

        modified_sents = nltk_tokenzier.tokenize(modified_text)

        offset = orig_offset
        sent_idx = 0
        while offset < len(modified_text) and sent_idx < len(modified_sents):
            if modified_text[offset] == " ":
                offset += 1
                continue

            # cut org_text based on lengths of modified_sent
            modified_sent = modified_sents[sent_idx]
            sents.append(org_text[offset: offset + len(modified_sent)])

            offset += len(modified_sent)
            sent_idx += 1
    if len(sents) >= 2 and re.match(r"^.\.$", sents[0]):
        sents[1] = sents[0] + " " + sents[1]
        sents = sents[1:]

    return sents


def divide_list_into_chunks(lst, n):
    # looping till length l
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def normalize(X):
    norms = np.einsum("ij,ij->i", X, X)
    np.sqrt(norms, norms)

    X /= norms[:, np.newaxis]
    return X


def detect_block_center_aligned(block, page_width):
    center_location = block["box_style"][1] + block["box_style"][3] / 2
    center_aligned = abs(center_location - page_width / 2) < page_width * 0.01
    width_check = block["box_style"][3] * 2 < page_width
    return center_aligned and width_check


def detect_block_center_of_page(block, page_height):
    bottom = block["box_style"][0] + block["box_style"][4]
    center_of_page = (page_height / 3) <= bottom <= ((2 * page_height) / 3)
    return center_of_page


def check_char_is_word_boundary(c):
    if c.isalnum():
        return False
    if c in ['-', '_']:
        return False
    return True

def blocks_to_sents(blocks, flatten_merged_table=False, debug=False):
    block_texts = []
    block_info = []
    header_block_idx = -1
    header_match_idx = -1
    header_match_idx_offset = -1
    header_block_text = ""
    is_rendering_table = False
    is_rendering_merged_cells = False
    table_idx = 0
    levels = []
    prev_header = None
    block_idx = 0
    for block_idx, block in enumerate(blocks):
        block_type = block["block_type"]
        if block_type == "header":
            if debug:
                print("---", block["level"], block["block_text"])
            header_block_text = block["block_text"]
            header_block_idx = block["block_idx"]
            header_match_idx = header_match_idx_offset + 1
            if prev_header and block["level"] <= prev_header['level'] and len(levels) > 0:
                while len(levels) > 0 and levels[-1]["level"] >= block["level"]:
                    if debug:
                        print("<<", levels[-1]["level"], levels[-1]["block_text"])
                    levels.pop(-1)
            if debug:
                print(">>", block["block_text"])
            levels.append(block)
            prev_header = block
            if debug:
                print("-", [str(level['level']) + "-" + level['block_text'] for level in levels])
        block["header_text"] = header_block_text
        block["header_block_idx"] = header_block_idx
        block["header_match_idx"] = header_match_idx
        block["block_idx"] = block_idx

        level_chain = []
        for level in levels:
            level_chain.append({"block_idx": level["block_idx"], "block_text": level["block_text"]})
        # remove a level for header
        if block_type == "header":
            level_chain = level_chain[:-1]
        level_chain.reverse()
        block["level_chain"] = level_chain

        # if block_type == "header" or block_type == "table_row":
        if (
                block_type == "header"
                and not is_rendering_table and 'is_table_start' not in block
        ):
            block_texts.append(block["block_text"])
            # append text from next block to header block
            # TODO: something happened here, it messed up the match_text
            # if block_type == "header" and block_idx + 1 < len(blocks):
            #     block[
            #         "block_text"
            #     ] += blocks[block_idx+1]['block_text']

            block_info.append(block)
            header_match_idx_offset += 1
        elif (
                block_type == "list_item" or block_type == "para" or block_type == "numbered_list_item"
        ) and not is_rendering_table:
            block_sents = block["block_sents"]
            header_match_idx_offset += len(block_sents)
            for sent in block_sents:
                block_texts.append(sent)
                block_info.append(block)
        elif 'is_table_start' in block:
            is_rendering_table = True
            if 'has_merged_cells' in block:
                is_rendering_merged_cells = True
        elif 'is_table_start' not in block and not is_rendering_table and block_type == "table_row":
            block_info.append(block)
            block_texts.append(block["block_text"])
            header_match_idx_offset += 1

        if is_rendering_table:
            if is_rendering_merged_cells and "effective_para" in block and flatten_merged_table:
                eff_header_block = block["effective_header"]
                eff_para_block = block["effective_para"]

                eff_header_block["header_text"] = block["header_text"]
                eff_header_block["header_block_idx"] = block["block_idx"]
                eff_header_block["header_match_idx"] = header_match_idx_offset + 1
                eff_header_block["level"] = block["level"] + 1
                eff_header_block["level_chain"] = block["level_chain"]

                eff_para_block["header_block_idx"] = block["block_idx"]
                eff_para_block["header_match_idx"] = header_match_idx_offset + 1
                eff_para_block["level"] = block["level"] + 2
                eff_para_block["level_chain"] = [
                                {
                                    "block_idx": eff_header_block["block_idx"],
                                    "block_text": eff_header_block["block_text"],
                                },
                ] + eff_header_block["level_chain"]
                header_match_idx_offset += 1
                block_info.append(block["effective_header"])
                block_texts.append(block["effective_header"]["block_text"])
                for sent in block["effective_para"]["block_sents"]:
                    block_texts.append(sent)
                    block_info.append(block["effective_para"])
                header_match_idx_offset += len(block["effective_para"]["block_sents"])
            else:
                block["table_idx"] = table_idx
                block_info.append(block)
                block_texts.append(block["block_text"])
                header_match_idx_offset += 1

        if 'is_table_end' in block:
            is_rendering_table = False
            table_idx += 1

    return block_texts, block_info


def get_block_texts(blocks):
    block_texts = []
    block_info = []
    for block in blocks:
        block_type = block["block_type"]
        if (
            block_type == "list_item"
            or block_type == "para"
            or block_type == "numbered_list_item"
            or block_type == "header"
        ):
            block_texts.append(block["block_text"])
            block_info.append(block)
    return block_texts, block_info


ARABIC_NUMBER_PATTERN = re.compile(r"^\d+(\.\d+)?$")


def is_arabic_number(text: str):
    """
    Check if the text is an arabic number.
    """
    return bool(ARABIC_NUMBER_PATTERN.match(text))


KANGXI_RADICAL_CHINESE_CHARACTER_MAPPING = (
    ('\u2F00', '\u4E00' ), # ⼀ -> 一
    ('\u2F01', '\u4E28' ), # ⼁ -> 丨
    ('\u2F02', '\u4E36' ), # ⼂ -> 丶
    ('\u2F03', '\u4E3F' ), # ⼃ -> 丿
    ('\u2F04', '\u4E59' ), # ⼄ -> 乙
    ('\u2F05', '\u4E85' ), # ⼅ -> 亅
    ('\u2F06', '\u4E8C' ), # ⼆ -> 二
    ('\u2F07', '\u4EA0' ), # ⼇ -> 亠
    ('\u2F08', '\u4EBA' ), # ⼈ -> 人
    ('\u2F09', '\u513F' ), # ⼉ -> 儿
    ('\u2F0A', '\u5165' ), # ⼊ -> 入
    ('\u2F0B', '\u516B' ), # ⼋ -> 八
    ('\u2F0C', '\u5182' ), # ⼌ -> 冂
    ('\u2F0D', '\u5196' ), # ⼍ -> 冖
    ('\u2F0E', '\u51AB' ), # ⼎ -> 冫
    ('\u2F0F', '\u51E0' ), # ⼏ -> 几
    ('\u2F10', '\u51F5' ), # ⼐ -> 凵
    ('\u2F11', '\u5200' ), # ⼑ -> 刀
    ('\u2F12', '\u529B' ), # ⼒ -> 力
    ('\u2F13', '\u52F9' ), # ⼓ -> 勹
    ('\u2F14', '\u5315' ), # ⼔ -> 匕
    ('\u2F15', '\u531A' ), # ⼕ -> 匚
    ('\u2F16', '\u5338' ), # ⼖ -> 匸
    ('\u2F17', '\u5341' ), # ⼗ -> 十
    ('\u2F18', '\u535C' ), # ⼘ -> 卜
    ('\u2F19', '\u5369' ), # ⼙ -> 卩
    ('\u2F1A', '\u5382' ), # ⼚ -> 厂
    ('\u2F1B', '\u53B6' ), # ⼛ -> 厶
    ('\u2F1C', '\u53C8' ), # ⼜ -> 又
    ('\u2F1D', '\u53E3' ), # ⼝ -> 口
    ('\u2F1E', '\u56D7' ), # ⼞ -> 囗
    ('\u2F1F', '\u571F' ), # ⼟ -> 土
    ('\u2F20', '\u58EB' ), # ⼠ -> 士
    ('\u2F21', '\u5902' ), # ⼡ -> 夂
    ('\u2F22', '\u590A' ), # ⼢ -> 夊
    ('\u2F23', '\u5915' ), # ⼣ -> 夕
    ('\u2F24', '\u5927' ), # ⼤ -> 大
    ('\u2F25', '\u5973' ), # ⼥ -> 女
    ('\u2F26', '\u5B50' ), # ⼦ -> 子
    ('\u2F27', '\u5B80' ), # ⼧ -> 宀
    ('\u2F28', '\u5BF8' ), # ⼨ -> 寸
    ('\u2F29', '\u5C0F' ), # ⼩ -> 小
    ('\u2F2A', '\u5C22' ), # ⼪ -> 尢
    ('\u2F2B', '\u5C38' ), # ⼫ -> 尸
    ('\u2F2C', '\u5C6E' ), # ⼬ -> 屮
    ('\u2F2D', '\u5C71' ), # ⼭ -> 山
    ('\u2F2E', '\u5DDB' ), # ⼮ -> 巛
    ('\u2F2F', '\u5DE5' ), # ⼯ -> 工
    ('\u2F30', '\u5DF1' ), # ⼰ -> 己
    ('\u2F31', '\u5DFE' ), # ⼱ -> 巾
    ('\u2F32', '\u5E72' ), # ⼲ -> 干
    ('\u2F33', '\u5E7A' ), # ⼳ -> 幺
    ('\u2F34', '\u5E7F' ), # ⼴ -> 广
    ('\u2F35', '\u5EF4' ), # ⼵ -> 廴
    ('\u2F36', '\u5EFE' ), # ⼶ -> 廾
    ('\u2F37', '\u5F0B' ), # ⼷ -> 弋
    ('\u2F38', '\u5F13' ), # ⼸ -> 弓
    ('\u2F39', '\u5F50' ), # ⼹ -> 彐
    ('\u2F3A', '\u5F61' ), # ⼺ -> 彡
    ('\u2F3B', '\u5F73' ), # ⼻ -> 彳
    ('\u2F3C', '\u5FC3' ), # ⼼ -> 心
    ('\u2F3D', '\u6208' ), # ⼽ -> 戈
    ('\u2F3E', '\u6236' ), # ⼾ -> 戶
    ('\u2F3F', '\u624B' ), # ⼿ -> 手
    ('\u2F40', '\u652F' ), # ⽀ -> 支
    ('\u2F41', '\u6534' ), # ⽁ -> 攴
    ('\u2F42', '\u6587' ), # ⽂ -> 文
    ('\u2F43', '\u6597' ), # ⽃ -> 斗
    ('\u2F44', '\u65A4' ), # ⽄ -> 斤
    ('\u2F45', '\u65B9' ), # ⽅ -> 方
    ('\u2F46', '\u65E0' ), # ⽆ -> 无
    ('\u2F47', '\u65E5' ), # ⽇ -> 日
    ('\u2F48', '\u66F0' ), # ⽈ -> 曰
    ('\u2F49', '\u6708' ), # ⽉ -> 月
    ('\u2F4A', '\u6728' ), # ⽊ -> 木
    ('\u2F4B', '\u6B20' ), # ⽋ -> 欠
    ('\u2F4C', '\u6B62' ), # ⽌ -> 止
    ('\u2F4D', '\u6B79' ), # ⽍ -> 歹
    ('\u2F4E', '\u6BB3' ), # ⽎ -> 殳
    ('\u2F4F', '\u6BCB' ), # ⽏ -> 毋
    ('\u2F50', '\u6BD4' ), # ⽐ -> 比
    ('\u2F51', '\u6BDB' ), # ⽑ -> 毛
    ('\u2F52', '\u6C0F' ), # ⽒ -> 氏
    ('\u2F53', '\u6C14' ), # ⽓ -> 气
    ('\u2F54', '\u6C34' ), # ⽔ -> 水
    ('\u2F55', '\u706B' ), # ⽕ -> 火
    ('\u2F56', '\u722A' ), # ⽖ -> 爪
    ('\u2F57', '\u7236' ), # ⽗ -> 父
    ('\u2F58', '\u723B' ), # ⽘ -> 爻
    ('\u2F59', '\u723F' ), # ⽙ -> 爿
    ('\u2F5A', '\u7247' ), # ⽚ -> 片
    ('\u2F5B', '\u7259' ), # ⽛ -> 牙
    ('\u2F5C', '\u725B' ), # ⽜ -> 牛
    ('\u2F5D', '\u72AC' ), # ⽝ -> 犬
    ('\u2F5E', '\u7384' ), # ⽞ -> 玄
    ('\u2F5F', '\u7389' ), # ⽟ -> 玉
    ('\u2F60', '\u74DC' ), # ⽠ -> 瓜
    ('\u2F61', '\u74E6' ), # ⽡ -> 瓦
    ('\u2F62', '\u7518' ), # ⽢ -> 甘
    ('\u2F63', '\u751F' ), # ⽣ -> 生
    ('\u2F64', '\u7528' ), # ⽤ -> 用
    ('\u2F65', '\u7530' ), # ⽥ -> 田
    ('\u2F66', '\u758B' ), # ⽦ -> 疋
    ('\u2F67', '\u7592' ), # ⽧ -> 疒
    ('\u2F68', '\u7676' ), # ⽨ -> 癶
    ('\u2F69', '\u767D' ), # ⽩ -> 白
    ('\u2F6A', '\u76AE' ), # ⽪ -> 皮
    ('\u2F6B', '\u76BF' ), # ⽫ -> 皿
    ('\u2F6C', '\u76EE' ), # ⽬ -> 目
    ('\u2F6D', '\u77DB' ), # ⽭ -> 矛
    ('\u2F6E', '\u77E2' ), # ⽮ -> 矢
    ('\u2F6F', '\u77F3' ), # ⽯ -> 石
    ('\u2F70', '\u793A' ), # ⽰ -> 示
    ('\u2F71', '\u79B8' ), # ⽱ -> 禸
    ('\u2F72', '\u79BE' ), # ⽲ -> 禾
    ('\u2F73', '\u7A74' ), # ⽳ -> 穴
    ('\u2F74', '\u7ACB' ), # ⽴ -> 立
    ('\u2F75', '\u7AF9' ), # ⽵ -> 竹
    ('\u2F76', '\u7C73' ), # ⽶ -> 米
    ('\u2F77', '\u7CF8' ), # ⽷ -> 糸
    ('\u2F78', '\u7F36' ), # ⽸ -> 缶
    ('\u2F79', '\u7F51' ), # ⽹ -> 网
    ('\u2F7A', '\u7F8A' ), # ⽺ -> 羊
    ('\u2F7B', '\u7FBD' ), # ⽻ -> 羽
    ('\u2F7C', '\u8001' ), # ⽼ -> 老
    ('\u2F7D', '\u800C' ), # ⽽ -> 而
    ('\u2F7E', '\u8012' ), # ⽾ -> 耒
    ('\u2F7F', '\u8033' ), # ⽿ -> 耳
    ('\u2F80', '\u807F' ), # ⾀ -> 聿
    ('\u2F81', '\u8089' ), # ⾁ -> 肉
    ('\u2F82', '\u81E3' ), # ⾂ -> 臣
    ('\u2F83', '\u81EA' ), # ⾃ -> 自
    ('\u2F84', '\u81F3' ), # ⾄ -> 至
    ('\u2F85', '\u81FC' ), # ⾅ -> 臼
    ('\u2F86', '\u820C' ), # ⾆ -> 舌
    ('\u2F87', '\u821B' ), # ⾇ -> 舛
    ('\u2F88', '\u821F' ), # ⾈ -> 舟
    ('\u2F89', '\u826E' ), # ⾉ -> 艮
    ('\u2F8A', '\u8272' ), # ⾊ -> 色
    ('\u2F8B', '\u8278' ), # ⾋ -> 艸
    ('\u2F8C', '\u864D' ), # ⾌ -> 虍
    ('\u2F8D', '\u866B' ), # ⾍ -> 虫
    ('\u2F8E', '\u8840' ), # ⾎ -> 血
    ('\u2F8F', '\u884C' ), # ⾏ -> 行
    ('\u2F90', '\u8863' ), # ⾐ -> 衣
    ('\u2F91', '\u897E' ), # ⾑ -> 襾
    ('\u2F92', '\u898B' ), # ⾒ -> 見
    ('\u2F93', '\u89D2' ), # ⾓ -> 角
    ('\u2F94', '\u8A00' ), # ⾔ -> 言
    ('\u2F95', '\u8C37' ), # ⾕ -> 谷
    ('\u2F96', '\u8C46' ), # ⾖ -> 豆
    ('\u2F97', '\u8C55' ), # ⾗ -> 豕
    ('\u2F98', '\u8C78' ), # ⾘ -> 豸
    ('\u2F99', '\u8C9D' ), # ⾙ -> 貝
    ('\u2F9A', '\u8D64' ), # ⾚ -> 赤
    ('\u2F9B', '\u8D70' ), # ⾛ -> 走
    ('\u2F9C', '\u8DB3' ), # ⾜ -> 足
    ('\u2F9D', '\u8EAB' ), # ⾝ -> 身
    ('\u2F9E', '\u8ECA' ), # ⾞ -> 車
    ('\u2F9F', '\u8F9B' ), # ⾟ -> 辛
    ('\u2FA0', '\u8FB0' ), # ⾠ -> 辰
    ('\u2FA1', '\u8FB5' ), # ⾡ -> 辵
    ('\u2FA2', '\u9091' ), # ⾢ -> 邑
    ('\u2FA3', '\u9149' ), # ⾣ -> 酉
    ('\u2FA4', '\u91C6' ), # ⾤ -> 釆
    ('\u2FA5', '\u91CC' ), # ⾥ -> 里
    ('\u2FA6', '\u91D1' ), # ⾦ -> 金
    ('\u2FA7', '\u9577' ), # ⾧ -> 長
    ('\u2FA8', '\u9580' ), # ⾨ -> 門
    ('\u2FA9', '\u961C' ), # ⾩ -> 阜
    ('\u2FAA', '\u96B6' ), # ⾪ -> 隶
    ('\u2FAB', '\u96B9' ), # ⾫ -> 隹
    ('\u2FAC', '\u96E8' ), # ⾬ -> 雨
    ('\u2FAD', '\u9752' ), # ⾭ -> 青
    ('\u2FAE', '\u975E' ), # ⾮ -> 非
    ('\u2FAF', '\u9762' ), # ⾯ -> 面
    ('\u2FB0', '\u9769' ), # ⾰ -> 革
    ('\u2FB1', '\u97CB' ), # ⾱ -> 韋
    ('\u2FB2', '\u97ED' ), # ⾲ -> 韭
    ('\u2FB3', '\u97F3' ), # ⾳ -> 音
    ('\u2FB4', '\u9801' ), # ⾴ -> 頁
    ('\u2FB5', '\u98A8' ), # ⾵ -> 風
    ('\u2FB6', '\u98DB' ), # ⾶ -> 飛
    ('\u2FB7', '\u98DF' ), # ⾷ -> 食
    ('\u2FB8', '\u9996' ), # ⾸ -> 首
    ('\u2FB9', '\u9999' ), # ⾹ -> 香
    ('\u2FBA', '\u99AC' ), # ⾺ -> 馬
    ('\u2FBB', '\u9AA8' ), # ⾻ -> 骨
    ('\u2FBC', '\u9AD8' ), # ⾼ -> 高
    ('\u2FBD', '\u9ADF' ), # ⾽ -> 髟
    ('\u2FBE', '\u9580' ), # ⾾ -> 門
    ('\u2FBF', '\u9B2F' ), # ⾿ -> 鬯
    ('\u2FC0', '\u9B32' ), # ⿀ -> 鬲
    ('\u2FC1', '\u9B3C' ), # ⿁ -> 鬼
    ('\u2FC2', '\u9B5A' ), # ⿂ -> 魚
    ('\u2FC3', '\u9CE5' ), # ⿃ -> 鳥
    ('\u2FC4', '\u9E75' ), # ⿄ -> 鹵
    ('\u2FC5', '\u9E7F' ), # ⿅ -> 鹿
    ('\u2FC6', '\u9EA5' ), # ⿆ -> 麥
    ('\u2FC7', '\u9EBB' ), # ⿇ -> 麻
    ('\u2FC8', '\u9EC3' ), # ⿈ -> 黃
    ('\u2FC9', '\u9ECD' ), # ⿉ -> 黍
    ('\u2FCA', '\u9ED1' ), # ⿊ -> 黑
    ('\u2FCB', '\u9EF9' ), # ⿋ -> 黹
    ('\u2FCC', '\u9EFD' ), # ⿌ -> 黽
    ('\u2FCD', '\u9F0E' ), # ⿍ -> 鼎
    ('\u2FCE', '\u9F13' ), # ⿎ -> 鼓
    ('\u2FCF', '\u9F20' ), # ⿏ -> 鼠
    ('\u2FD0', '\u9F3B' ), # ⿐ -> 鼻
    ('\u2FD1', '\u9F4A' ), # ⿑ -> 齊
    ('\u2FD2', '\u9F52' ), # ⿒ -> 齒
    ('\u2FD3', '\u9F8D' ), # ⿓ -> 龍
    ('\u2FD4', '\u9F9C' ), # ⿔ -> 龜
    ('\u2FD5', '\u9FA0' ), # ⿕ -> 龠
)


def normalize_kangxi_radicals(text: str) -> str:
    """
    Replace Kangxi radicals with Chinese characters in the Kangxi radicals list.
    Check https://en.wikipedia.org/wiki/Kangxi_radical for more information.
    """
    for kangxi_radical, chinese_character in KANGXI_RADICAL_CHINESE_CHARACTER_MAPPING:
        text = text.replace(kangxi_radical, chinese_character)
    return text


def is_integer(text: str | None) -> bool:
    """
    Check if all the characters in the text are integers.
    """
    return bool(re.fullmatch(r'\d+', text or ''))

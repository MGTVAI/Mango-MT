import pandas as pd


def parse_srt2list(srt_data, dataframe=False):
    """Parse SRT data into blocks with sequence number, timestamp, and text."""
    # 移除 BOM 字符
    srt_data = srt_data.replace("\ufeff", "")
    lines = srt_data.strip().split("\n")
    blocks = []
    i = 0

    while i < len(lines):
        block = {"seq_num": "", "timestamp": None, "text": ""}

        # Get sequence number
        if i < len(lines) and lines[i].strip().isdigit():
            block["seq_num"] = lines[i].strip()
            i += 1
        else:
            # Skip empty lines
            if i < len(lines) and lines[i].strip() == "":
                i += 1
                continue
            # break

        # Get timestamp only if it exists
        if i < len(lines) and "-->" in lines[i]:
            block["timestamp"] = lines[i].strip()
            i += 1
        else:
            # If no timestamp, continue to text
            pass

        # Get text
        text_lines = []
        while i < len(lines) and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1

        block["text"] = "\n".join(text_lines)
        blocks.append(block)

        # Skip empty line between blocks
        if i < len(lines) and lines[i].strip() == "":
            i += 1

    if dataframe:
        return pd.DataFrame(blocks)
    return blocks

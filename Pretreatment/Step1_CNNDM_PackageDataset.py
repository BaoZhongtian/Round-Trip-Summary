import os
import json
import tqdm
import hashlib

dm_single_close_quote = u'\u2019'  # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]


def hashhex(s):
    """Returns a heximal formated SHA1 hash of the input string."""
    h = hashlib.sha1()
    h.update(s.encode('utf-8'))
    return h.hexdigest()


def get_url_hashes(url_list):
    return [hashhex(url) for url in url_list]


def read_text_file(text_file):
    lines = []
    with open(text_file, "r", encoding='UTF-8') as f:
        for line in f:
            lines.append(line.strip())
    return lines


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line: return line
    if line == "": return line
    if line[-1] in END_TOKENS: return line
    # print line[-1]
    return line + " ."


def get_art_abs(story_file):
    lines = read_text_file(story_file)
    # Lowercase everything
    lines = [line.lower() for line in lines]

    # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for idx, line in enumerate(lines):
        if line == "":
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = ' '.join(article_lines)
    # Make abstract into a signle string, putting <s> and </s> tags around the sentences
    abstract = ' '.join(highlights)
    return article, abstract


if __name__ == '__main__':
    cnn_path = 'C:/ProjectData/CNNDM_Dataset/cnn_stories/cnn/stories/'
    dm_path = 'C:/ProjectData/CNNDM_Dataset/dailymail_stories/dailymail/stories'
    url_list = read_text_file('C:/ProjectData/CNNDM_Dataset/url_lists/all_val.txt')
    save_name = 'CNNDM_val.json'
    url_hashes = get_url_hashes(url_list)

    total_sample = []

    for filename in tqdm.tqdm(url_hashes):
        if os.path.exists(os.path.join(cnn_path, filename + '.story')):
            filepath = os.path.join(cnn_path, filename + '.story')
        if os.path.exists(os.path.join(dm_path, filename + '.story')):
            filepath = os.path.join(dm_path, filename + '.story')
        article, summary = get_art_abs(filepath)

        current_sample = {'filename': filename, 'article': article, 'summary': summary}
        total_sample.append(current_sample)
    json.dump(total_sample, open(save_name, 'w'))

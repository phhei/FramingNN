"""
Script to convert the Media-Frames-Corpus, provided as json, in a format which fits the Webis-format (csv)
media_frames = {"Economic",
                    "Capacity and resources",
                    "Morality", "Fairness and equality",
                    "Legality, constitutionality and jurisprudence",
                    "Policy prescription and evaluation",
                    "Crime and punishment",
                    "Security and defense",
                    "Health and safety",
                    "Quality of life",
                    "Cultural identity",
                    "Public opinion",
                    "Political",
                    "External regulation and reputation",
                    "Other"}
"""

import csv
import json
import pathlib
from typing import List, Tuple

import nltk
from regex import regex

nltk.download("punkt")


def fetch_sentences_of_span(whole_text, start, end) -> Tuple[List[str], List[int]]:
    f_sentences = whole_text.split("\n\n", 3)
    if len(f_sentences) <= 2:
        return f_sentences, []
    else:
        sentences_nltk = f_sentences[-1]
        f_sentences = f_sentences[:-1]
        f_sentences.extend(nltk.sent_tokenize(text=sentences_nltk, language="english"))
        return f_sentences, [fi for fi, s_fi in enumerate(f_sentences)
                             if (whole_text.index(s_fi) <= start < whole_text.index(s_fi) + len(s_fi)) or
                             (whole_text.index(s_fi) < end <= whole_text.index(s_fi) + len(s_fi))]


if __name__ == "__main__":
    codes = pathlib.Path("codes.json")
    data_files = [
        pathlib.Path("immigration.json"),
        #pathlib.Path("samesex.json"),
        pathlib.Path("smoking.json")
    ]

    use_only_gold_labels = False  # Naderi and Hirst tried it with True
    filter_not_media_frames = True  # Naderi and Hirst said False
    exact_frames = True  # Naderi and Hirst set True
    extract_sentences = True  # Naderi and Hirst set True
                                # - True: indicates the broad context regarding to our paper
                                # - False: indicates the narrow context regarding to our paper
    no_premise = False  # Naderi and Hirst set True - was not their problem
    ignore_article_frames = True  # Naderi and Hirst set True

    csv_output = pathlib.Path("converted").joinpath("Media-{}-framing_{}_{}{}{}{}{}.csv".format(
        "".join(map(lambda df: df.stem, data_files)),
        "gold" if use_only_gold_labels else "dirty",
        "pure" if filter_not_media_frames else "all",
        "+" if ignore_article_frames else "_",
        "sent" if extract_sentences else "",
        "noprem" if no_premise else "",
        "exact" if exact_frames else "")
    )

    decoder = json.JSONDecoder()
    decoder.parse_float = False
    decoder.parse_int = True
    decoder.parse_constant = False
    decoder.strict = False

    frame_encodings = decoder.decode(codes.read_text(encoding="utf-8"))
    print("Got {} media frames encodings".format(len(frame_encodings)))

    article_dict = dict()
    for data_file in data_files:
        decoded = decoder.decode(data_file.read_text(encoding="utf-8"))
        print(type(decoded))
        article_dict.update(decoded)

    print("Collected {} articles".format(len(article_dict)))

    dict_collector = dict()
    for article_id, content in article_dict.items():
        text = content["text"]
        for name, annotations in content["annotations"]["framing"].items():
            for annotation in annotations:
                if ignore_article_frames and str(annotation["code"]).endswith(".2"):
                    # it's a frame annotation over the whole article
                    continue
                spanned_text = text[annotation["start"]-1:annotation["end"]+1]
                #ignore the spanned text. Sometimes, it's one character to much or missing - the annotations are not quite correct :\
                if (66 <= ord(spanned_text[0]) <= 90 or 98 <= ord(spanned_text[0]) <= 122) and ord(spanned_text[0]) != 73:
                    spanned_text = spanned_text[1:]
                if 65 <= ord(spanned_text[-1]) <= 90 or 98 <= ord(spanned_text[-1]) <= 122:
                    spanned_text = spanned_text[:-1]
                spanned_text = spanned_text.strip(" \n-',").lstrip(".)]").rstrip("[(")
                annotation["start"] = text.index(spanned_text, annotation["start"]-1)
                annotation["end"] = annotation["start"]+len(spanned_text)

                if spanned_text == "PRIMARY":
                    spanned_text = text
                entry_id = "{}_{}_{}".format(article_id, annotation["start"], annotation["end"])
                annotation["code"] = str(annotation["code"])
                frame = frame_encodings.get(annotation["code"],
                                            None) if filter_not_media_frames else frame_encodings.get(
                    annotation["code"], "Irrelevant")
                if frame is not None:
                    if not exact_frames:
                        if frame.endswith(" headline"):
                            frame = frame[:-9]
                        elif frame.endswith(" primary"):
                            frame = frame[:-8]
                    frame_id = [annotation["code"] if exact_frames else str(annotation["code"]).split(".", 2)[0]]
                    if entry_id in dict_collector.keys():
                        dict_collector[entry_id]["frame_id"] += frame_id
                        dict_collector[entry_id]["frame"] += [frame]
                    else:
                        index_of_text_body = text.index(content["title"]) + len(content["title"]) + 2
                        paragraph_split = spanned_text.split("\n\n")
                        if len(paragraph_split) >= 2:
                            premise = "" if no_premise else paragraph_split[-2]
                            conclusion = paragraph_split[-1]
                        else:
                            all_sentences, candidate_sentences_index = \
                                fetch_sentences_of_span(whole_text=text, start=annotation["start"],
                                                        end=annotation["end"])
                            if len(candidate_sentences_index) >= 2:
                                if no_premise:
                                    premise = ""
                                    conclusion = ""
                                    for i in candidate_sentences_index:
                                        conclusion += all_sentences[i] + " "
                                elif extract_sentences:
                                    premise = ""
                                    for i in candidate_sentences_index[:-1]:
                                        premise += all_sentences[i] + " "
                                    conclusion = all_sentences[candidate_sentences_index[-1]]
                                else:
                                    conclusion = all_sentences[candidate_sentences_index[-1]]
                                    len_conclusion = annotation["end"] - text.rindex(conclusion, annotation["start"])
                                    if len_conclusion <= 3:
                                        conclusion = all_sentences[candidate_sentences_index[-1]]
                                        if len(candidate_sentences_index) >= 3:
                                            premise = ""
                                            for i in candidate_sentences_index[:-2]:
                                                premise += all_sentences[i] + " "
                                        else:
                                            premise = ""
                                    else:
                                        conclusion = conclusion[:len_conclusion].strip()
                                        premise = ""
                                        for i in candidate_sentences_index[:-1]:
                                            premise += all_sentences[i] + " "
                            else:
                                if extract_sentences:
                                    if len(candidate_sentences_index) >= 1:
                                        candidate_sentence_index = candidate_sentences_index[0]
                                        premise = all_sentences[candidate_sentence_index - 1] \
                                            if candidate_sentence_index > 1 else ""
                                        conclusion = all_sentences[candidate_sentence_index]
                                    else:
                                        premise = ""
                                        conclusion = spanned_text
                                else:
                                    try:
                                        premise = "" if no_premise else\
                                            text[text.rindex(".", 0, annotation["start"]) + 1:annotation["start"]]
                                    except ValueError:
                                        if "\n" in text[:annotation["start"]]:
                                            premise = text[
                                                      text.rindex("\n", 0, annotation["start"]) + 1:annotation["start"]
                                                      ]
                                        else:
                                            premise = text[:annotation["start"]]
                                    conclusion = spanned_text
                        dict_collector[entry_id] = {
                            "argument_id": "{}_{}-{}".format(article_id, annotation["start"], annotation["end"]),
                            "frame_id": frame_id,
                            "frame": [frame],
                            "topic_id": content["csi"],
                            "topic": "{}: {}".format(article_id[:article_id.index("1.0")], content["title"]),
                            "premise": premise.replace("\t", " ").replace("\n", " ").strip(),
                            "conclusion": conclusion.replace("\t", " ").replace("\n", " ").strip()
                        }

    print("Went through data and found {} samples.".format(len(dict_collector)))

    ret = []
    for sample in dict_collector.values():
        ret_line = [sample["argument_id"]]
        frame_dict = dict()
        for f in zip(sample["frame_id"], sample["frame"]):
            frame_dict[f] = frame_dict.get(f, 0) + 1

        if max(frame_dict.values()) <= 1 and use_only_gold_labels:
            # only one annotator for this text piece or no majority - unsure!
            continue
        elif sum(frame_dict.values()) > 1 >= max(frame_dict.values()):
            # To create our dataset, we first gathered the annotations that at least two annotators agreed upon;
            # however, that process resulted in a small corpus
            # because a majority of the articles on smoking were
            # annotated only once. Therefore, we kept the cases
            # that were annotated only once, and for the more
            # controversial cases, where multiple frame dimensions were assigned, we kept only the annotations
            # that were agreed upon by at least two annotators.
            continue

        if len(frame_dict) == 0 and filter_not_media_frames:
            continue
        elif len(frame_dict) == 0:
            ret_line.append("16.2" if exact_frames else "16")
            ret_line.append("Irrelevant")
        else:
            flat_frame_dict = [(k, v) for k, v in frame_dict.items()]
            flat_frame_dict.sort(key=lambda k: k[1], reverse=True)
            ret_line.extend(flat_frame_dict[0][0])

        ret_line.append(sample["topic_id"])
        ret_line.append(sample["topic"])
        # The sentences were further lower-cased and all
        # numeric tokens were converted to <NUM>.
        ret_line.append(regex.sub("\d+", "<num>", sample["premise"]))
        ret_line.append(regex.sub("\d+", "<num>", sample["conclusion"]))

        ret.append(ret_line)

    print("Will write {} samples to {}".format(len(ret), csv_output.absolute()))


    class ExcelPipe(csv.Dialect):
        """Describe the usual properties of Excel-generated CSV files."""
        delimiter = '|'
        quotechar = '"'
        doublequote = True
        skipinitialspace = False
        lineterminator = '\n'
        quoting = csv.QUOTE_MINIMAL


    csv_output.parent.mkdir(parents=True, exist_ok=True)
    writer = csv.writer(csv_output.open(mode="w", encoding="utf-8", buffering=True), ExcelPipe)
    writer.writerow(["argument_id", "frame_id", "frame", "topic_id", "topic", "premise", "conclusion"])
    writer.writerows(ret)

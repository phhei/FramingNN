"""
The algorithm behind this file is outdated! It's just usable because the base algorithms bases on the structure which
is introduced here.
"""

import csv
import datetime
import os
import pathlib
import sys
import tempfile
from functools import reduce

import loguru
import nltk
from nltk.stem import WordNetLemmatizer

import Frames

# #####################HELPER FUNCTIONS###################
# see Frames


# #####################H###################################


logger = loguru.logger
logger.remove()
logger.add(sys.stdout, format="{time} {level} {message}", level="WARNING")

# PARAMETERS
# CSV_FILE = pathlib.Path("Webis-argument-framing-mediaFramesTopics.csv")
CSV_FILE = pathlib.Path("MediaFramesCorpus").joinpath("converted").\
    joinpath("Media-immigration-framing_dirty_pure+sentexact.csv")
logger.info("Will read and process {}", os.path.abspath(CSV_FILE))
genericFrames = Frames.most_frequent_media_frames_set

# NLP
nltk.download("punkt")
nltk.download("wordnet")
lemming = WordNetLemmatizer()

if __name__ == "__main__":
    if not os.path.exists(CSV_FILE):
        logger.error(" {} does not exists. Cancel...", os.path.basename(CSV_FILE))
        exit(1)

    logger.add(os.path.join(".", "logs", "{}_{}{}".format(os.path.basename(CSV_FILE),
                                                          str(int(datetime.datetime.timestamp(datetime.datetime.now()))),
                                                          ".log")), level="DEBUG")
    logger.debug("Writing in a logging file now at {}", datetime.datetime.isoformat(datetime.datetime.now()))

    data = []
    mapUserLabelToGeneric = {}

    with open(CSV_FILE, newline="\n", encoding="utf-8") as csv_file:
        csvReader = csv.reader(csv_file, delimiter="|", quotechar='"')
        scheme = None
        for row in csvReader:
            if csvReader.line_num == 1:
                scheme = row
            else:
                if scheme is None:
                    logger.error("No scheme!")
                    raise AttributeError
                else:
                    logger.debug("Fetch {}", ', '.join(row))
                    argumentMapping = dict()
                    for i in range(0, len(row)):
                        try:
                            argumentMapping[scheme[i]] = row[i]
                        except IndexError as err:
                            logger.warning(err)
                    logger.debug("Collected {} elements.", len(argumentMapping))

                    userLabel = argumentMapping.get("frame", "__UNKNOWN__")
                    logger.debug("The user label is: {}. Determine the generic label now...", userLabel)
                    if userLabel in mapUserLabelToGeneric.keys():
                        currentGenericFuzzyLabel = mapUserLabelToGeneric[userLabel][0]
                        argumentMapping["genericFrame"] = currentGenericFuzzyLabel[0]
                        currentCount = mapUserLabelToGeneric[userLabel][1]
                        logger.debug("{} was already sorted in (generic: {}, {}x)", userLabel, currentGenericFuzzyLabel[0],
                                     currentCount)
                        argumentMapping["fuzzyFrame"] = " ".join(
                            ["({}:{})".format(k, str(v)) for (k, v) in currentGenericFuzzyLabel[1].items()])
                        mapUserLabelToGeneric[userLabel] = (currentGenericFuzzyLabel, currentCount + 1)
                    else:
                        favouriteFrames = genericFrames.compute_fuzzy_frame(userLabel)
                        genericFrame = (None, -1)
                        for favouriteFrameKey, favouriteFrameValue in favouriteFrames.items():
                            if genericFrame[1] < favouriteFrameValue:
                                genericFrame = (favouriteFrameKey, favouriteFrameValue)
                        argumentMapping["genericFrame"] = genericFrame[0]
                        logger.info("Label \"{}\" was sorted in (generic: {}, 1x)", userLabel, genericFrame[0])
                        argumentMapping["fuzzyFrame"] = \
                            " ".join(["({}:{})".format(k, str(v)) for (k, v) in favouriteFrames.items()])
                        logger.info("Label \"{}\" was sorted in (fuzzy: {})", userLabel, favouriteFrames)
                        mapUserLabelToGeneric[userLabel] = ((genericFrame[0], favouriteFrames), 1)
                    data.append(argumentMapping)
            logger.debug("Finished processing line {}.", str(csvReader.line_num))

    # Stats
    CSV_FILE_Out = pathlib.Path(os.path.dirname(CSV_FILE),
                                "{}_{}out.csv".format("_".join(os.path.basename(CSV_FILE).split(".")[0:-1]),
                                                      "mostfrequent" if genericFrames == Frames.most_frequent_media_frames_set else ""))
    CSV_FILE_Out_info = pathlib.Path("{}.info".format(CSV_FILE_Out.absolute()))

    logger.add(sink=CSV_FILE_Out_info.open(mode="a", encoding="utf-8"), level="INFO", colorize=False, backtrace=True,
               diagnose=False, catch=True)

    logger.info("Collected {} tuples now with {} elements in total", len(data), reduce(
        lambda x, y: (len(x) if (isinstance(x, dict) or isinstance(x, list)) else x) + (
            len(y) if (isinstance(y, dict) or isinstance(y, list)) else y), data))
    logger.info("{} out of {} user frames are without a generic label!",
                len([i for i in mapUserLabelToGeneric.values() if i[0][0] == "__UNKNOWN__"]), len(mapUserLabelToGeneric))

    logger.warning("We provide the generic frame distribution now:")
    for specificFrame in genericFrames.frame_names:
        try:
            selectedUserLabels, selectedUserLabelCounts = zip(*[(user, general[1])
                                                                for (user, general) in mapUserLabelToGeneric.items()
                                                                if general[0][0] == specificFrame])
            logger.info("{} : {} user labels which contain {} arguments: {}", specificFrame, len(selectedUserLabels),
                        reduce(lambda x, y: x + y, selectedUserLabelCounts), "; ".join(selectedUserLabels))
        except ValueError as e:
            logger.warning("{} didn't occur as mapped generic frame!", specificFrame)
            logger.debug(e)
    try:
        nonGenericUserLabels, nonGenericUserLabelCounts = zip(
            *[(user, general[1]) for (user, general) in mapUserLabelToGeneric.items() if general[0][0] == "__UNKNOWN__"])
        logger.info("UNKNOWN (no fitting generic label match) : {} user labels which contain {} arguments: {}",
                    len(nonGenericUserLabels), reduce(lambda x, y: x + y, nonGenericUserLabelCounts),
                    "; ".join(nonGenericUserLabels))
    except ValueError as e:
        logger.info("No user label was mapped to __UNKNOWN__-label =)")
        logger.debug(e)

    # Writing back
    if CSV_FILE_Out.exists():
        CSVFILEReplacement = pathlib.Path(tempfile.gettempdir(),
                                          "{}{}.csv".format(os.path.basename(CSV_FILE),
                                                            str(int(datetime.datetime.timestamp(datetime.datetime.now())))))
        logger.error("{} exists already - we'll move the file to {}", CSV_FILE_Out, CSVFILEReplacement)
        pathlib.Path(CSV_FILE_Out).rename(CSVFILEReplacement)
    else:
        logger.info("Write {}", CSV_FILE_Out)

    with open(CSV_FILE_Out, newline="\n", encoding="utf-8", mode="w") as csv_file:
        writer = csv.writer(csv_file, delimiter="|", quotechar='"')
        schemeUpdated = data[0].keys()
        writer.writerow(schemeUpdated)
        writer.writerows([line.values() for line in data if len(line.keys()) == len(schemeUpdated)])

    logger.info("Finished {} ...", CSV_FILE_Out)

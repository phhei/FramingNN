from functools import reduce
from typing import List, Tuple

import loguru
import nltk
import numpy
from nltk.stem import WordNetLemmatizer

# NLP

nltk.download("punkt")
nltk.download("wordnet")
lemma_converter = WordNetLemmatizer()


class GenericFrame:
    name_of_frame_set = "UNKNOWN"
    logFrame = loguru.logger

    def __init__(self, name):
        self.name_of_frame_set = str(name)
        self.frame_names = list()
        self.dict_map_frame_to_user_keywords = dict()
        self.logFrame.info("A new frame set was added: {}", self.name_of_frame_set)

    def add_blank_generic_frame(self, name):
        self.frame_names.append(name)

    def add_generic_frame(self, name, user_label_words):
        name = str(name).upper()

        if not isinstance(user_label_words, set):
            self.logFrame.warning("The keywordList to the frame {} was not a list but {}", name, type(user_label_words))
            user_label_words = {user_label_words}

        if name in self.dict_map_frame_to_user_keywords.keys():
            self.logFrame.warning("The frame {} already exists in the frame set {}. Redirect to <extend_generic_frame>",
                                  name, self.name_of_frame_set)
            self.extend_generic_frame(name, user_label_words)
            return
        self.frame_names.append(name)
        self.dict_map_frame_to_user_keywords[name] = user_label_words

        self.logFrame.debug("Added a new frame to {}: {} with the keyword list {}", self.name_of_frame_set, name,
                            user_label_words)

    def extend_generic_frame(self, name, user_label_words):
        name = str(name).upper()

        if not isinstance(user_label_words, set):
            self.logFrame.warning("The keywordList to the frame {} was not a list but {}", name, type(user_label_words))
            user_label_words = {user_label_words}

        try:
            self.logFrame.trace("Extension query in frame set {}", self.name_of_frame_set)
            assert isinstance(self.dict_map_frame_to_user_keywords[name], set)
            if isinstance(user_label_words, str):
                self.dict_map_frame_to_user_keywords[name].put(user_label_words)
            else:
                self.dict_map_frame_to_user_keywords[name].extend(user_label_words)
            self.logFrame.info("Extended successfully the frame {}. Has {} user-label-keywords now.", name,
                               len(self.dict_map_frame_to_user_keywords[name]))
        except IndexError:
            self.logFrame.warning(
                "Can not find the frame {} in frame set {} <extend_generic_frame({}, {})>. Skip the query", name,
                self.name_of_frame_set, name, user_label_words)

    def map_user_label_to_generic(self, user_label) -> List[Tuple]:
        self.logFrame.debug("The user label is \"{}\". Determine the generic label now...", user_label)

        if str(user_label).upper() in self.frame_names:
            return [(str(user_label).upper(), 1)]
        else:
            outputFrames = self.compute_fuzzy_frame(user_label)
            favourite_frames = sorted(outputFrames.items(), key=lambda frame: frame[1], reverse=True)
            self.logFrame.debug("Label \"{}\" was sorted in (generic: {}, 1x)", user_label, favourite_frames)
            return favourite_frames

    def compute_fuzzy_frame(self, label) -> dict:
        output_frames = dict()
        words = [word for word in nltk.word_tokenize(label.lower()) if len(word) >= 3]
        lemmas = [lemma_converter.lemmatize(word) for word in words]
        self.logFrame.info("Following lemmas contains the user_label: {} ({})", ",".join(lemmas), len(lemmas))
        for frame, keyWordList in dict(self.dict_map_frame_to_user_keywords).items():
            number = len([lemma for lemma in lemmas if lemma in keyWordList])
            if number >= 1:
                output_frames[frame] = number
        if len(output_frames) == 0:
            self.logFrame.warning("Founds no generic frame in {}", label)
            output_frames["__UNKNOWN__"] = 1
            return output_frames

        # normalize
        total_count = reduce(lambda d1, d2: d1 + d2, output_frames.values())
        return {k: float(v) / total_count for (k, v) in output_frames.items()}

    def decode_frame_label(self, frame_name_distribution, ignore_unknown=True):
        ret = numpy.zeros(self.get_prediction_vector_length(ignore_unknown=ignore_unknown), dtype="float32")

        if isinstance(frame_name_distribution, str):
            frame_name_distribution = {frame_name_distribution: 1}

        for (frame_name, probability) in frame_name_distribution.items():
            try:
                position = self.frame_names.index(frame_name)
                ret[position] = probability
                self.logFrame.trace("Found {} at position {}: Hence, the vector is: {}", frame_name_distribution,
                                    position,
                                    ret)
            except ValueError as e:
                self.logFrame.warning(
                    "{}: The frame {} was not found in the frame set {} - no one-hot-encode!",
                    e, frame_name_distribution, self.name_of_frame_set)

        if all(numpy.equal(ret,
                           numpy.zeros(self.get_prediction_vector_length(ignore_unknown=ignore_unknown),
                                       dtype="float32"))):
            self.logFrame.debug("We still have a zero-vector after encoding the distribution {}."
                                "Hence, let's use the last hot for encode \"unknown\"", frame_name_distribution)
            if not ignore_unknown:
                ret[-1] = 1

        self.logFrame.debug("Computed a frame-prediction-ground-truth-vector: {}", ret)
        return numpy.reshape(ret, (1, self.get_prediction_vector_length(ignore_unknown=ignore_unknown)))

    def get_prediction_vector_length(self, ignore_unknown=True):
        return len(self.frame_names) if ignore_unknown else len(self.frame_names) + 1

    def get_all_frame_names(self, tokenized=False, lower=False) -> List:
        def lower_f(f_input):
            if lower:
                if isinstance(f_input, List):
                    return [lf.lower() for lf in f_input]
                else:
                    return f_input.lower()
            return f_input

        return [lower_f(nltk.word_tokenize(text=f) if tokenized else f) for f in self.frame_names]


# CORE
media_frames_set = GenericFrame("MediaFramesSet")
media_frames_set.add_generic_frame("ECONOMIC", {"economic", "economics", "cost", "financial", "finance"})
media_frames_set.add_generic_frame("CAPACITY AND RESOURCES", {"capacity", "resource", "limit"})
media_frames_set.add_generic_frame("MORALITY", {"religion", "ethic", "moral", "morality"})
media_frames_set.add_generic_frame("FAIRNESS AND EQUALITY", {"right", "fair", "equality"})
media_frames_set.add_generic_frame("LEGALITY, CONSTITUTIONALITY AND JURISPRUDENCE", {"right", "legal", "jurisprudence"})
media_frames_set.add_generic_frame("POLICY PRESCRIPTION AND EVALUATION", {"strategy", "policy", "evaluation"})
media_frames_set.add_generic_frame("CRIME AND PUNISHMENT", {"law", "crime", "punish"})
media_frames_set.add_generic_frame("SECURITY AND DEFENSE", {"secure", "security", "defense", "welfare"})
media_frames_set.add_generic_frame("HEALTH AND SAFETY", {"health", "care", "safety"})
media_frames_set.add_generic_frame("QUALITY OF LIFE", {"life", "individual", "wealth"})
media_frames_set.add_generic_frame("CULTURAL IDENTITY", {"culture", "identity", "tradition"})
media_frames_set.add_generic_frame("PUBLIC OPINION", {"public", "general", "opinion"})
media_frames_set.add_generic_frame("POLITICAL", {"political", "government", "election"})
media_frames_set.add_generic_frame("EXTERNAL REGULATION AND REPUTATION", {"regulation", "reputation", "external"})

one_vs_other_frame_set = GenericFrame("one vs. other")
one_vs_other_frame_set.add_generic_frame("ECONOMIC", {"economic", "economics", "cost", "financial", "finance"})
one_vs_other_frame_set.add_generic_frame("CAPACITY AND RESOURCES", {"capacity", "resource", "limit"})
# one_vs_other_frame_set.add_generic_frame("MORALITY", {"religion", "ethic", "moral", "morality"})
# one_vs_other_frame_set.add_generic_frame("FAIRNESS AND EQUALITY", {"right", "fair", "equality"})
# one_vs_other_frame_set.add_generic_frame("LEGALITY, CONSTITUTIONALITY AND JURISPRUDENCE", {"right", "legal", "jurisprudence"})
# one_vs_other_frame_set.add_generic_frame("POLICY PRESCRIPTION AND EVALUATION", {"strategy", "policy", "evaluation"})
# one_vs_other_frame_set.add_generic_frame("CRIME AND PUNISHMENT", {"law", "crime", "punish"})
# one_vs_other_frame_set.add_generic_frame("SECURITY AND DEFENSE", {"secure", "security", "defense", "welfare"})
# one_vs_other_frame_set.add_generic_frame("HEALTH AND SAFETY", {"health", "care", "safety"})
# one_vs_other_frame_set.add_generic_frame("QUALITY OF LIFE", {"life", "individual", "wealth"})
# one_vs_other_frame_set.add_generic_frame("CULTURAL IDENTITY", {"culture", "identity", "tradition"})
# one_vs_other_frame_set.add_generic_frame("PUBLIC OPINION", {"public", "general", "opinion"})
# one_vs_other_frame_set.add_generic_frame("POLITICAL", {"political", "government", "election"})
# one_vs_other_frame_set.add_generic_frame("EXTERNAL REGULATION AND REPUTATION", {"regulation", "reputation", "external"})

most_frequent_media_frames_set = GenericFrame("MediaFramesSet (most frequent)")
most_frequent_media_frames_set.add_generic_frame("ECONOMIC", {"economic", "economics", "cost", "financial", "finance"})
most_frequent_media_frames_set.add_generic_frame("LEGALITY, CONSTITUTIONALITY AND JURISPRUDENCE",
                                                 {"right", "legal", "jurisprudence"})
most_frequent_media_frames_set.add_generic_frame("POLICY PRESCRIPTION AND EVALUATION",
                                                 {"strategy", "policy", "evaluation"})
most_frequent_media_frames_set.add_generic_frame("CRIME AND PUNISHMENT", {"law", "crime", "punish"})
most_frequent_media_frames_set.add_generic_frame("POLITICAL", {"political", "government", "election"})

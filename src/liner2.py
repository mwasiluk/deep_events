#! /usr/bin/python
# -*- coding: utf-8 -*-
from jpype import *

def start_jvm(jars, libs):
    if isJVMStarted():
        return
    startJVM(getDefaultJVMPath(), "-Djava.library.path="+':'.join(libs), "-Djava.class.path="+':'.join(jars))

class Liner2(object):
    def __init__(self, liner_ini, tagset='nkjp'):

        ChunkerFactory = JClass("g419.liner2.api.chunker.factory.ChunkerFactory")
        self.options = JClass("g419.liner2.api.LinerOptions")()
        self.options.parseModelIni(liner_ini)
        #self.chunkerManager = ChunkerFactory.loadChunkers(self.options)
        if not self.options.features.isEmpty():
            self.featureGen = JClass("g419.liner2.api.features.TokenFeatureGenerator")(self.options.features)
        else:
            self.featureGen = None


    def get_reader(self, input_file, input_format="cclrel"):
        return JClass("g419.corpus.io.reader.ReaderFactory").get().getStreamReader(input_file, input_format)

    def get_batch_reader(self, input_data, root, input_format="cclrel"):
        return JClass("g419.corpus.io.reader.BatchReader")(JClass("org.apache.commons.io.IOUtils").toInputStream(input_data), root, input_format)

    def get_token_feature_generator(self):
        return JClass("g419.liner2.api.features.TokenFeatureGenerator")(self.options.features)

    def add_chunker(self, name, description):
        self.chunkerManager.addChunker(name, description)

    def get_chunker(self, name=''):
        if not name:
            name = self.options.getOptionUse()
        return self.chunkerManager.getChunkerByName(name)

    def generate_features(self, ps):
        if self.featureGen is not None:
            self.featureGen.generateFeatures(ps)

    def prepare_paragraph_set(self):
        attribute_index = JClass("liner2.structure.TokenAttributeIndex")()
        attribute_index.addAttribute("orth");
        attribute_index.addAttribute("base");
        attribute_index.addAttribute("ctag"); 
        ps = JClass("liner2.structure.ParagraphSet")()
        ps.setAttributeIndex(attribute_index)
        return ps

    # def corpus_sent_to_liner(self, corpus2_sentence):
    #     sentence = JClass("liner2.structure.Sentence")()
    #     sentence.setId(corpus2_sentence.id())
    #     for token in corpus2_sentence.tokens():
    #         sentence.addToken(self.corpus_token_to_liner(token))
    #     asent = corpus2.AnnotatedSentence.wrap_sentence(corpus2_sentence)
    #     for chan_name in asent.all_channels():
    #         for ann in asent.get_channel(chan_name).make_annotation_vector():
    #             indices = [i for i in ann.indices]
    #             annotation = JClass("liner2.structure.Annotation")(indices[0], chan_name.decode('utf-8'), ann.seg_number, sentence)
    #             for i in indices[1:]:
    #                 annotation.addToken(i)
    #             annotation.setHead(ann.head_index)
    #             sentence.addChunk(annotation)
    #     return sentence

    # def corpus_token_to_liner(self, corpus2_token):
    #     token = JClass("liner2.structure.Token")()
    #     token.setAttributeValue(0, corpus2_token.orth_utf8().decode('utf-8'))
    #     has_preffered_lexeme = False
    #     for lex in corpus2_token.lexemes():
    #         ctag = self.tagset.tag_to_string(lex.tag())
    #         if not has_preffered_lexeme and lex.is_disamb():
    #             token.setAttributeValue(1, lex.lemma_utf8().decode('utf-8'))
    #             token.setAttributeValue(2, ctag)
    #             has_preffered_lexeme = True
    #         new_tag = JClass("liner2.structure.Tag")(lex.lemma_utf8().decode('utf-8'), ctag, lex.is_disamb())
    #         token.addTag(new_tag)
    #     return token

    # def liner_annotations_to_corpus_sentence(self, liner_sentence, corpus_sentence):
    #     annotated_sentence = corpus2.AnnotatedSentence.wrap_sentence(corpus_sentence)
    #     for chan in annotated_sentence.all_channels():
    #             annotated_sentence.remove_channel(chan)
    #     for ann in liner_sentence.getChunks():
    #         chan_name = str(ann.getType().encode('utf-8'))
    #         if not annotated_sentence.has_channel(chan_name):
    #             annotated_sentence.create_channel(chan_name)
    #         chan = annotated_sentence.get_channel(chan_name)
    #         new_ann_idx = chan.get_new_segment_index()
    #         for tok_idx in ann.getTokens():
    #             tok_idx = int(tok_idx.toString())
    #             chan.set_segment_at(tok_idx, new_ann_idx)
    #     for chan_name in annotated_sentence.all_channels():
    #         chan = annotated_sentence.get_channel(chan_name)
    #         chan.renumber_segments()

    def get_document_writer(self, output_file, output_format):
        JClass("g419.corpus.io.writer.WriterFactory").get().getStreamWriter(output_file, output_format)

class LinerWordnet(object):
    def __init__(self, path, liner_jar, liner_lib):
        start_jvm(liner_jar, liner_lib)
        self.database = JClass("g419.liner2.api.features.tokens.WordnetLoader")(path)

    def get_hypernym_feature(self, name, distance):
        return JClass("g419.liner2.api.features.tokens.HypernymFeature")(name, self.database, distance)

    def get_synonym_feature(self):
        return JClass("g419.liner2.api.features.tokens.SynonymFeature")("synonym", self.database)


def create_annotation(begin, end, type, sentence):
        return JClass("g419.corpus.structure.Annotation")(begin, end, type, sentence)


def create_relation(annotation_from, annotation_to, type):
    return JClass("g419.corpus.structure.Relation")(annotation_from, annotation_to, type)


def get_writer(output_file, output_format):
    return JClass("g419.corpus.io.writer.WriterFactory").get().getStreamWriter(output_file, output_format)
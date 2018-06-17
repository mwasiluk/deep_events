class Relation(object):

    def __init__(self, type, ann_from, ann_to):
        self.type = type
        self.ann_from = ann_from
        self.ann_to = ann_to

    def getType(self):
        return self.type

    def getAnnotationFrom(self):
        return self.ann_from

    def getAnnotationTo(self):
        return self.ann_to
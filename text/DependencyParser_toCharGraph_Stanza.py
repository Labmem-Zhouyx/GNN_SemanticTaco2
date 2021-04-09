# coding:utf-8
import stanza

nlp = stanza.Pipeline('en')


def DependencyParser_stanza_word(sen):

    doc = nlp(sen)
    text = []
    head = []
    deprel = []
    upos = []
    for sent in doc.sentences:
        for word in sent.words:
            text.append(word.text)
            head.append(word.head)
            deprel.append(word.deprel)
            upos.append(word.upos)

    List1 = [i for i in range(len(head)) if head[i] != 0]
    List2 = [head[i] - 1 for i in range(len(head)) if head[i] != 0]
    deprel = [deprel[i] for i in range(len(head)) if head[i] != 0]

    # print("CHECK text:", sen)
    # print("CHECK List1:", List1)
    # print("CHECK List2:", List2)
    # print("CHECK deprel:", deprel)
    return List1, List2, deprel

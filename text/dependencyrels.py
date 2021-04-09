deprel_labels = ['conj', 'nmod:npmod', 'punct', 'advcl', 'appos', 'obl:tmod', \
                 'obl:npmod', 'nsubj', 'obj', 'parataxis', 'obl', 'expl', 'case', \
                 'ccomp', 'fixed', 'goeswith', 'dep', 'flat', 'vocative', 'xcomp', \
                 'cop', 'det', 'discourse', 'mark', 'aux:pass', 'nummod', 'list', \
                 'acl:relcl', 'csubj', 'cc:preconj', 'nmod:tmod', 'nsubj:pass', 'compound:prt', \
                 'acl', 'iobj', 'orphan', 'aux', 'nmod:poss', 'dislocated', 'det:predet', \
                 'compound', 'csubj:pass', 'amod', 'nmod', 'advmod', 'cc']

deprel_labels_to_id = {s: i for i, s in enumerate(deprel_labels)}
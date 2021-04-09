from text import text_to_sequence
from text.dependencyrels import deprel_labels_to_id
from text.DependencyParser_toCharGraph_Stanza import DependencyParser_stanza_word
from utils import load_filepaths_and_text
from tqdm import tqdm
melpaths_and_text = './training_data/mel-bc2013_character_data_val.txt'
melpaths_and_text = load_filepaths_and_text(melpaths_and_text)

output = ''
deprel_set = []
for melpath_and_text in tqdm(melpaths_and_text):
    melpath, char_text= melpath_and_text[0], melpath_and_text[1]
    _, char_text_norm = text_to_sequence(char_text, ['basic_cleaners'])
    edge_node1, edge_node2, edge_label = DependencyParser_stanza_word(char_text_norm)
    List1_str = ' '.join([str(i) for i in edge_node1])
    List2_str = ' '.join([str(i) for i in edge_node2])
    deprel = ' '.join(edge_label)
    deprel_set.extend(edge_label)
    output += melpath + '|' + char_text_norm + '|' + List1_str + '|' + List2_str + '|' + deprel + '\n'

with open('./training_data/mel-bc2013_DependencyParsing_Stanza_WordGraph_val.txt', 'w', encoding='utf8') as f:
    f.write(output)

print(set(deprel_set))
print(len(set(deprel_set)))
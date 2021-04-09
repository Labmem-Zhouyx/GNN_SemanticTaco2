with open('/ceph/home/zhouyx20/data/LJSpeech_training_data/train.txt', 'r', encoding='utf-8') as f:
    content = f.readlines()
output = ''
for line in content:
    text = line.strip().split('|')
    output += '/ceph/home/zhouyx20/data/LJSpeech_training_data/mels/' + text[1] + '|' + text[5] + '\n'

with open('mel-ljspeech_character_data_train.txt', 'w', encoding='utf-8') as f:
    f.write(output)
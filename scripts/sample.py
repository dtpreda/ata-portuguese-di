import pandas as pd

### FULL-LENGTH ###

f = open("your\path\here\TED2020.pt-pt_br.pt_br", "r", encoding="utf-8")

prefix = "tedtalks-1k-full-length"

train_lst, dev_lst, test_lst = [], [], []
line_group = []
talk_counter = 0
line = "-"
while talk_counter < 1200:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        train_lst.append([" ".join(line_group), 'BR'])
        line_group = []
    else:
        line_group.append(line)

while talk_counter < 1600:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        dev_lst.append([" ".join(line_group), 'BR'])
        line_group = []
    else:
        line_group.append(line)

while talk_counter < 2000:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        test_lst.append([" ".join(line_group), 'BR'])
        line_group = []
    else:
        line_group.append(line)

f.close()

f = open("your\path\here\TED2020.pt-pt_br.pt", "r", encoding="utf-8")

line_group = []
talk_counter = 0
line = "-"
while talk_counter < 1200:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        train_lst.append([" ".join(line_group), 'PT'])
        line_group = []
    else:
        line_group.append(line)

while talk_counter < 1600:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        dev_lst.append([" ".join(line_group), 'PT'])
        line_group = []
    else:
        line_group.append(line)

while talk_counter < 2000:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        test_lst.append([" ".join(line_group), 'PT'])
        line_group = []
    else:
        line_group.append(line)

f.close()

df = pd.DataFrame(train_lst, columns=['text', 'label'])
df.to_csv(f'{prefix}-train-raw.csv', index=False, encoding='utf-8', sep=';')

df = pd.DataFrame(dev_lst, columns=['text', 'label'])
df.to_csv(f'{prefix}-dev-raw.csv', index=False, encoding='utf-8', sep=';')

df = pd.DataFrame(test_lst, columns=['text', 'label'])
df.to_csv(f'{prefix}-test-raw.csv', index=False, encoding='utf-8', sep=';')

### 4 SENTENCES ###

f = open("your\path\here\TED2020.pt-pt_br.pt_br", "r", encoding="utf-8")

prefix = "tedtalks-1k-4sent"

train_lst, dev_lst, test_lst = [], [], []
line_group = []
talk_counter = 0
line = "-"

while talk_counter < 1200:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        line_group = []
    elif len(line_group) == 4:
        train_lst.append([" ".join(line_group), 'BR'])
        line_group = []
    else:
        line_group.append(line)

while talk_counter < 1600:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        line_group = []
    elif len(line_group) == 4:
        dev_lst.append([" ".join(line_group), 'BR'])
        line_group = []
    else:
        line_group.append(line)

while talk_counter < 2000:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        line_group = []
    elif len(line_group) == 4:
        test_lst.append([" ".join(line_group), 'BR'])
        line_group = []
    else:
        line_group.append(line)

f.close()

f = open("your\path\here\TED2020.pt-pt_br.pt", "r", encoding="utf-8")

line_group = []
talk_counter = 0
line = "-"
while talk_counter < 1200:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        line_group = []
    elif len(line_group) == 4:
        train_lst.append([" ".join(line_group), 'PT'])
        line_group = []
    else:
        line_group.append(line)

while talk_counter < 1600:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        line_group = []
    elif len(line_group) == 4:
        dev_lst.append([" ".join(line_group), 'PT'])
        line_group = []
    else:
        line_group.append(line)

while talk_counter < 2000:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
        line_group = []
    elif len(line_group) == 4:
        test_lst.append([" ".join(line_group), 'PT'])
        line_group = []
    else:
        line_group.append(line)

f.close()

df = pd.DataFrame(train_lst, columns=['text', 'label'])
df.to_csv(f'{prefix}-train-raw.csv', index=False, encoding='utf-8', sep=';')

df = pd.DataFrame(dev_lst, columns=['text', 'label'])
df.to_csv(f'{prefix}-dev-raw.csv', index=False, encoding='utf-8', sep=';')

df = pd.DataFrame(test_lst, columns=['text', 'label'])
df.to_csv(f'{prefix}-test-raw.csv', index=False, encoding='utf-8', sep=';')

### SINGLE SENTENCE ###

f = open("your\path\here\TED2020.pt-pt_br.pt_br", "r", encoding="utf-8")

prefix = "tedtalks-1k-sent"

train_lst, dev_lst, test_lst = [], [], []
talk_counter = 0
line = "-"
while talk_counter < 1200:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
    else:
        train_lst.append([line, 'BR'])

while talk_counter < 1600:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
    else:
        dev_lst.append([line, 'BR'])

while talk_counter < 2000:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
    else:
        test_lst.append([line, 'BR'])

f.close()

f = open("your\path\here\TED2020.pt-pt_br.pt", "r", encoding="utf-8")

talk_counter = 0
line = "-"
while talk_counter < 1200:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
    else:
        train_lst.append([line, 'PT'])

while talk_counter < 1600:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
    else:
        dev_lst.append([line, 'PT'])

while talk_counter < 2000:
    line = f.readline().strip()
    if line == "":
        talk_counter += 1
    else:
        test_lst.append([line, 'PT'])

f.close()

df = pd.DataFrame(train_lst, columns=['text', 'label'])
df.to_csv(f'{prefix}-train-raw.csv', index=False, encoding='utf-8', sep=';')

df = pd.DataFrame(dev_lst, columns=['text', 'label'])
df.to_csv(f'{prefix}-dev-raw.csv', index=False, encoding='utf-8', sep=';')

df = pd.DataFrame(test_lst, columns=['text', 'label'])
df.to_csv(f'{prefix}-test-raw.csv', index=False, encoding='utf-8', sep=';')



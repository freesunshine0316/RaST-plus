
data = [x.strip() for x in open('coai.all.tsv', 'r')]
st, ed = 0, 2000
for i in range(10):
    data_dev = data[st:ed]
    with open('coai.dev_{}.tsv'.format(i+1), 'w') as f:
        for x in data_dev:
            f.write(x+'\n')
    data_train = data[:st] + data[ed:]
    with open('coai.train_{}.tsv'.format(i+1), 'w') as f:
        for x in data_train:
            f.write(x+'\n')
    st += 2000
    ed += 2000

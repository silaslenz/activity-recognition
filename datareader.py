import numpy as np
from activities import Activities

def read_data(filename, activity, fs, chunk_size):
    from numpy import genfromtxt
    my_data = genfromtxt(filename, delimiter=',')
    x,y,z = split_into_chunks([my_data[1:,1],my_data[1:,2],my_data[1:,3]],seconds_per_chunk=chunk_size, fs = fs)
    data = []
    labels = []
    for i in range(len(x)):
        data.append([x[i],y[i],z[i]])
        labels.append(activity)
    return data, labels

def chunk_splitter(l, n):
    """Yield successive n-sized chunks from l."""
    return_data = []
    for i in range(0, len(l), n):
        if i+n<len(l):
            return_data.append(l[i:i + n])
    return return_data

def split_into_chunks(data, fs, seconds_per_chunk = 2):
    return list(map(lambda x: chunk_splitter(list(x),fs*seconds_per_chunk),data))

def get_dataset(used_files, fs, chunk_size):
    dataset = []
    labels = []
    for file in used_files:
        filename, activity = file
        print(filename, activity)
        data, label = read_data(filename, activity, fs, chunk_size)
        dataset += data
        labels += label
    return np.array(dataset), np.array(labels)

def get_dominating_frequency(data):
    yf = np.fft.fft(data)
    T = 1/200
    N = len(data)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

    return xf[np.argmax(np.abs(yf[:N//2]))], 2.0/N * np.max(np.abs(yf[:N//2]))

def generate_statistics(dataset,labels):
    dominating_frequency = dict((key, []) for key in Activities)
    dominating_frequency_unlabeld = []
    for i, chunk in enumerate(dataset):
        chunk = chunk[0]
        dominating_frequency[labels[i]].append(get_dominating_frequency(chunk))
        dominating_frequency_unlabeld.append(get_dominating_frequency(chunk))
    return  dominating_frequency, np.array(dominating_frequency_unlabeld)
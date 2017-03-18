from numpy.random import randint, random_integers

chars = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
ch_begin = ['A', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
ch_end = ['A', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'Y']

max_distance = 2
    
def close(seq):
    
    distance = randint(low=1, high=max_distance+1)
    
    for i in range(distance):
        pos = randint(1, len(seq)-1)
        n = randint(3)
        if n == 0:
            new_seq = seq[:pos] + seq[pos+1:]
        elif n == 1:
            m = randint(len(chars))
            new_seq = seq[:pos] + chars[m] + seq[pos+1:]
        elif n == 2:
            m = randint(len(chars)) 
            new_seq = seq[:pos] + chars[m] + seq[pos+1:]
        seq = new_seq
    if not seq in base_seqs:
        return new_seq
    else:
        return close(seq)

def not_receptors(n):
    lst = []
    for i in range(n):
        length = randint(6,28)
        first = ch_begin[randint(len(ch_begin))]
        last = ch_end[randint(len(ch_end))]
        middle = str()
        middle_ind = random_integers(len(chars)-1, size=length-2)
        for k in middle_ind:
            middle += chars[k]
        seq = first + middle + last
        lst.append(seq)
    return lst

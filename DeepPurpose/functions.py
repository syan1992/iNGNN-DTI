import numpy as np
import torch
def hilbert_curve(n):
    # recursion base
    if n == 1:
        return np.zeros((1, 1), np.int32)
    # make (n/2, n/2) index
    t = hilbert_curve(n // 2)
    # flip it four times and add index offsets
    a = list(np.flipud(np.rot90(t)))
    b = list(t + t.size)
    c = list(t + t.size * 2)
    d = list(np.flipud(np.rot90(t, -1)) + t.size * 3)
    # and stack four tiles into resulting array
    return np.vstack(list(map(np.hstack, [[a, b], [d, c]])))

def plot_hb_dna(seq, H_curve):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    '''
    r, c = H_curve.shape
    H_dna = np.zeros((r, c, seq.shape[1]))#.to(device)
    for i in range(len(seq)):
        H_dna[pos[i,0], pos[i,1], :] = seq[i, :]
    return H_dna
    '''

    r,c = H_curve.shape
    H_curve = np.array(H_curve).flatten()
    #H_dna = np.zeros((r*c, seq.shape[1]))#.to(device)
    H_dna = np.take(seq,list(H_curve),axis=0)

    #H_dna = list(map(lambda i:seq[i,:],list(H_curve)))
    #print(H_dna.shape)
    return np.reshape(H_dna,[r,c,seq.shape[1]])

'''
hh = hilbert_curve(2)
print(hh)
seq = np.random.random([2*2,2])
print(seq)
a = plot_hb_dna(seq,hh)
print(a)
'''




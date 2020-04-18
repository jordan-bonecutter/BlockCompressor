#
#
#
#

import numpy as np

def get_huff_symbols(array):
  # TODO: add functionality for specifying number
  # of bits in a symbol to be compressed
  #
  # find probabilities
  prob = {}
  for v in np.nditer(array):
    i = int(v)
    if i not in prob:
      prob[i] = 0
    prob[i] += 1

  # get rid of probability 0 entries
  probabilities = []
  for symbol, count in prob.items():
    probabilities.append([count, [symbol]])
  probabilities = sorted(probabilities, key=lambda x: x[0])
  while probabilities[0][0] == 0:
    del probabilities[0]

  # begin reducing
  L = len(probabilities)
  symbols = {}
  if L == 1:
    symbols[probabilities[0][1][0]] = '1'
    return symbols
  for _ in range(L - 1):
    # append symbol
    for v in probabilities[0][1]:
      if v not in symbols:
        symbols[v] = ''
      symbols[v] += '1'
    for v in probabilities[1][1]:
      if v not in symbols:
        symbols[v] = ''
      symbols[v] += '0'

    # combine first two elements
    probabilities[1][0] += probabilities[0][0]
    probabilities[1][1] += probabilities[0][1]
    del probabilities[0]

    # sort
    # TODO: use insertion sort as only sorting 1 element
    probabilities = sorted(probabilities, key=lambda x: x[0]) 

  for key in symbols.keys():
    symbols[key] = symbols[key][::-1]
  return symbols

def main():
  import cv2 as cv

  im = cv.imread('test.png')
  im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
  symbols = get_huff_symbols(im)
  print(symbols)
  s = ''

  for a in im:
    for i in a:
      s += symbols[i]
  tosave = int(s, 2)
  print(len('{0:b}'.format(tosave)))
  print(len(s))

if __name__ == '__main__':
  main()

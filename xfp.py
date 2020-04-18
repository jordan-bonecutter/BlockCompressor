#
#
#
#

from huffman import get_huff_symbols
import numpy as np
from bitstring import BitString

def _add_data(toAdd, compressedBytes, nextByte):
  # add bytes to compressedBytes bytestring
  for i in toAdd:
    nextByte += i
    if len(nextByte) == 8:
      compressedBytes.append(int(nextByte, 2))
      nextByte = ''

  return nextByte

def tobits(n, nbits=8, signed=False):
  # convert the integer n to a bit string of n bits
  # if signed then the returned value is 2's comp
  if not signed:
    return BitString(uint=n, length=nbits).bin
  else:
    return BitString(int=n, length=nbits).bin

def getsigned(n, nbits=8):  
  # cast uint n as an nbit int
  return BitString(uint=n, length=nbits).int
  
def minbits(n):
  # return the minimum amount of bits
  # required to represent integer n
  if n < 0:
    n = -(n + 1)
    n <<= 1
  ret = 1
  while n != 1 and n != 0:
    n >>= 1
    ret += 1
  return ret

def compressData(data):
  # initialize byte array
  compressedBytes = []
  nextByte = ''

  # get symbol table
  symbols = get_huff_symbols(data)

  # len of symbol table
  tableLen = len(symbols)
  compressedBytes += ([(tableLen&(0xff<<(8*i)))>>(8*i) for i in range(2)])

  # length of key
  if min(symbols.keys()) >= 0:
    # includes no negative values
    signed   = False
    keybits  = minbits(max(symbols.keys()))
    nextByte = _add_data(tobits(keybits, nbits=8), compressedBytes, nextByte)
    nextByte = _add_data('0', compressedBytes, nextByte) 
  else:
    # includes negative and positive values
    signed   = True
    posbits  = minbits(max(symbols.keys()))
    negbits  = minbits(min(symbols.keys()))
    if negbits > posbits:
      keybits = negbits
    else:
      keybits = posbits + 1

    nextByte = _add_data(tobits(keybits, nbits=8), compressedBytes, nextByte)
    nextByte = _add_data('1', compressedBytes, nextByte)

  # save symbol table
  for key, value in symbols.items():
    nextByte = _add_data(tobits(key, nbits=keybits, signed=signed), compressedBytes, nextByte)
    nextByte = _add_data(tobits(len(value), nbits=8), compressedBytes, nextByte)
    nextByte = _add_data(value, compressedBytes, nextByte)

  # save data np shape
  nextByte = _add_data(tobits(len(data.shape), nbits=8), compressedBytes, nextByte)
  for dim in data.shape:
    nextByte = _add_data(tobits(dim, nbits=64), compressedBytes, nextByte)

  # save number of symbols as 64 bit integer
  nextByte = _add_data(tobits(data.size, nbits=64), compressedBytes, nextByte)

  # compress data
  for v in np.nditer(data):
    i = int(v)
    nextByte = _add_data(symbols[i], compressedBytes, nextByte)
    
  # pad until last byte is full
  if nextByte != '':
    nextByte += '0'*(8 - len(nextByte))
    compressedBytes.append(int(nextByte, 2))

  return bytes(compressedBytes)

def decompressData(compressed):
  tableLen = compressed[0] + (compressed[1]<<8)
  keybits = compressed[2]
  bindata = ''
  for i in compressed[3:]:
    bindata += tobits(i, nbits=8)

  if bindata[0] == '0':
    signed = False
  else:
    signed = True

  rindex = 1
  tindex = 0
  table  = {}
  while tindex < tableLen:
    # get table key
    symbol = bindata[rindex:(rindex+keybits)]
    rindex += keybits

    symbol = int(symbol, 2)
    if signed:
      symbol = getsigned(symbol, keybits)

    # get value length
    vlen = bindata[rindex:(rindex+8)]
    rindex += 8

    vlen = int(vlen, 2)

    # get value
    val = bindata[rindex:(rindex+vlen)]
    rindex += vlen

    table[val] = symbol

    tindex += 1

  # get data shape
  slen = int(bindata[rindex:rindex+8], 2)
  rindex += 8
  shape = []
  for i in range(slen):
    shape.append(int(bindata[rindex:rindex+64], 2))
    rindex += 64
  shape = tuple(shape)

  # get compressed data length
  clen = int(bindata[rindex:rindex+64], 2)
  rindex += 64

  # decompress
  if keybits <= 8:
    if not signed:
      dc_dtype = np.uint8
    else:
      dc_dtype = np.int8
  elif keybits <= 16:
    if not signed:
      dc_dtype = np.uint16
    else:
      dc_dtype = np.int16
  elif keybits <= 32:
    if not signed:
      dc_dtype = np.uint32
    else:
      dc_dtype = np.int32
  else:
    if not signed:
      dc_dtype = np.uint64
    else:
      dc_dtype = np.int64

  # decompress array using symbol table
  decompressed = np.zeros(clen, dtype=dc_dtype)
  for dcindex in range(clen):
    curr = ''
    while curr not in table:
      curr += bindata[rindex]
      rindex += 1

    decompressed[dcindex] = table[curr]

  return decompressed.reshape(shape)

def main():
  testdata = np.zeros((10000), dtype=np.int16) 
  for i in range(100):
    testdata[i] = (random.random()*6553)*random.random()
  #testdata = np.array(list(open('../../bible.txt', 'rb').read())).astype(np.uint8)

  compressed = compressData(testdata)
  decompressed = decompressData(compressed)

  iterator = np.nditer(op=[testdata, decompressed], flags=['c_index'])
  for truth, test in iterator:
    if truth != test:
      print('Failure at index ' + str(iterator.index))
      return 1

  print('Success with compression ratio: ' + str(len(compressed)/testdata.size))

  return 0

if __name__ == '__main__':
  import numpy as np
  import random
  import cv2 as cv
  main()

#
#
#
#

import numpy as np
import cv2 as cv
from scipy.fftpack import dct, idct
from math import sqrt
import struct
import xfp

def dct2d(im):
  # take the dct along rows and cols
  return dct(dct(im, axis=0)/sqrt(im.shape[0]), axis=1)/(2*sqrt(im.shape[1]))

def idct2d(im):
  # take idct along rows and cols
  return idct(idct(im, axis=1)/sqrt(im.shape[1]), axis=0)/(2*sqrt(im.shape[0]))

def pad(im, mod=8):
  # since we are using a block transform, the image size must
  # be a multiple of the block size
  rows, cols = im.shape[0], im.shape[1]
  rpad = (mod - (rows%mod))%mod
  cpad = (mod - (cols%mod))%mod

  if rpad == 0 and cpad == 0:
    return im

  newsize = list(im.shape)
  newsize[0], newsize[1] = rows+rpad, cols+cpad
  ret = np.zeros(tuple(newsize), dtype=np.uint8)
  ret[:rows, :cols] = im

  # mirror padding
  for y in range(rpad):
      ret[rows + y, :cols] = im[rows - 1 - y, :]
  for x in range(cpad):
    ret[:rows, cols + x] = im[:, cols - 1 - x]
  for y in range(rpad):
    for x in range(cpad):
      ret[rows + y, cols + x] = im[rows - 1 - y, cols - 1 - x]
  return ret

def blockdct(im, bsize=8):
  # take dcts in blocks of the image
  padded = pad(im, mod=bsize)
  ret = np.zeros(padded.shape)

  for i in range(padded.shape[0]//bsize):
    for j in range(padded.shape[1]//bsize):
      ret[i*bsize:(i+1)*bsize, j*bsize:(j+1)*bsize] = dct2d(padded[i*bsize:(i+1)*bsize, j*bsize:(j+1)*bsize])

  return ret
  
def blocksqueeze(bdct, bsize=8, quality=0.8):
  # take smallest values and quash them to 0
  for i in range(bdct.shape[0]//bsize):
    for j in range(bdct.shape[1]//bsize):
      # find lowest energy cells
      cells = []
      for y in range(bsize):
        for x in range(bsize):
          cells.append((abs(bdct[(i*bsize) + y, (j*bsize) + x]), (y, x)))
      
      # sort the cells by weight
      cells = sorted(cells, key=lambda x: x[0])

      # get rid of the cells w/ lightest weight
      for cell in cells[:len(cells) - int(quality*(len(cells) - 1))]:
        bdct[(i*bsize) + cell[1][0], (j*bsize) + cell[1][1]] = 0

def quantize(bdct, bits=11):
  # quantize the transform coefficients
  ret  = np.zeros(bdct.shape, dtype=np.int16)
  maxq = 1<<(bits-1)
  scale = max(abs(bdct.max()), abs(bdct.min()))
  ret[:, :] = (maxq - 1)*(bdct / scale)
  return scale, ret

def dequantize(quan, scale, bits=11):
  # bring the transform coefficients back 
  # to their original values (or as close
  # as you can with quantization noise)
  maxq = 1<<(bits-1)
  return (quan / (maxq - 1)) * scale

def iblockdct(bdct, bsize=8):
  # take the idct on each block of the image
  ret = np.zeros(bdct.shape, dtype=np.uint8)
  
  for i in range(bdct.shape[0]//bsize):
    for j in range(bdct.shape[1]//bsize):
      tmp = idct2d(bdct[i*bsize:(i+1)*bsize, j*bsize:(j+1)*bsize])
      for y in range(bsize):
        for x in range(bsize):
          # make sure the value is in range for an 8 bit rgb image
          ret[(i*bsize) + y, (j*bsize) + x] = min(255, max(0, tmp[y, x]))

  return ret

def float2bits(f):
  # get the bits for 32 bit float values
  s = struct.pack('>f', f)
  return struct.unpack('>l', s)[0]

def bits2float(b):
  # get the float for a 32 bit value
  if b > 2147483647:
    b -= 4294967296
  elif b < -2147483648:
    b += 4294967296
  s = struct.pack('>l', b)
  return struct.unpack('>f', s)[0]

def dctcompress(im, dctquality=0.8, bitres=12, bsize=8):
  # we need to know the original image size before padding,
  # the blocksize used, and the number of bits per pixel in
  # the quantization stage. We will encode these first and
  # then use huffman compression to compress the dct coeff
  rows, cols = im.shape[0], im.shape[1]
  bdct = blockdct(im, bsize=bsize)
  blocksqueeze(bdct, quality=dctquality, bsize=bsize)
  scale, quan = quantize(bdct, bits=bitres)
  sbits = float2bits(scale)
  return bytes([(sbits&(0xff<<(8*i)))>>(8*i) for i in range(4)]) + \
         bytes([(rows &(0xff<<(8*i)))>>(8*i) for i in range(8)]) + \
         bytes([(cols &(0xff<<(8*i)))>>(8*i) for i in range(8)]) + \
         bytes([bsize, bitres]) + xfp.compressData(quan)

def dctdecompress(comp):
  # retrieve the values from the compressed bytestring
  sbits = comp[0] + (comp[1]<<8) + (comp[2]<<16) + (comp[3]<<24)
  rows, cols = 0, 0
  for i in range(8):
    rows |= comp[4 + i]<<(8*i)
    cols |= comp[12 + i]<<(8*i)
  scale = bits2float(sbits)

  bsize  = comp[20]
  bitres = comp[21]
  quan = xfp.decompressData(comp[22:])
  bdct = dequantize(quan, scale, bits=bitres)
  return iblockdct(bdct, bsize=bsize)[:rows, :cols]

def main():
  i = cv.imread('test.png')
  i = cv.cvtColor(i, cv.COLOR_BGR2GRAY)
  comp = dctcompress(i, dctquality=0.6, bitres=9, bsize=8)
  print(len(comp)/i.size)
  dcmp = dctdecompress(comp)
  cv.imwrite('dctcomptest.png', dcmp)

def generateBasisImage(bsize=8):
  # just for funsies
  ret = np.zeros((bsize**2, bsize**2), dtype=np.uint8)
  
  for y in range(bsize):
    for x in range(bsize):
      tmp = np.zeros((bsize, bsize), dtype=np.uint8)
      tmp[y, x] = 1
      tmp = idct2d(tmp)

      if tmp.max() == tmp.min():
        tmp = tmp - tmp.max() + 255
      else:
        tmp = 255. * ((tmp - tmp.min()) / (tmp.max() - tmp.min()))
      ret[y*bsize:(y+1)*bsize, x*bsize:(x+1)*bsize] = tmp
  
  return ret

if __name__ == '__main__':
  main()


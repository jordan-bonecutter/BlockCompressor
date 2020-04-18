#
#
#
#

import cv2 as cv
import numpy as np
import xfp

def _get_diff_image(im):
  # x derivative without first column
  newshape = list(im.shape)
  newshape[1] -= 1
  newshape = tuple(newshape)
  ret = np.zeros(newshape, dtype=np.uint16)

  for y in range(newshape[0]):
    for x in range(1, newshape[1]):
      ret[y, x] = int(im[y, x]) - int(im[y, x - 1])

  return ret

def compress(im):
  # losslessly compress im
  leftcol = np.zeros(im.shape[0], dtype=np.uint8)
  for y in range(im.shape[0]):
    leftcol[y] = im[y, 0]

  diffim = _get_diff_image(im)
  diffimcomp = xfp.compressData(diffim)
  
  return bytes([(leftcol.size&(0xff<<(8*i)))>>(8*i) for i in range(4)]) + bytes(leftcol) + diffimcomp

def decompress(compressed):
  # decompress compressed image
  rows    = compressed[0] + (compressed[1]<<8) + (compressed[2]<<16) + (compressed[3]<<24)
  diffim  = xfp.decompressData(compressed[4+rows:])
  ret     = np.zeros((rows, diffim.shape[1]+1), dtype=np.uint8)

  # since we are encoding the difference image,
  #  we need to integrate along the x direction
  for y in range(ret.shape[0]):
    ret[y, 0] = compressed[4+y]
    for x in range(1, ret.shape[1]):
      ret[y, x] = (diffim[y, x-1]) + (ret[y, x-1])
  return ret

def colorcompress(im):
  # compress each channel separately
  red = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
  grn = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)
  blu = np.zeros((im.shape[0], im.shape[1]), dtype=np.uint8)

  for y in range(im.shape[0]):
    for x in range(im.shape[1]):
      red[y, x], grn[y, x], blu[y, x] = im[y, x]

  cr, cg, cb = compress(red), compress(grn), compress(blu)
  # we also need to save the length of the compressed channels
  # so we can split them up when decoding
  return bytes([(len(cr)&(0xff<<(8*i)))>>(8*i) for i in range(8)]) + cr + \
         bytes([(len(cg)&(0xff<<(8*i)))>>(8*i) for i in range(8)]) + cg + cb

def colordecompress(cdc):
  # split and decode the compressed channels
  rlen = 0
  for i in range(8):
    rlen += cdc[i]<<(8*i)
  cr = cdc[8:rlen+8]

  glen = 0
  for i in range(8):
    glen += cdc[rlen+8+i]<<(8*i)
  cg = cdc[rlen+16:rlen+glen+16]

  cb = cdc[rlen+glen+16:]

  # combine the channels into one image
  r, g, b = decompress(cr), decompress(cg), decompress(cb)
  ret = np.zeros((r.shape[0], r.shape[1], 3), dtype=np.uint8)
  for y in range(r.shape[0]):
    for x in range(r.shape[1]):
      ret[y, x] = r[y, x], g[y, x], b[y, x]

  return ret

def main():
  i = cv.imread('test.png')
  compressed = colorcompress(i)
  with open('test.xfp', 'wb') as fi:
    fi.write(compressed)
  print(len(compressed)/i.size)
  dc = colordecompress(compressed)
  cv.imwrite('dctest.png', dc)
  return 0

if __name__ == '__main__':
  main()
   

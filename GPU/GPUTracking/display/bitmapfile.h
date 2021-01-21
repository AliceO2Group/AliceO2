// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file bitmapfile.h
/// \author David Rohr

struct BITMAPFILEHEADER {
  unsigned short bfType;
  unsigned int bfSize;
  unsigned int bfReserved;
  unsigned int bfOffBits;
} __attribute__((packed));

struct BITMAPINFOHEADER {
  unsigned int biSize;
  unsigned int biWidth;
  unsigned int biHeight;
  unsigned short biPlanes;
  unsigned short biBitCount;
  unsigned int biCompression;
  unsigned int biSizeImage;
  unsigned int biXPelsPerMeter;
  unsigned int biYPelsPerMeter;
  unsigned int biClrUsed;
  unsigned int biClrImportant;
} __attribute__((packed));

enum BI_Compression { BI_RGB = 0x0000,
                      BI_RLE8 = 0x0001,
                      BI_RLE4 = 0x0002,
                      BI_BITFIELDS = 0x0003,
                      BI_JPEG = 0x0004,
                      BI_PNG = 0x0005,
                      BI_CMYK = 0x000B,
                      BI_CMYKRLE8 = 0x000C,
                      BI_CMYKRLE4 = 0x000D };

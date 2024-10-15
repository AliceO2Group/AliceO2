// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file bitmapfile.h
/// \author David Rohr

struct BITMAPFILEHEADER {
  uint16_t bfType;
  uint32_t bfSize;
  uint32_t bfReserved;
  uint32_t bfOffBits;
} __attribute__((packed));

struct BITMAPINFOHEADER {
  uint32_t biSize;
  uint32_t biWidth;
  uint32_t biHeight;
  uint16_t biPlanes;
  uint16_t biBitCount;
  uint32_t biCompression;
  uint32_t biSizeImage;
  uint32_t biXPelsPerMeter;
  uint32_t biYPelsPerMeter;
  uint32_t biClrUsed;
  uint32_t biClrImportant;
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

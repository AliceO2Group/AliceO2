// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// This file has been generated. Do not modify it by hand or your changes might be lost.
//
#include "SegmentationCreator.h"

namespace o2
{
namespace mch
{
namespace mapping
{
namespace impl3
{
Segmentation* createSegType4(bool isBendingPlane)
{
  if (isBendingPlane) {
    return new Segmentation{
      4,
      true,
      /* PG */
      { { 1, 2, 0, -80, -20 },    { 2, 17, 0, -77.5, -20 }, { 3, 4, 0, -72.5, -20 },  { 4, 18, 0, -70, -20 },
        { 5, 3, 0, -65, -20 },    { 6, 2, 0, -60, -20 },    { 7, 17, 0, -57.5, -20 }, { 8, 4, 0, -52.5, -20 },
        { 9, 18, 0, -50, -20 },   { 10, 3, 0, -45, -20 },   { 104, 6, 1, 40, -20 },   { 105, 7, 1, 45, -20 },
        { 106, 8, 1, 55, -20 },   { 107, 9, 1, 60, -20 },   { 111, 2, 1, 0, -20 },    { 112, 17, 1, 5, -20 },
        { 113, 4, 1, 15, -20 },   { 114, 18, 1, 20, -20 },  { 115, 3, 1, 30, -20 },   { 119, 2, 1, -40, -20 },
        { 120, 17, 1, -35, -20 }, { 121, 4, 1, -25, -20 },  { 122, 18, 1, -20, -20 }, { 123, 3, 1, -10, -20 },
        { 201, 10, 1, 70, -12 },  { 202, 11, 1, 60, 0 },    { 203, 12, 1, 55, 4 },    { 204, 13, 1, 45, 0 },
        { 205, 14, 1, 40, 0 },    { 209, 1, 1, 30, 0 },     { 210, 16, 1, 20, 0 },    { 211, 5, 1, 15, 4 },
        { 212, 15, 1, 5, 0 },     { 213, 0, 1, 0, 0 },      { 218, 1, 1, -10, 0 },    { 219, 16, 1, -20, 0 },
        { 220, 5, 1, -25, 4 },    { 221, 15, 1, -35, 0 },   { 222, 0, 1, -40, 0 },    { 308, 1, 0, -45, 0 },
        { 309, 16, 0, -50, 0 },   { 310, 5, 0, -52.5, 4 },  { 311, 15, 0, -57.5, 0 }, { 312, 0, 0, -60, 0 },
        { 313, 1, 0, -65, 0 },    { 314, 16, 0, -70, 0 },   { 315, 5, 0, -72.5, 4 },  { 316, 15, 0, -77.5, 0 },
        { 317, 0, 0, -80, 0 } },
      /* PGT */
      { /* L5 */ { 2, 40, { 55, 56, 54, 57, 53, 58, 52, 59, 51, 60, 50, 61, 49, 62, 48, 63, 31, 32, 30, 33,
                            29, 34, 28, 35, 27, 36, 26, 37, 25, 38, 24, 39, 23, 40, 22, 41, 21, 42, 20, 43,
                            19, 44, 18, 45, 17, 46, 16, 47, 15, -1, 14, -1, 13, -1, 12, -1, 11, -1, 10, -1,
                            9,  -1, 8,  -1, 7,  -1, 6,  -1, 5,  -1, 4,  -1, 3,  -1, 2,  -1, 1,  -1, 0,  -1 } },
        /* L6 */ { 2, 40, { 23, 24, 22, 25, 21, 26, 20, 27, 19, 28, 18, 29, 17, 30, 16, 31, 15, 48, 14, 49,
                            13, 50, 12, 51, 11, 52, 10, 53, 9,  54, 8,  55, 7,  56, 6,  57, 5,  58, 4,  59,
                            3,  60, 2,  61, 1,  62, 0,  63, -1, 32, -1, 33, -1, 34, -1, 35, -1, 36, -1, 37,
                            -1, 38, -1, 39, -1, 40, -1, 41, -1, 42, -1, 43, -1, 44, -1, 45, -1, 46, -1, 47 } },
        /* L7 */ { 2, 40, { 47, -1, 46, -1, 45, -1, 44, -1, 43, -1, 42, -1, 41, -1, 40, -1, 39, -1, 38, -1,
                            37, -1, 36, -1, 35, -1, 34, -1, 33, -1, 32, -1, 63, 0,  62, 1,  61, 2,  60, 3,
                            59, 4,  58, 5,  57, 6,  56, 7,  55, 8,  54, 9,  53, 10, 52, 11, 51, 12, 50, 13,
                            49, 14, 48, 15, 31, 16, 30, 17, 29, 18, 28, 19, 27, 20, 26, 21, 25, 22, 24, 23 } },
        /* L8 */ { 2, 40, { -1, 0,  -1, 1,  -1, 2,  -1, 3,  -1, 4,  -1, 5,  -1, 6,  -1, 7,  -1, 8,  -1, 9,
                            -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, 47, 16, 46, 17, 45, 18, 44, 19,
                            43, 20, 42, 21, 41, 22, 40, 23, 39, 24, 38, 25, 37, 26, 36, 27, 35, 28, 34, 29,
                            33, 30, 32, 31, 63, 48, 62, 49, 61, 50, 60, 51, 59, 52, 58, 53, 57, 54, 56, 55 } },
        /* O10 */ { 2, 32, { 48, 31, 49, 30, 50, 29, 51, 28, 52, 27, 53, 26, 54, 25, 55, 24, 56, 23, 57, 22, 58, 21,
                             59, 20, 60, 19, 61, 18, 62, 17, 63, 16, 32, 15, 33, 14, 34, 13, 35, 12, 36, 11, 37, 10,
                             38, 9,  39, 8,  40, 7,  41, 6,  42, 5,  43, 4,  44, 3,  45, 2,  46, 1,  47, 0 } },
        /* O9 */ { 2, 32, { 0,  47, 1,  46, 2,  45, 3,  44, 4,  43, 5,  42, 6,  41, 7,  40, 8,  39, 9,  38, 10, 37,
                            11, 36, 12, 35, 13, 34, 14, 33, 15, 32, 16, 63, 17, 62, 18, 61, 19, 60, 20, 59, 21, 58,
                            22, 57, 23, 56, 24, 55, 25, 54, 26, 53, 27, 52, 28, 51, 29, 50, 30, 49, 31, 48 } },
        /* S0 */ { 2, 40, { 47, -1, 46, -1, 45, -1, 44, -1, 43, -1, 42, -1, 41, -1, 40, -1, 39, -1, 38, -1,
                            37, -1, 36, -1, 35, -1, 34, -1, 33, -1, 32, -1, 63, 0,  62, 1,  61, 2,  60, 3,
                            59, 4,  58, 5,  57, 6,  56, 7,  55, 8,  54, 9,  53, 10, 52, 11, 51, 12, 50, 13,
                            49, 14, 48, 15, 31, 16, 30, 17, 29, 18, 28, 19, 27, 20, 26, 21, 25, 22, 24, 23 } },
        /* S1 */ { 3, 40, { 32, 63, -1, 33, 62, -1, 34, 61, -1, 35, 60, -1, 36, 59, -1, 37, 58, -1, 38, 57,
                            -1, 39, 56, -1, 40, 55, -1, 41, 54, -1, 42, 53, -1, 43, 52, -1, 44, 51, -1, 45,
                            50, -1, 46, 49, -1, 47, 48, -1, -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1,
                            -1, 27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1, -1, 23, -1, -1, 22, -1, -1, 21,
                            -1, -1, 20, -1, -1, 19, -1, -1, 18, -1, -1, 17, -1, -1, 16, -1, -1, 15, 0,  -1,
                            14, 1,  -1, 13, 2,  -1, 12, 3,  -1, 11, 4,  -1, 10, 5,  -1, 9,  6,  -1, 8,  7 } },
        /* S2 */ { 2, 32, { 48, 31, 49, 30, 50, 29, 51, 28, 52, 27, 53, 26, 54, 25, 55, 24, 56, 23, 57, 22, 58, 21,
                            59, 20, 60, 19, 61, 18, 62, 17, 63, 16, 32, 15, 33, 14, 34, 13, 35, 12, 36, 11, 37, 10,
                            38, 9,  39, 8,  40, 7,  41, 6,  42, 5,  43, 4,  44, 3,  45, 2,  46, 1,  47, 0 } },
        /* S3 */ { 3, 40, { -1, 16, 15, -1, 17, 14, -1, 18, 13, -1, 19, 12, -1, 20, 11, -1, 21, 10, -1, 22,
                            9,  -1, 23, 8,  -1, 24, 7,  -1, 25, 6,  -1, 26, 5,  -1, 27, 4,  -1, 28, 3,  -1,
                            29, 2,  -1, 30, 1,  -1, 31, 0,  -1, 48, -1, -1, 49, -1, -1, 50, -1, -1, 51, -1,
                            -1, 52, -1, -1, 53, -1, -1, 54, -1, -1, 55, -1, -1, 56, -1, -1, 57, -1, -1, 58,
                            -1, -1, 59, -1, -1, 60, -1, -1, 61, -1, -1, 62, -1, -1, 63, -1, 47, 32, -1, 46,
                            33, -1, 45, 34, -1, 44, 35, -1, 43, 36, -1, 42, 37, -1, 41, 38, -1, 40, 39, -1 } },
        /* S4 */ { 1, 64, { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                            22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
                            60, 61, 62, 63, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47 } },
        /* S5 */ { 2, 40, { 7,  8,  6,  9,  5,  10, 4,  11, 3,  12, 2,  13, 1,  14, 0,  15, -1, 16, -1, 17,
                            -1, 18, -1, 19, -1, 20, -1, 21, -1, 22, -1, 23, -1, 24, -1, 25, -1, 26, -1, 27,
                            -1, 28, -1, 29, -1, 30, -1, 31, -1, 48, -1, 49, -1, 50, -1, 51, -1, 52, -1, 53,
                            -1, 54, -1, 55, -1, 56, -1, 57, -1, 58, -1, 59, -1, 60, -1, 61, -1, 62, -1, 63 } },
        /* S6 */ { 2, 32, { 0,  47, 1,  46, 2,  45, 3,  44, 4,  43, 5,  42, 6,  41, 7,  40, 8,  39, 9,  38, 10, 37,
                            11, 36, 12, 35, 13, 34, 14, 33, 15, 32, 16, 63, 17, 62, 18, 61, 19, 60, 20, 59, 21, 58,
                            22, 57, 23, 56, 24, 55, 25, 54, 26, 53, 27, 52, 28, 51, 29, 50, 30, 49, 31, 48 } },
        /* S7 */ { 3, 40, { -1, 39, 40, -1, 38, 41, -1, 37, 42, -1, 36, 43, -1, 35, 44, -1, 34, 45, -1, 33,
                            46, -1, 32, 47, -1, 63, -1, -1, 62, -1, -1, 61, -1, -1, 60, -1, -1, 59, -1, -1,
                            58, -1, -1, 57, -1, -1, 56, -1, -1, 55, -1, -1, 54, -1, -1, 53, -1, -1, 52, -1,
                            -1, 51, -1, -1, 50, -1, -1, 49, -1, -1, 48, -1, 0,  31, -1, 1,  30, -1, 2,  29,
                            -1, 3,  28, -1, 4,  27, -1, 5,  26, -1, 6,  25, -1, 7,  24, -1, 8,  23, -1, 9,
                            22, -1, 10, 21, -1, 11, 20, -1, 12, 19, -1, 13, 18, -1, 14, 17, -1, 15, 16, -1 } },
        /* S8 */ { 2, 40, { 55, 56, 54, 57, 53, 58, 52, 59, 51, 60, 50, 61, 49, 62, 48, 63, 31, 32, 30, 33,
                            29, 34, 28, 35, 27, 36, 26, 37, 25, 38, 24, 39, 23, 40, 22, 41, 21, 42, 20, 43,
                            19, 44, 18, 45, 17, 46, 16, 47, 15, -1, 14, -1, 13, -1, 12, -1, 11, -1, 10, -1,
                            9,  -1, 8,  -1, 7,  -1, 6,  -1, 5,  -1, 4,  -1, 3,  -1, 2,  -1, 1,  -1, 0,  -1 } },
        /* Z1 */ { 3, 40, { -1, 39, 40, -1, 38, 41, -1, 37, 42, -1, 36, 43, -1, 35, 44, -1, 34, 45, -1, 33,
                            46, -1, 32, 47, -1, 63, -1, -1, 62, -1, -1, 61, -1, -1, 60, -1, -1, 59, -1, -1,
                            58, -1, -1, 57, -1, -1, 56, -1, -1, 55, -1, -1, 54, -1, -1, 53, -1, -1, 52, -1,
                            -1, 51, -1, -1, 50, -1, -1, 49, -1, -1, 48, -1, 0,  31, -1, 1,  30, -1, 2,  29,
                            -1, 3,  28, -1, 4,  27, -1, 5,  26, -1, 6,  25, -1, 7,  24, -1, 8,  23, -1, 9,
                            22, -1, 10, 21, -1, 11, 20, -1, 12, 19, -1, 13, 18, -1, 14, 17, -1, 15, 16, -1 } },
        /* Z2 */ { 3, 40, { 7,  8,  -1, 6,  9,  -1, 5,  10, -1, 4,  11, -1, 3,  12, -1, 2,  13, -1, 1,  14,
                            -1, 0,  15, -1, -1, 16, -1, -1, 17, -1, -1, 18, -1, -1, 19, -1, -1, 20, -1, -1,
                            21, -1, -1, 22, -1, -1, 23, -1, -1, 24, -1, -1, 25, -1, -1, 26, -1, -1, 27, -1,
                            -1, 28, -1, -1, 29, -1, -1, 30, -1, -1, 31, -1, -1, 48, 47, -1, 49, 46, -1, 50,
                            45, -1, 51, 44, -1, 52, 43, -1, 53, 42, -1, 54, 41, -1, 55, 40, -1, 56, 39, -1,
                            57, 38, -1, 58, 37, -1, 59, 36, -1, 60, 35, -1, 61, 34, -1, 62, 33, -1, 63, 32 } },
        /* Z3 */ { 3, 40, { 32, 63, -1, 33, 62, -1, 34, 61, -1, 35, 60, -1, 36, 59, -1, 37, 58, -1, 38, 57,
                            -1, 39, 56, -1, 40, 55, -1, 41, 54, -1, 42, 53, -1, 43, 52, -1, 44, 51, -1, 45,
                            50, -1, 46, 49, -1, 47, 48, -1, -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1,
                            -1, 27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1, -1, 23, -1, -1, 22, -1, -1, 21,
                            -1, -1, 20, -1, -1, 19, -1, -1, 18, -1, -1, 17, -1, -1, 16, -1, -1, 15, 0,  -1,
                            14, 1,  -1, 13, 2,  -1, 12, 3,  -1, 11, 4,  -1, 10, 5,  -1, 9,  6,  -1, 8,  7 } },
        /* Z4 */ { 3, 40, { -1, 16, 15, -1, 17, 14, -1, 18, 13, -1, 19, 12, -1, 20, 11, -1, 21, 10, -1, 22,
                            9,  -1, 23, 8,  -1, 24, 7,  -1, 25, 6,  -1, 26, 5,  -1, 27, 4,  -1, 28, 3,  -1,
                            29, 2,  -1, 30, 1,  -1, 31, 0,  -1, 48, -1, -1, 49, -1, -1, 50, -1, -1, 51, -1,
                            -1, 52, -1, -1, 53, -1, -1, 54, -1, -1, 55, -1, -1, 56, -1, -1, 57, -1, -1, 58,
                            -1, -1, 59, -1, -1, 60, -1, -1, 61, -1, -1, 62, -1, -1, 63, -1, 47, 32, -1, 46,
                            33, -1, 45, 34, -1, 44, 35, -1, 43, 36, -1, 42, 37, -1, 41, 38, -1, 40, 39, -1 } } },
      /* PS */
      { { 2.5, 0.5 }, { 5, 0.5 } }
    };
  } else {
    return new Segmentation{
      4,
      false,
      /* PG */
      { { 1035, 3, 0, -45.7142868, -20 },
        { 1036, 3, 0, -51.42856979, -20 },
        { 1037, 3, 0, -57.1428566, -20 },
        { 1038, 3, 0, -62.8571434, -20 },
        { 1039, 3, 0, -68.57142639, -20 },
        { 1040, 3, 0, -74.2857132, -20 },
        { 1041, 3, 0, -80, -20 },
        { 1125, 4, 1, 62.8571434, -20 },
        { 1126, 4, 1, 51.42856979, -20 },
        { 1127, 4, 1, 40, -20 },
        { 1132, 1, 1, 25.7142849, -20 },
        { 1133, 4, 1, 14.28571415, -20 },
        { 1134, 0, 1, 4.440892099e-15, -20 },
        { 1140, 1, 1, -14.28571415, -20 },
        { 1141, 4, 1, -25.7142849, -20 },
        { 1142, 0, 1, -40, -20 },
        { 1230, 5, 1, 40, 0 },
        { 1231, 5, 1, 51.42856979, 0 },
        { 1232, 5, 1, 62.8571434, 0 },
        { 1238, 8, 1, -7.105427358e-15, -5 },
        { 1239, 6, 1, 8.571428299, -5 },
        { 1240, 7, 1, 20, -5 },
        { 1241, 9, 1, 30, -5 },
        { 1247, 8, 1, -40, -5 },
        { 1248, 6, 1, -31.4285717, -5 },
        { 1249, 7, 1, -20, -5 },
        { 1250, 9, 1, -10, -5 },
        { 1325, 2, 0, -80, 0 },
        { 1326, 2, 0, -74.2857132, 0 },
        { 1327, 2, 0, -68.57142639, 0 },
        { 1328, 2, 0, -62.8571434, 0 },
        { 1329, 2, 0, -57.1428566, 0 },
        { 1330, 2, 0, -51.42856979, 0 },
        { 1331, 2, 0, -45.7142868, 0 } },
      /* PGT */
      { /* L1 */ { 20, 4, { 3, 7, 11, 15, 18, 21, 24, 27, 30, 49, 52, 55, 58, 61, 32, 35, 38, 41, 44, 47,
                            2, 6, 10, 14, 17, 20, 23, 26, 29, 48, 51, 54, 57, 60, 63, 34, 37, 40, 43, 46,
                            1, 5, 9,  13, 16, 19, 22, 25, 28, 31, 50, 53, 56, 59, 62, 33, 36, 39, 42, 45,
                            0, 4, 8,  12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1 } },
        /* L2 */ { 20, 4, { 2,  5,  8,  11, 14, 17, 20, 23, 26, 29, 48, 51, 54, 57, 60, 63, 35, 39, 43, 47,
                            1,  4,  7,  10, 13, 16, 19, 22, 25, 28, 31, 50, 53, 56, 59, 62, 34, 38, 42, 46,
                            0,  3,  6,  9,  12, 15, 18, 21, 24, 27, 30, 49, 52, 55, 58, 61, 33, 37, 41, 45,
                            -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 32, 36, 40, 44 } },
        /* O1 */ { 8, 8, { 40, 32, 56, 48, 24, 16, 8,  0,  41, 33, 57, 49, 25, 17, 9,  1,  42, 34, 58, 50, 26, 18,
                           10, 2,  43, 35, 59, 51, 27, 19, 11, 3,  44, 36, 60, 52, 28, 20, 12, 4,  45, 37, 61, 53,
                           29, 21, 13, 5,  46, 38, 62, 54, 30, 22, 14, 6,  47, 39, 63, 55, 31, 23, 15, 7 } },
        /* O2 */ { 8, 8, { 7,  15, 23, 31, 55, 63, 39, 47, 6,  14, 22, 30, 54, 62, 38, 46, 5,  13, 21, 29, 53, 61,
                           37, 45, 4,  12, 20, 28, 52, 60, 36, 44, 3,  11, 19, 27, 51, 59, 35, 43, 2,  10, 18, 26,
                           50, 58, 34, 42, 1,  9,  17, 25, 49, 57, 33, 41, 0,  8,  16, 24, 48, 56, 32, 40 } },
        /* O3 */ { 16, 4, { 3,  7,  11, 15, 19, 23, 27, 31, 51, 55, 59, 63, 35, 39, 43, 47, 2,  6,  10, 14, 18, 22,
                            26, 30, 50, 54, 58, 62, 34, 38, 42, 46, 1,  5,  9,  13, 17, 21, 25, 29, 49, 53, 57, 61,
                            33, 37, 41, 45, 0,  4,  8,  12, 16, 20, 24, 28, 48, 52, 56, 60, 32, 36, 40, 44 } },
        /* O4 */ { 16, 4, { 44, 40, 36, 32, 60, 56, 52, 48, 28, 24, 20, 16, 12, 8,  4,  0,  45, 41, 37, 33, 61, 57,
                            53, 49, 29, 25, 21, 17, 13, 9,  5,  1,  46, 42, 38, 34, 62, 58, 54, 50, 30, 26, 22, 18,
                            14, 10, 6,  2,  47, 43, 39, 35, 63, 59, 55, 51, 31, 27, 23, 19, 15, 11, 7,  3 } },
        /* P1 */ { 16, 5, { 47, 46, 41, 36, 63, 58, 53, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 42, 37,
                            32, 59, 54, 49, 28, 24, 20, 16, 12, 8,  4,  0,  -1, -1, 43, 38, 33, 60, 55, 50,
                            29, 25, 21, 17, 13, 9,  5,  1,  -1, -1, 44, 39, 34, 61, 56, 51, 30, 26, 22, 18,
                            14, 10, 6,  2,  -1, -1, 45, 40, 35, 62, 57, 52, 31, 27, 23, 19, 15, 11, 7,  3 } },
        /* P2 */ { 16, 5, { -1, -1, -1, -1, -1, -1, -1, -1, 27, 22, 17, 12, 7,  2,  1,  0,  44, 40, 36, 32,
                            60, 56, 52, 48, 28, 23, 18, 13, 8,  3,  -1, -1, 45, 41, 37, 33, 61, 57, 53, 49,
                            29, 24, 19, 14, 9,  4,  -1, -1, 46, 42, 38, 34, 62, 58, 54, 50, 30, 25, 20, 15,
                            10, 5,  -1, -1, 47, 43, 39, 35, 63, 59, 55, 51, 31, 26, 21, 16, 11, 6,  -1, -1 } },
        /* Q1 */ { 14, 5, { -1, -1, -1, -1, 59, 54, 49, 28, 23, 18, 13, 8,  -1, -1, 44, 40, 36, 32,
                            60, 55, 50, 29, 24, 19, 14, 9,  4,  0,  45, 41, 37, 33, 61, 56, 51, 30,
                            25, 20, 15, 10, 5,  1,  46, 42, 38, 34, 62, 57, 52, 31, 26, 21, 16, 11,
                            6,  2,  47, 43, 39, 35, 63, 58, 53, 48, 27, 22, 17, 12, 7,  3 } },
        /* Q2 */ { 14, 5, { -1, -1, 35, 62, 57, 52, 31, 26, 21, 16, -1, -1, -1, -1, 44, 40, 36, 63,
                            58, 53, 48, 27, 22, 17, 12, 8,  4,  0,  45, 41, 37, 32, 59, 54, 49, 28,
                            23, 18, 13, 9,  5,  1,  46, 42, 38, 33, 60, 55, 50, 29, 24, 19, 14, 10,
                            6,  2,  47, 43, 39, 34, 61, 56, 51, 30, 25, 20, 15, 11, 7,  3 } } },
      /* PS */
      { { 0.714285714, 2.5 }, { 0.714285714, 5 } }
    };
  }
}
class SegmentationCreatorRegisterCreateSegType4
{
 public:
  SegmentationCreatorRegisterCreateSegType4() { registerSegmentationCreator(4, createSegType4); }
} aSegmentationCreatorRegisterCreateSegType4;

} // namespace impl3
} // namespace mapping
} // namespace mch
} // namespace o2

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// This file has been generated. Do not modify it by hand or your changes might
// be lost.
//
#include "CathodeSegmentationCreator.h"

namespace o2
{
namespace mch
{
namespace mapping
{
namespace impl4
{
CathodeSegmentation* createSegType8(bool isBendingPlane)
{
  if (isBendingPlane) {
    return new CathodeSegmentation{
      8,
      true,
      /* PG */
      { { 4, 18, 0, 40, -20 },     { 5, 19, 0, 45, -20 },
        { 6, 19, 0, 50, -20 },     { 7, 11, 0, 55, -17.5 },
        { 101, 15, 1, -80, -20 },  { 102, 23, 1, -75, -20 },
        { 103, 17, 1, -65, -20 },  { 104, 24, 1, -60, -20 },
        { 105, 16, 1, -50, -20 },  { 110, 15, 1, -40, -20 },
        { 111, 23, 1, -35, -20 },  { 112, 17, 1, -25, -20 },
        { 113, 24, 1, -20, -20 },  { 114, 16, 1, -10, -20 },
        { 119, 15, 0, 0, -20 },    { 120, 23, 0, 2.5, -20 },
        { 121, 17, 0, 7.5, -20 },  { 122, 24, 0, 10, -20 },
        { 123, 16, 0, 15, -20 },   { 124, 15, 0, 20, -20 },
        { 125, 23, 0, 22.5, -20 }, { 126, 17, 0, 27.5, -20 },
        { 127, 24, 0, 30, -20 },   { 128, 16, 0, 35, -20 },
        { 204, 14, 1, -50, 0 },    { 205, 22, 1, -60, 0 },
        { 206, 20, 1, -65, 4 },    { 207, 21, 1, -75, 0 },
        { 208, 13, 1, -80, 0 },    { 212, 14, 1, -10, 0 },
        { 213, 22, 1, -20, 0 },    { 214, 20, 1, -25, 4 },
        { 215, 21, 1, -35, 0 },    { 216, 13, 1, -40, 0 },
        { 224, 14, 0, 35, 0 },     { 225, 22, 0, 30, 0 },
        { 226, 20, 0, 27.5, 4 },   { 227, 21, 0, 22.5, 0 },
        { 228, 13, 0, 20, 0 },     { 229, 14, 0, 15, 0 },
        { 230, 22, 0, 10, 0 },     { 231, 20, 0, 7.5, 4 },
        { 232, 21, 0, 2.5, 0 },    { 233, 13, 0, 0, 0 },
        { 401, 12, 0, 75, -7 },    { 402, 0, 0, 72.5, -7.5 },
        { 403, 1, 0, 70, -8 },     { 404, 2, 0, 67.5, 1 },
        { 405, 3, 0, 65, -8.5 },   { 406, 4, 0, 62.5, -10 },
        { 407, 5, 0, 60, -11 },    { 408, 6, 0, 55, -4 },
        { 409, 7, 0, 52.5, -4 },   { 410, 8, 0, 50, -4 },
        { 411, 10, 0, 45, -4 },    { 412, 7, 0, 42.5, -4 },
        { 413, 9, 0, 40, -4 } },
      /* PGT */
      { /* A10 */ { 2, 55, { 32, -1, 37, -1, 34, -1, 33, -1, 36, -1, 35, -1, 38,
                             -1, 41, -1, 44, -1, 45, -1, 47, -1, 50, -1, 51, 25,
                             53, 22, 56, 18, 57, 17, 59, 15, 60, 10, 61, 9,  62,
                             4,  63, 0,  -1, 3,  -1, 1,  -1, 6,  -1, 2,  -1, 5,
                             -1, 8,  -1, 7,  -1, 13, -1, 11, -1, 14, -1, 12, -1,
                             19, -1, 16, -1, 21, -1, 20, -1, 23, -1, 24, -1, 26,
                             -1, 27, -1, 28, -1, 29, -1, 30, -1, 31, -1, 39, -1,
                             40, -1, 42, -1, 43, -1, 46, -1, 48, -1, 49, -1, 52,
                             -1, 54, -1, 55, -1, 58 } },
        /* A11 */ { 2, 56, { 54, -1, 52, -1, 49, -1, 48, -1, 46, -1, 43, -1, 42,
                             -1, 40, -1, 39, -1, 32, -1, 37, -1, 34, -1, 33, -1,
                             36, -1, 35, -1, 38, -1, 41, -1, 44, -1, 45, -1, 47,
                             -1, 50, -1, 51, -1, 53, 25, 56, 22, 57, 18, 59, 17,
                             60, 15, 61, 10, 62, 9,  63, 4,  -1, 0,  -1, 3,  -1,
                             1,  -1, 6,  -1, 2,  -1, 5,  -1, 8,  -1, 7,  -1, 13,
                             -1, 11, -1, 14, -1, 12, -1, 19, -1, 16, -1, 21, -1,
                             20, -1, 23, -1, 24, -1, 26, -1, 27, -1, 28, -1, 29,
                             -1, 30, -1, 31, -1, 55, -1, 58 } },
        /* A12 */ { 2, 38, { 25, -1, 22, -1, 18, -1, 17, -1, 15, -1, 10, -1, 58,
                             -1, 55, -1, 54, -1, 52, -1, 49, -1, 48, -1, 46, 9,
                             43, 4,  42, 0,  40, 3,  39, 1,  32, 6,  37, 2,  34,
                             5,  33, 8,  36, 7,  35, 13, 38, 11, 41, 14, 44, 12,
                             45, 19, 47, 16, 50, 21, 51, 20, 53, 23, 56, 24, 57,
                             26, 59, 27, 60, 28, 61, 29, 62, 30, 63, 31 } },
        /* A13 */ { 2, 57, { -1, 14, -1, 11, -1, 13, -1, 7,  -1, 8,  -1, 5,  -1,
                             2,  -1, 6,  -1, 1,  -1, 3,  -1, 0,  -1, 4,  12, 9,
                             19, 10, 16, 15, 21, 17, 20, 18, 23, 22, 24, 25, 26,
                             -1, 27, -1, 28, -1, 29, -1, 30, -1, 31, -1, 58, -1,
                             55, -1, 54, -1, 52, -1, 49, -1, 48, -1, 46, -1, 43,
                             -1, 42, -1, 40, -1, 39, -1, 32, -1, 37, -1, 34, -1,
                             33, -1, 36, -1, 35, -1, 38, -1, 41, -1, 44, -1, 45,
                             -1, 47, -1, 50, -1, 51, -1, 53, -1, 56, -1, 57, -1,
                             59, -1, 60, -1, 61, -1, 62, -1, 63, -1 } },
        /* A14 */ { 2, 60, { -1, 33, -1, 34, -1, 37, -1, 32, -1, 39, -1, 40,
                             -1, 42, -1, 43, -1, 46, -1, 48, -1, 49, 36, 52,
                             35, 54, 38, 55, 41, 58, 44, -1, 45, -1, 47, -1,
                             50, -1, 51, -1, 53, -1, 56, -1, 57, -1, 59, -1,
                             60, -1, 61, -1, 62, -1, 63, -1, 25, -1, 22, -1,
                             18, -1, 17, -1, 15, -1, 10, -1, 9,  -1, 4,  -1,
                             0,  -1, 3,  -1, 1,  -1, 6,  -1, 2,  -1, 5,  -1,
                             8,  -1, 7,  -1, 13, -1, 11, -1, 14, -1, 12, -1,
                             19, -1, 16, -1, 21, -1, 20, -1, 23, -1, 24, -1,
                             26, -1, 27, -1, 28, -1, 29, -1, 30, -1, 31, -1 } },
        /* A15 */ { 2, 62, { -1, 2,  -1, 6,  -1, 1,  -1, 3,  -1, 0,  -1, 4,  -1,
                             9,  -1, 10, -1, 15, -1, 17, -1, 18, 5,  22, 8,  25,
                             7,  -1, 13, -1, 11, -1, 14, -1, 12, -1, 19, -1, 16,
                             -1, 21, -1, 20, -1, 23, -1, 24, -1, 26, -1, 27, -1,
                             28, -1, 29, -1, 30, -1, 31, -1, 63, -1, 62, -1, 61,
                             -1, 60, -1, 59, -1, 57, -1, 56, -1, 53, -1, 51, -1,
                             50, -1, 47, -1, 45, -1, 44, -1, 41, -1, 38, -1, 35,
                             -1, 36, -1, 33, -1, 34, -1, 37, -1, 32, -1, 39, -1,
                             40, -1, 42, -1, 43, -1, 46, -1, 48, -1, 49, -1, 52,
                             -1, 54, -1, 55, -1, 58, -1 } },
        /* A16 */ { 2, 48, { -1, 36, -1, 33, -1, 34, -1, 37, -1, 32, -1, 39,
                             -1, 40, -1, 42, -1, 43, -1, 46, -1, 48, -1, 49,
                             -1, 52, -1, 54, -1, 55, -1, 58, -1, 25, -1, 22,
                             -1, 18, -1, 17, -1, 15, -1, 10, -1, 9,  -1, 4,
                             -1, 0,  -1, 3,  -1, 1,  -1, 6,  -1, 2,  -1, 5,
                             -1, 8,  -1, 7,  63, 13, 62, 11, 61, 14, 60, 12,
                             59, 19, 57, 16, 56, 21, 53, 20, 51, 23, 50, 24,
                             47, 26, 45, 27, 44, 28, 41, 29, 38, 30, 35, 31 } },
        /* A17 */ { 2, 48, { -1, 31, -1, 30, -1, 29, -1, 28, -1, 27, -1, 26,
                             -1, 24, -1, 23, -1, 20, -1, 21, -1, 16, -1, 19,
                             -1, 12, -1, 14, -1, 11, -1, 13, 63, 7,  62, 8,
                             61, 5,  60, 2,  59, 6,  57, 1,  56, 3,  53, 0,
                             51, 4,  50, 9,  47, 10, 45, 15, 44, 17, 41, 18,
                             38, 22, 35, 25, 36, -1, 33, -1, 34, -1, 37, -1,
                             32, -1, 39, -1, 40, -1, 42, -1, 43, -1, 46, -1,
                             48, -1, 49, -1, 52, -1, 54, -1, 55, -1, 58, -1 } },
        /* A18 */ { 2, 48, { 13, 7,  11, 8,  14, 5,  12, 2,  19, 6,  16, 1,
                             21, 3,  20, 0,  23, 4,  24, 9,  26, 10, 27, 15,
                             28, 17, 29, 18, 30, 22, 31, 25, 63, -1, 62, -1,
                             61, -1, 60, -1, 59, -1, 57, -1, 56, -1, 53, -1,
                             51, -1, 50, -1, 47, -1, 45, -1, 44, -1, 41, -1,
                             38, -1, 35, -1, 36, -1, 33, -1, 34, -1, 37, -1,
                             32, -1, 39, -1, 40, -1, 42, -1, 43, -1, 46, -1,
                             48, -1, 49, -1, 52, -1, 54, -1, 55, -1, 58, -1 } },
        /* A19 */ { 2, 48, { 13, 7,  11, 8,  14, 5,  12, 2,  19, 6,  16, 1,
                             21, 3,  20, 0,  23, 4,  24, 9,  26, 10, 27, 15,
                             28, 17, 29, 18, 30, 22, 31, 25, 58, -1, 55, -1,
                             54, -1, 52, -1, 49, -1, 48, -1, 46, -1, 43, -1,
                             42, -1, 40, -1, 39, -1, 32, -1, 37, -1, 34, -1,
                             33, -1, 36, -1, 35, -1, 38, -1, 41, -1, 44, -1,
                             45, -1, 47, -1, 50, -1, 51, -1, 53, -1, 56, -1,
                             57, -1, 59, -1, 60, -1, 61, -1, 62, -1, 63, -1 } },
        /* A20 */ { 2, 48, { -1, 36, -1, 33, -1, 34, -1, 37, -1, 32, -1, 39,
                             -1, 40, -1, 42, -1, 43, -1, 46, -1, 48, -1, 49,
                             -1, 52, -1, 54, -1, 55, -1, 58, -1, 31, -1, 30,
                             -1, 29, -1, 28, -1, 27, -1, 26, -1, 24, -1, 23,
                             -1, 20, -1, 21, -1, 16, -1, 19, -1, 12, -1, 14,
                             -1, 11, -1, 13, 63, 7,  62, 8,  61, 5,  60, 2,
                             59, 6,  57, 1,  56, 3,  53, 0,  51, 4,  50, 9,
                             47, 10, 45, 15, 44, 17, 41, 18, 38, 22, 35, 25 } },
        /* A8 */ { 3, 27, { 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1, -1,
                            27, -1, -1, 26, 63, -1, 24, 62, -1, 23, 61, -1,
                            20, 60, -1, 21, 57, 59, 16, 53, 56, 19, 50, 51,
                            12, 45, 47, 14, 41, 44, 11, 35, 38, 13, 33, 36,
                            7,  37, 34, 8,  39, 32, 5,  42, 40, 2,  46, 43,
                            6,  49, 48, 1,  54, 52, 0,  3,  55, 9,  4,  58,
                            15, 10, -1, 18, 17, -1, 25, 22, -1 } },
        /* A9 */ { 2, 54, { 10, -1, 4,  25, 3,  22, 6,  18, 5,  17, 7,  15,
                            11, 9,  12, 0,  16, 1,  20, 2,  24, 8,  -1, 13,
                            -1, 14, -1, 19, -1, 21, -1, 23, -1, 26, -1, 27,
                            -1, 28, -1, 29, -1, 30, -1, 31, -1, 58, -1, 55,
                            -1, 54, -1, 52, -1, 49, -1, 48, -1, 46, -1, 43,
                            -1, 42, -1, 40, -1, 39, -1, 32, -1, 37, -1, 34,
                            -1, 33, -1, 36, -1, 35, -1, 38, -1, 41, -1, 44,
                            -1, 45, -1, 47, -1, 50, -1, 51, -1, 53, -1, 56,
                            -1, 57, -1, 59, -1, 60, -1, 61, -1, 62, -1, 63 } },
        /* L5 */ { 2, 40, { 23, 20, 24, 21, 26, 16, 27, 19, 28, 12, 29, 14,
                            30, 11, 31, 13, 58, 7,  55, 8,  54, 5,  52, 2,
                            49, 6,  48, 1,  46, 3,  43, 0,  42, 4,  40, 9,
                            39, 10, 32, 15, 37, 17, 34, 18, 33, 22, 36, 25,
                            35, -1, 38, -1, 41, -1, 44, -1, 45, -1, 47, -1,
                            50, -1, 51, -1, 53, -1, 56, -1, 57, -1, 59, -1,
                            60, -1, 61, -1, 62, -1, 63, -1 } },
        /* L6 */ { 2, 40, { 42, 43, 40, 46, 39, 48, 32, 49, 37, 52, 34, 54,
                            33, 55, 36, 58, 35, 31, 38, 30, 41, 29, 44, 28,
                            45, 27, 47, 26, 50, 24, 51, 23, 53, 20, 56, 21,
                            57, 16, 59, 19, 60, 12, 61, 14, 62, 11, 63, 13,
                            -1, 7,  -1, 8,  -1, 5,  -1, 2,  -1, 6,  -1, 1,
                            -1, 3,  -1, 0,  -1, 4,  -1, 9,  -1, 10, -1, 15,
                            -1, 17, -1, 18, -1, 22, -1, 25 } },
        /* L7 */ { 2, 40, { 25, -1, 22, -1, 18, -1, 17, -1, 15, -1, 10, -1,
                            9,  -1, 4,  -1, 0,  -1, 3,  -1, 1,  -1, 6,  -1,
                            2,  -1, 5,  -1, 8,  -1, 7,  -1, 13, 63, 11, 62,
                            14, 61, 12, 60, 19, 59, 16, 57, 21, 56, 20, 53,
                            23, 51, 24, 50, 26, 47, 27, 45, 28, 44, 29, 41,
                            30, 38, 31, 35, 58, 36, 55, 33, 54, 34, 52, 37,
                            49, 32, 48, 39, 46, 40, 43, 42 } },
        /* L8 */ { 2, 40, { -1, 63, -1, 62, -1, 61, -1, 60, -1, 59, -1, 57,
                            -1, 56, -1, 53, -1, 51, -1, 50, -1, 47, -1, 45,
                            -1, 44, -1, 41, -1, 38, -1, 35, 25, 36, 22, 33,
                            18, 34, 17, 37, 15, 32, 10, 39, 9,  40, 4,  42,
                            0,  43, 3,  46, 1,  48, 6,  49, 2,  52, 5,  54,
                            8,  55, 7,  58, 13, 31, 11, 30, 14, 29, 12, 28,
                            19, 27, 16, 26, 21, 24, 20, 23 } },
        /* O10 */ { 2, 32, { 31, 58, 30, 55, 29, 54, 28, 52, 27, 49, 26, 48, 24,
                             46, 23, 43, 20, 42, 21, 40, 16, 39, 19, 32, 12, 37,
                             14, 34, 11, 33, 13, 36, 7,  35, 8,  38, 5,  41, 2,
                             44, 6,  45, 1,  47, 3,  50, 0,  51, 4,  53, 9,  56,
                             10, 57, 15, 59, 17, 60, 18, 61, 22, 62, 25, 63 } },
        /* O23 */ { 2, 32, { 25, 63, 22, 62, 18, 61, 17, 60, 15, 59, 10, 57, 9,
                             56, 4,  53, 0,  51, 3,  50, 1,  47, 6,  45, 2,  44,
                             5,  41, 8,  38, 7,  35, 13, 36, 11, 33, 14, 34, 12,
                             37, 19, 32, 16, 39, 21, 40, 20, 42, 23, 43, 24, 46,
                             26, 48, 27, 49, 28, 52, 29, 54, 30, 55, 31, 58 } },
        /* O24 */ { 2, 32, { 31, 63, 30, 62, 29, 61, 28, 60, 27, 59, 26, 57, 24,
                             56, 23, 53, 20, 51, 21, 50, 16, 47, 19, 45, 12, 44,
                             14, 41, 11, 38, 13, 35, 7,  36, 8,  33, 5,  34, 2,
                             37, 6,  32, 1,  39, 3,  40, 0,  42, 4,  43, 9,  46,
                             10, 48, 15, 49, 17, 52, 18, 54, 22, 55, 25, 58 } },
        /* O9 */ { 2, 32, { 63, 25, 62, 22, 61, 18, 60, 17, 59, 15, 57, 10, 56,
                            9,  53, 4,  51, 0,  50, 3,  47, 1,  45, 6,  44, 2,
                            41, 5,  38, 8,  35, 7,  36, 13, 33, 11, 34, 14, 37,
                            12, 32, 19, 39, 16, 40, 21, 42, 20, 43, 23, 46, 24,
                            48, 26, 49, 27, 52, 28, 54, 29, 55, 30, 58, 31 } },
        /* Z1 */ { 3, 40, { -1, 0,  4,  -1, 3,  9,  -1, 1,  10, -1, 6,  15,
                            -1, 2,  17, -1, 5,  18, -1, 8,  22, -1, 7,  25,
                            -1, 13, -1, -1, 11, -1, -1, 14, -1, -1, 12, -1,
                            -1, 19, -1, -1, 16, -1, -1, 21, -1, -1, 20, -1,
                            -1, 23, -1, -1, 24, -1, -1, 26, -1, -1, 27, -1,
                            -1, 28, -1, -1, 29, -1, -1, 30, -1, -1, 31, -1,
                            63, 58, -1, 62, 55, -1, 61, 54, -1, 60, 52, -1,
                            59, 49, -1, 57, 48, -1, 56, 46, -1, 53, 43, -1,
                            51, 42, -1, 50, 40, -1, 47, 39, -1, 45, 32, -1,
                            44, 37, -1, 41, 34, -1, 38, 33, -1, 35, 36, -1 } },
        /* Z2 */ { 3, 40, { 53, 51, -1, 56, 50, -1, 57, 47, -1, 59, 45, -1,
                            60, 44, -1, 61, 41, -1, 62, 38, -1, 63, 35, -1,
                            -1, 36, -1, -1, 33, -1, -1, 34, -1, -1, 37, -1,
                            -1, 32, -1, -1, 39, -1, -1, 40, -1, -1, 42, -1,
                            -1, 43, -1, -1, 46, -1, -1, 48, -1, -1, 49, -1,
                            -1, 52, -1, -1, 54, -1, -1, 55, -1, -1, 58, -1,
                            -1, 31, 25, -1, 30, 22, -1, 29, 18, -1, 28, 17,
                            -1, 27, 15, -1, 26, 10, -1, 24, 9,  -1, 23, 4,
                            -1, 20, 0,  -1, 21, 3,  -1, 16, 1,  -1, 19, 6,
                            -1, 12, 2,  -1, 14, 5,  -1, 11, 8,  -1, 13, 7 } },
        /* Z3 */ { 3, 40, { 7,  13, -1, 8,  11, -1, 5,  14, -1, 2,  12, -1,
                            6,  19, -1, 1,  16, -1, 3,  21, -1, 0,  20, -1,
                            4,  23, -1, 9,  24, -1, 10, 26, -1, 15, 27, -1,
                            17, 28, -1, 18, 29, -1, 22, 30, -1, 25, 31, -1,
                            -1, 58, -1, -1, 55, -1, -1, 54, -1, -1, 52, -1,
                            -1, 49, -1, -1, 48, -1, -1, 46, -1, -1, 43, -1,
                            -1, 42, -1, -1, 40, -1, -1, 39, -1, -1, 32, -1,
                            -1, 37, -1, -1, 34, -1, -1, 33, -1, -1, 36, -1,
                            -1, 35, 63, -1, 38, 62, -1, 41, 61, -1, 44, 60,
                            -1, 45, 59, -1, 47, 57, -1, 50, 56, -1, 51, 53 } },
        /* Z4 */
        { 3,
          40,
          { -1, 36, 35, -1, 33, 38, -1, 34, 41, -1, 37, 44, -1, 32, 45,
            -1, 39, 47, -1, 40, 50, -1, 42, 51, -1, 43, 53, -1, 46, 56,
            -1, 48, 57, -1, 49, 59, -1, 52, 60, -1, 54, 61, -1, 55, 62,
            -1, 58, 63, -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1,
            -1, 27, -1, -1, 26, -1, -1, 24, -1, -1, 23, -1, -1, 20, -1,
            -1, 21, -1, -1, 16, -1, -1, 19, -1, -1, 12, -1, -1, 14, -1,
            -1, 11, -1, -1, 13, -1, 25, 7,  -1, 22, 8,  -1, 18, 5,  -1,
            17, 2,  -1, 15, 6,  -1, 10, 1,  -1, 9,  3,  -1, 4,  0,  -1 } } },
      /* PS */
      { { 2.5, 0.5 }, { 5, 0.5 } }
    };
  } else {
    return new CathodeSegmentation{
      8,
      false,
      /* PG */
      { { 1025, 0, 0, 51.42856979, -20 },
        { 1026, 11, 0, 45.7142868, -20 },
        { 1027, 11, 0, 40, -20 },
        { 1130, 15, 1, -50, -20 },
        { 1131, 17, 1, -60, -20 },
        { 1132, 16, 1, -71.42857361, -20 },
        { 1133, 14, 1, -80, -20 },
        { 1139, 15, 1, -10, -20 },
        { 1140, 17, 1, -20, -20 },
        { 1141, 16, 1, -31.4285717, -20 },
        { 1142, 14, 1, -40, -20 },
        { 1153, 10, 0, 34.2857132, -20 },
        { 1154, 10, 0, 28.5714283, -20 },
        { 1155, 10, 0, 22.8571434, -20 },
        { 1156, 10, 0, 17.1428566, -20 },
        { 1157, 10, 0, 11.4285717, -20 },
        { 1158, 10, 0, 5.714285851, -20 },
        { 1159, 10, 0, -3.996802889e-15, -20 },
        { 1225, 7, 1, -80, 0 },
        { 1226, 13, 1, -65.7142868, 0 },
        { 1227, 8, 1, -54.2857132, 0 },
        { 1233, 7, 1, -40, 0 },
        { 1234, 13, 1, -25.7142849, 0 },
        { 1235, 8, 1, -14.28571415, 0 },
        { 1241, 9, 0, -3.996802889e-15, 0 },
        { 1242, 9, 0, 5.714285851, 0 },
        { 1243, 9, 0, 11.4285717, 0 },
        { 1244, 9, 0, 17.1428566, 0 },
        { 1245, 9, 0, 22.8571434, 0 },
        { 1246, 9, 0, 28.5714283, 0 },
        { 1247, 9, 0, 34.2857132, 0 },
        { 1325, 12, 0, 40, 0 },
        { 1326, 12, 0, 45.7142868, 0 },
        { 1327, 12, 0, 51.42856979, 0 },
        { 1328, 1, 0, 57.1428566, -15 },
        { 1329, 2, 0, 60, -12.5 },
        { 1330, 3, 0, 63.57143021, -10 },
        { 1331, 4, 0, 67.14286041, -10 },
        { 1332, 5, 0, 71.42857361, -7.5 },
        { 1333, 6, 0, 75.7142868, -7.5 } },
      /* PGT */
      { /* A1 */ { 9, 8, { 53, 35, 42, 58, 23, 13, -1, -1, -1, 56, 38, 40,
                           55, 24, 11, 3,  18, 25, 57, 41, 39, 54, 26, 14,
                           1,  17, 22, 59, 44, 32, 52, 27, 12, 6,  15, -1,
                           60, 45, 37, 49, 28, 19, 2,  10, -1, 61, 47, 34,
                           48, 29, 16, 5,  9,  -1, 62, 50, 33, 46, 30, 21,
                           8,  4,  -1, 63, 51, 36, 43, 31, 20, 7,  0,  -1 } },
        /* A2 */ { 5, 14, { -1, 5,  27, 40, 51, 25, 8,  28, 39, 53, 22, 7,
                            29, 32, 56, 18, 13, 30, 37, 57, 17, 11, 31, 34,
                            59, 15, 14, 58, 33, 60, 10, 12, 55, 36, 61, 9,
                            19, 54, 35, 62, 4,  16, 52, 38, 63, 0,  21, 49,
                            41, -1, 3,  20, 48, 44, -1, 1,  23, 46, 45, -1,
                            6,  24, 43, 47, -1, 2,  26, 42, 50, -1 } },
        /* A3 */
        { 6, 13, { -1, 10, 14, 31, 37, 56, -1, 9,  12, 58, 34, 57, -1,
                   4,  19, 55, 33, 59, -1, 0,  16, 54, 36, 60, -1, 3,
                   21, 52, 35, 61, -1, 1,  20, 49, 38, 62, -1, 6,  23,
                   48, 41, 63, -1, 2,  24, 46, 44, -1, 25, 5,  26, 43,
                   45, -1, 22, 8,  27, 42, 47, -1, 18, 7,  28, 40, 50,
                   -1, 17, 13, 29, 39, 51, -1, 15, 11, 30, 32, 53, -1 } },
        /* A4 */ { 6, 12, { -1, 9,  14, 30, 39, 50, -1, 4,  12, 31, 32, 51,
                            -1, 0,  19, 58, 37, 53, -1, 3,  16, 55, 34, 56,
                            -1, 1,  21, 54, 33, 57, -1, 6,  20, 52, 36, 59,
                            25, 2,  23, 49, 35, 60, 22, 5,  24, 48, 38, 61,
                            18, 8,  26, 46, 41, 62, 17, 7,  27, 43, 44, 63,
                            15, 13, 28, 42, 45, -1, 10, 11, 29, 40, 47, -1 } },
        /* A5 */ { 7, 12, { -1, 18, 8,  26, -1, -1, -1, -1, 17, 7,  27, 46,
                            38, 60, -1, 15, 13, 28, 43, 41, 61, -1, 10, 11,
                            29, 42, 44, 62, -1, 9,  14, 30, 40, 45, 63, -1,
                            4,  12, 31, 39, 47, -1, -1, 0,  19, 58, 32, 50,
                            -1, -1, 3,  16, 55, 37, 51, -1, -1, 1,  21, 54,
                            34, 53, -1, -1, 6,  20, 52, 33, 56, -1, 25, 2,
                            23, 49, 36, 57, -1, 22, 5,  24, 48, 35, 59, -1 } },
        /* A6 */ { 7, 11, { -1, 4,  14, 29, 42, 44, 62, -1, 0,  12, 30, 40, 45,
                            63, -1, 3,  19, 31, 39, 47, -1, -1, 1,  16, 58, 32,
                            50, -1, 25, 6,  21, 55, 37, 51, -1, 22, 2,  20, 54,
                            34, 53, -1, 18, 5,  23, 52, 33, 56, -1, 17, 8,  24,
                            49, 36, 57, -1, 15, 7,  26, 48, 35, 59, -1, 10, 13,
                            27, 46, 38, 60, -1, 9,  11, 28, 43, 41, 61, -1 } },
        /* A7 */ { 6, 11, { -1, 3,  19, 31, 39, 47, -1, 1,  16, 58, 32,
                            50, 25, 6,  21, 55, 37, 51, 22, 2,  20, 54,
                            34, 53, 18, 5,  23, 52, 33, 56, 17, 8,  24,
                            49, 36, 57, 15, 7,  26, 48, 35, 59, 10, 13,
                            27, 46, 38, 60, 9,  11, 28, 43, 41, 61, 4,
                            14, 29, 42, 44, 62, 0,  12, 30, 40, 45, 63 } },
        /* L3 */ { 20, 4, { 17, 4,  6,  7,  -1, -1, -1, -1, -1, -1, -1, -1,
                            -1, -1, -1, -1, -1, -1, -1, -1, 18, 9,  1,  8,
                            14, 16, 23, 27, 30, 55, 49, 43, 39, 34, 35, 44,
                            50, 56, 60, 63, 22, 10, 3,  5,  11, 19, 20, 26,
                            29, 58, 52, 46, 40, 37, 36, 41, 47, 53, 59, 62,
                            25, 15, 0,  2,  13, 12, 21, 24, 28, 31, 54, 48,
                            42, 32, 33, 38, 45, 51, 57, 61 } },
        /* L4 */ { 20, 4, { -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                            -1, -1, -1, -1, 44, 51, 59, 63, 18, 10, 0,  6,
                            8,  11, 19, 20, 26, 29, 58, 52, 46, 40, 37, 36,
                            41, 50, 57, 62, 22, 15, 4,  1,  5,  13, 12, 21,
                            24, 28, 31, 54, 48, 42, 32, 33, 38, 47, 56, 61,
                            25, 17, 9,  3,  2,  7,  14, 16, 23, 27, 30, 55,
                            49, 43, 39, 34, 35, 45, 53, 60 } },
        /* O1 */ { 8, 8, { 4,  7,  20, 31, 43, 36, 51, 63, 9,  8,  21, 30, 46,
                           33, 50, 62, 10, 5,  16, 29, 48, 34, 47, 61, 15, 2,
                           19, 28, 49, 37, 45, 60, 17, 6,  12, 27, 52, 32, 44,
                           59, 18, 1,  14, 26, 54, 39, 41, 57, 22, 3,  11, 24,
                           55, 40, 38, 56, 25, 0,  13, 23, 58, 42, 35, 53 } },
        /* O2 */ { 8, 8, { 53, 35, 42, 58, 23, 13, 0,  25, 56, 38, 40, 55, 24,
                           11, 3,  22, 57, 41, 39, 54, 26, 14, 1,  18, 59, 44,
                           32, 52, 27, 12, 6,  17, 60, 45, 37, 49, 28, 19, 2,
                           15, 61, 47, 34, 48, 29, 16, 5,  10, 62, 50, 33, 46,
                           30, 21, 8,  9,  63, 51, 36, 43, 31, 20, 7,  4 } },
        /* O21 */ { 8, 8, { 53, 35, 42, 58, 23, 13, 0,  25, 56, 38, 40, 55, 24,
                            11, 3,  22, 57, 41, 39, 54, 26, 14, 1,  18, 59, 44,
                            32, 52, 27, 12, 6,  17, 60, 45, 37, 49, 28, 19, 2,
                            15, 61, 47, 34, 48, 29, 16, 5,  10, 62, 50, 33, 46,
                            30, 21, 8,  9,  63, 51, 36, 43, 31, 20, 7,  4 } },
        /* O22 */ { 8, 8, { 25, 0,  13, 23, 58, 42, 35, 53, 22, 3,  11, 24, 55,
                            40, 38, 56, 18, 1,  14, 26, 54, 39, 41, 57, 17, 6,
                            12, 27, 52, 32, 44, 59, 15, 2,  19, 28, 49, 37, 45,
                            60, 10, 5,  16, 29, 48, 34, 47, 61, 9,  8,  21, 30,
                            46, 33, 50, 62, 4,  7,  20, 31, 43, 36, 51, 63 } },
        /* O4 */ { 16, 4, { 17, 4,  6,  7,  12, 20, 27, 31, 52, 43, 32, 36, 44,
                            51, 59, 63, 18, 9,  1,  8,  14, 21, 26, 30, 54, 46,
                            39, 33, 41, 50, 57, 62, 22, 10, 3,  5,  11, 16, 24,
                            29, 55, 48, 40, 34, 38, 47, 56, 61, 25, 15, 0,  2,
                            13, 19, 23, 28, 58, 49, 42, 37, 35, 45, 53, 60 } },
        /* P3 */ { 14, 5, { 60, 53, 45, 35, 32, 46, 55, 28, 20, 14, 5,  0,
                            15, 25, 61, 56, 47, 38, 37, 43, 54, 29, 23, 12,
                            8,  3,  10, 22, 62, 57, 50, 41, 34, 42, 52, 30,
                            24, 19, 7,  1,  9,  18, 63, 59, 51, 44, 33, 40,
                            49, 31, 26, 16, 13, 6,  4,  17, -1, -1, -1, -1,
                            36, 39, 48, 58, 27, 21, 11, 2,  -1, -1 } },
        /* P4 */ { 14, 5, { 60, 53, 44, 33, 40, 49, 31, 26, 16, 13, 2,  0,
                            15, 25, 61, 56, 45, 36, 39, 48, 58, 27, 21, 11,
                            5,  3,  10, 22, 62, 57, 47, 35, 32, 46, 55, 28,
                            20, 14, 8,  1,  9,  18, 63, 59, 50, 38, 37, 43,
                            54, 29, 23, 12, 7,  6,  4,  17, -1, -1, 51, 41,
                            34, 42, 52, 30, 24, 19, -1, -1, -1, -1 } },
        /* Q3 */ { 16, 5, { -1, -1, 56, 45, 36, 39, 48, 58, 28, 23, 19, 13,
                            2,  0,  15, 25, -1, -1, 57, 47, 35, 32, 46, 55,
                            29, 24, 16, 11, 5,  3,  10, 22, -1, -1, 59, 50,
                            38, 37, 43, 54, 30, 26, 21, 14, 8,  1,  9,  18,
                            -1, -1, 60, 51, 41, 34, 42, 52, 31, 27, 20, 12,
                            7,  6,  4,  17, 63, 62, 61, 53, 44, 33, 40, 49,
                            -1, -1, -1, -1, -1, -1, -1, -1 } },
        /* Q4 */ { 16, 5, { 60, 53, 45, 35, 37, 42, 49, 58, 27, 21, 11, 2,
                            4,  18, -1, -1, 61, 56, 47, 38, 34, 40, 48, 55,
                            28, 20, 14, 5,  0,  17, -1, -1, 62, 57, 50, 41,
                            33, 39, 46, 54, 29, 23, 12, 8,  3,  15, -1, -1,
                            63, 59, 51, 44, 36, 32, 43, 52, 30, 24, 19, 7,
                            1,  10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                            31, 26, 16, 13, 6,  9,  22, 25 } } },
      /* PS */
      { { 0.714285714, 2.5 }, { 0.714285714, 5 } }
    };
  }
}
class CathodeSegmentationCreatorRegisterCreateSegType8
{
 public:
  CathodeSegmentationCreatorRegisterCreateSegType8()
  {
    registerCathodeSegmentationCreator(8, createSegType8);
  }
} aCathodeSegmentationCreatorRegisterCreateSegType8;

} // namespace impl4
} // namespace mapping
} // namespace mch
} // namespace o2

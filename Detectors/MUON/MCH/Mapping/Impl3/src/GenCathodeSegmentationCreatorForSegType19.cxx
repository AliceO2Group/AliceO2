// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See http://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
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
namespace impl3
{
CathodeSegmentation* createSegType19(bool isBendingPlane)
{
  if (isBendingPlane) {
    return new CathodeSegmentation{
      19,
      true,
      /* PG */
      {{1, 3, 0, -100, -20}, {2, 12, 0, -95, -20}, {3, 6, 0, -85, -20}, {4, 13, 0, -80, -20}, {5, 4, 0, -70, -20}, {10, 3, 0, -60, -20}, {11, 12, 0, -55, -20}, {12, 6, 0, -45, -20}, {13, 13, 0, -40, -20}, {14, 4, 0, -30, -20}, {103, 8, 1, 60, -20}, {104, 8, 1, 80, -20}, {107, 8, 1, 20, -20}, {108, 8, 1, 40, -20}, {111, 8, 1, -20, -20}, {112, 8, 1, 0, -20}, {201, 0, 1, 80, -4}, {202, 7, 1, 70, 4}, {203, 5, 1, 60, -4}, {206, 0, 1, 40, -4}, {207, 7, 1, 30, 4}, {208, 5, 1, 20, -4}, {211, 0, 1, 0, -4}, {212, 7, 1, -10, 4}, {213, 5, 1, -20, -4}, {304, 2, 0, -70, 0}, {305, 11, 0, -80, 0}, {306, 9, 0, -85, 4}, {307, 10, 0, -95, 0}, {308, 1, 0, -100, 0}, {312, 2, 0, -30, 0}, {313, 11, 0, -40, 0}, {314, 9, 0, -45, 4}, {315, 10, 0, -55, 0}, {316, 1, 0, -60, 0}},
      /* PGT */
      {/* L10 */ {2, 48, {15, 16, 14, 17, 13, 18, 12, 19, 11, 20, 10, 21, 9, 22, 8, 23, 7, 24, 6, 25, 5, 26, 4, 27, 3, 28, 2, 29, 1, 30, 0, 31, -1, 48, -1, 49, -1, 50, -1, 51, -1, 52, -1, 53, -1, 54, -1, 55, -1, 56, -1, 57, -1, 58, -1, 59, -1, 60, -1, 61, -1, 62, -1, 63, -1, 32, -1, 33, -1, 34, -1, 35, -1, 36, -1, 37, -1, 38, -1, 39, -1, 40, -1, 41, -1, 42, -1, 43, -1, 44, -1, 45, -1, 46, -1, 47}},
       /* L5 */ {2, 40, {55, 56, 54, 57, 53, 58, 52, 59, 51, 60, 50, 61, 49, 62, 48, 63, 31, 32, 30, 33, 29, 34, 28, 35, 27, 36, 26, 37, 25, 38, 24, 39, 23, 40, 22, 41, 21, 42, 20, 43, 19, 44, 18, 45, 17, 46, 16, 47, 15, -1, 14, -1, 13, -1, 12, -1, 11, -1, 10, -1, 9, -1, 8, -1, 7, -1, 6, -1, 5, -1, 4, -1, 3, -1, 2, -1, 1, -1, 0, -1}},
       /* L6 */ {2, 40, {23, 24, 22, 25, 21, 26, 20, 27, 19, 28, 18, 29, 17, 30, 16, 31, 15, 48, 14, 49, 13, 50, 12, 51, 11, 52, 10, 53, 9, 54, 8, 55, 7, 56, 6, 57, 5, 58, 4, 59, 3, 60, 2, 61, 1, 62, 0, 63, -1, 32, -1, 33, -1, 34, -1, 35, -1, 36, -1, 37, -1, 38, -1, 39, -1, 40, -1, 41, -1, 42, -1, 43, -1, 44, -1, 45, -1, 46, -1, 47}},
       /* L7 */ {2, 40, {47, -1, 46, -1, 45, -1, 44, -1, 43, -1, 42, -1, 41, -1, 40, -1, 39, -1, 38, -1, 37, -1, 36, -1, 35, -1, 34, -1, 33, -1, 32, -1, 63, 0, 62, 1, 61, 2, 60, 3, 59, 4, 58, 5, 57, 6, 56, 7, 55, 8, 54, 9, 53, 10, 52, 11, 51, 12, 50, 13, 49, 14, 48, 15, 31, 16, 30, 17, 29, 18, 28, 19, 27, 20, 26, 21, 25, 22, 24, 23}},
       /* L8 */ {2, 40, {-1, 0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1, 8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, 47, 16, 46, 17, 45, 18, 44, 19, 43, 20, 42, 21, 41, 22, 40, 23, 39, 24, 38, 25, 37, 26, 36, 27, 35, 28, 34, 29, 33, 30, 32, 31, 63, 48, 62, 49, 61, 50, 60, 51, 59, 52, 58, 53, 57, 54, 56, 55}},
       /* L9 */ {2, 48, {63, 32, 62, 33, 61, 34, 60, 35, 59, 36, 58, 37, 57, 38, 56, 39, 55, 40, 54, 41, 53, 42, 52, 43, 51, 44, 50, 45, 49, 46, 48, 47, 31, -1, 30, -1, 29, -1, 28, -1, 27, -1, 26, -1, 25, -1, 24, -1, 23, -1, 22, -1, 21, -1, 20, -1, 19, -1, 18, -1, 17, -1, 16, -1, 15, -1, 14, -1, 13, -1, 12, -1, 11, -1, 10, -1, 9, -1, 8, -1, 7, -1, 6, -1, 5, -1, 4, -1, 3, -1, 2, -1, 1, -1, 0, -1}},
       /* O10 */ {2, 32, {48, 31, 49, 30, 50, 29, 51, 28, 52, 27, 53, 26, 54, 25, 55, 24, 56, 23, 57, 22, 58, 21, 59, 20, 60, 19, 61, 18, 62, 17, 63, 16, 32, 15, 33, 14, 34, 13, 35, 12, 36, 11, 37, 10, 38, 9, 39, 8, 40, 7, 41, 6, 42, 5, 43, 4, 44, 3, 45, 2, 46, 1, 47, 0}},
       /* O11 */ {2, 32, {31, 48, 30, 49, 29, 50, 28, 51, 27, 52, 26, 53, 25, 54, 24, 55, 23, 56, 22, 57, 21, 58, 20, 59, 19, 60, 18, 61, 17, 62, 16, 63, 15, 32, 14, 33, 13, 34, 12, 35, 11, 36, 10, 37, 9, 38, 8, 39, 7, 40, 6, 41, 5, 42, 4, 43, 3, 44, 2, 45, 1, 46, 0, 47}},
       /* O12 */ {2, 32, {47, 0, 46, 1, 45, 2, 44, 3, 43, 4, 42, 5, 41, 6, 40, 7, 39, 8, 38, 9, 37, 10, 36, 11, 35, 12, 34, 13, 33, 14, 32, 15, 63, 16, 62, 17, 61, 18, 60, 19, 59, 20, 58, 21, 57, 22, 56, 23, 55, 24, 54, 25, 53, 26, 52, 27, 51, 28, 50, 29, 49, 30, 48, 31}},
       /* O9 */ {2, 32, {0, 47, 1, 46, 2, 45, 3, 44, 4, 43, 5, 42, 6, 41, 7, 40, 8, 39, 9, 38, 10, 37, 11, 36, 12, 35, 13, 34, 14, 33, 15, 32, 16, 63, 17, 62, 18, 61, 19, 60, 20, 59, 21, 58, 22, 57, 23, 56, 24, 55, 25, 54, 26, 53, 27, 52, 28, 51, 29, 50, 30, 49, 31, 48}},
       /* Z1 */ {3, 40, {-1, 39, 40, -1, 38, 41, -1, 37, 42, -1, 36, 43, -1, 35, 44, -1, 34, 45, -1, 33, 46, -1, 32, 47, -1, 63, -1, -1, 62, -1, -1, 61, -1, -1, 60, -1, -1, 59, -1, -1, 58, -1, -1, 57, -1, -1, 56, -1, -1, 55, -1, -1, 54, -1, -1, 53, -1, -1, 52, -1, -1, 51, -1, -1, 50, -1, -1, 49, -1, -1, 48, -1, 0, 31, -1, 1, 30, -1, 2, 29, -1, 3, 28, -1, 4, 27, -1, 5, 26, -1, 6, 25, -1, 7, 24, -1, 8, 23, -1, 9, 22, -1, 10, 21, -1, 11, 20, -1, 12, 19, -1, 13, 18, -1, 14, 17, -1, 15, 16, -1}},
       /* Z2 */ {3, 40, {7, 8, -1, 6, 9, -1, 5, 10, -1, 4, 11, -1, 3, 12, -1, 2, 13, -1, 1, 14, -1, 0, 15, -1, -1, 16, -1, -1, 17, -1, -1, 18, -1, -1, 19, -1, -1, 20, -1, -1, 21, -1, -1, 22, -1, -1, 23, -1, -1, 24, -1, -1, 25, -1, -1, 26, -1, -1, 27, -1, -1, 28, -1, -1, 29, -1, -1, 30, -1, -1, 31, -1, -1, 48, 47, -1, 49, 46, -1, 50, 45, -1, 51, 44, -1, 52, 43, -1, 53, 42, -1, 54, 41, -1, 55, 40, -1, 56, 39, -1, 57, 38, -1, 58, 37, -1, 59, 36, -1, 60, 35, -1, 61, 34, -1, 62, 33, -1, 63, 32}},
       /* Z3 */ {3, 40, {32, 63, -1, 33, 62, -1, 34, 61, -1, 35, 60, -1, 36, 59, -1, 37, 58, -1, 38, 57, -1, 39, 56, -1, 40, 55, -1, 41, 54, -1, 42, 53, -1, 43, 52, -1, 44, 51, -1, 45, 50, -1, 46, 49, -1, 47, 48, -1, -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1, -1, 27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1, -1, 23, -1, -1, 22, -1, -1, 21, -1, -1, 20, -1, -1, 19, -1, -1, 18, -1, -1, 17, -1, -1, 16, -1, -1, 15, 0, -1, 14, 1, -1, 13, 2, -1, 12, 3, -1, 11, 4, -1, 10, 5, -1, 9, 6, -1, 8, 7}},
       /* Z4 */ {3, 40, {-1, 16, 15, -1, 17, 14, -1, 18, 13, -1, 19, 12, -1, 20, 11, -1, 21, 10, -1, 22, 9, -1, 23, 8, -1, 24, 7, -1, 25, 6, -1, 26, 5, -1, 27, 4, -1, 28, 3, -1, 29, 2, -1, 30, 1, -1, 31, 0, -1, 48, -1, -1, 49, -1, -1, 50, -1, -1, 51, -1, -1, 52, -1, -1, 53, -1, -1, 54, -1, -1, 55, -1, -1, 56, -1, -1, 57, -1, -1, 58, -1, -1, 59, -1, -1, 60, -1, -1, 61, -1, -1, 62, -1, -1, 63, -1, 47, 32, -1, 46, 33, -1, 45, 34, -1, 44, 35, -1, 43, 36, -1, 42, 37, -1, 41, 38, -1, 40, 39, -1}}},
      /* PS */
      {{5, 0.5}, {10, 0.5}}};
  } else {
    return new CathodeSegmentation{
      19,
      false,
      /* PG */
      {{1030, 8, 0, -70, -20},
       {1031, 10, 0, -80, -20},
       {1032, 9, 0, -91.42857361, -20},
       {1033, 7, 0, -100, -20},
       {1039, 8, 0, -30, -20},
       {1040, 10, 0, -40, -20},
       {1041, 9, 0, -51.42856979, -20},
       {1042, 7, 0, -60, -20},
       {1125, 6, 1, 80, -20},
       {1126, 5, 1, 60, -20},
       {1129, 6, 1, 40, -20},
       {1130, 5, 1, 20, -20},
       {1133, 6, 1, -8.000000662e-09, -20},
       {1134, 5, 1, -20, -20},
       {1228, 3, 1, 60, 0},
       {1229, 4, 1, 80, 0},
       {1233, 3, 1, 20, 0},
       {1234, 4, 1, 40, 0},
       {1238, 3, 1, -20, 0},
       {1239, 4, 1, -8.000000662e-09, 0},
       {1325, 0, 0, -100, 0},
       {1326, 2, 0, -85.7142868, 0},
       {1327, 1, 0, -74.2857132, 0},
       {1333, 0, 0, -60, 0},
       {1334, 2, 0, -45.7142868, 0},
       {1335, 1, 0, -34.2857132, 0}},
      /* PGT */
      {/* L3 */ {20, 4, {44, 40, 36, 32, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 45, 41, 37, 33, 61, 58, 55, 52, 49, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, 46, 42, 38, 34, 62, 59, 56, 53, 50, 31, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 47, 43, 39, 35, 63, 60, 57, 54, 51, 48, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2}},
       /* L4 */ {20, 4, {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, 45, 42, 39, 36, 33, 62, 59, 56, 53, 50, 31, 28, 25, 22, 19, 16, 13, 9, 5, 1, 46, 43, 40, 37, 34, 63, 60, 57, 54, 51, 48, 29, 26, 23, 20, 17, 14, 10, 6, 2, 47, 44, 41, 38, 35, 32, 61, 58, 55, 52, 49, 30, 27, 24, 21, 18, 15, 11, 7, 3}},
       /* O4 */ {16, 4, {44, 40, 36, 32, 60, 56, 52, 48, 28, 24, 20, 16, 12, 8, 4, 0, 45, 41, 37, 33, 61, 57, 53, 49, 29, 25, 21, 17, 13, 9, 5, 1, 46, 42, 38, 34, 62, 58, 54, 50, 30, 26, 22, 18, 14, 10, 6, 2, 47, 43, 39, 35, 63, 59, 55, 51, 31, 27, 23, 19, 15, 11, 7, 3}},
       /* O5 */ {28, 2, {47, 45, 43, 41, 39, 37, 35, 33, 63, 61, 59, 57, 55, 53, 51, 49, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 46, 44, 42, 40, 38, 36, 34, 32, 62, 60, 58, 56, 54, 52, 50, 48, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8}},
       /* O6 */ {28, 2, {39, 37, 35, 33, 63, 61, 59, 57, 55, 53, 51, 49, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1, 38, 36, 34, 32, 62, 60, 58, 56, 54, 52, 50, 48, 30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0}},
       /* O7 */ {28, 2, {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 48, 50, 52, 54, 56, 58, 60, 62, 32, 34, 36, 38, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 49, 51, 53, 55, 57, 59, 61, 63, 33, 35, 37, 39}},
       /* O8 */ {28, 2, {8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 48, 50, 52, 54, 56, 58, 60, 62, 32, 34, 36, 38, 40, 42, 44, 46, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 49, 51, 53, 55, 57, 59, 61, 63, 33, 35, 37, 39, 41, 43, 45, 47}},
       /* P3 */ {14, 5, {3, 7, 11, 15, 20, 25, 30, 51, 56, 61, 34, 39, 43, 47, 2, 6, 10, 14, 19, 24, 29, 50, 55, 60, 33, 38, 42, 46, 1, 5, 9, 13, 18, 23, 28, 49, 54, 59, 32, 37, 41, 45, 0, 4, 8, 12, 17, 22, 27, 48, 53, 58, 63, 36, 40, 44, -1, -1, -1, -1, 16, 21, 26, 31, 52, 57, 62, 35, -1, -1}},
       /* P4 */ {14, 5, {3, 7, 12, 17, 22, 27, 48, 53, 58, 63, 35, 39, 43, 47, 2, 6, 11, 16, 21, 26, 31, 52, 57, 62, 34, 38, 42, 46, 1, 5, 10, 15, 20, 25, 30, 51, 56, 61, 33, 37, 41, 45, 0, 4, 9, 14, 19, 24, 29, 50, 55, 60, 32, 36, 40, 44, -1, -1, 8, 13, 18, 23, 28, 49, 54, 59, -1, -1, -1, -1}},
       /* Q3 */ {16, 5, {-1, -1, 6, 11, 16, 21, 26, 31, 51, 55, 59, 63, 35, 39, 43, 47, -1, -1, 5, 10, 15, 20, 25, 30, 50, 54, 58, 62, 34, 38, 42, 46, -1, -1, 4, 9, 14, 19, 24, 29, 49, 53, 57, 61, 33, 37, 41, 45, -1, -1, 3, 8, 13, 18, 23, 28, 48, 52, 56, 60, 32, 36, 40, 44, 0, 1, 2, 7, 12, 17, 22, 27, -1, -1, -1, -1, -1, -1, -1, -1}},
       /* Q4 */ {16, 5, {3, 7, 11, 15, 19, 23, 27, 31, 52, 57, 62, 35, 40, 45, -1, -1, 2, 6, 10, 14, 18, 22, 26, 30, 51, 56, 61, 34, 39, 44, -1, -1, 1, 5, 9, 13, 17, 21, 25, 29, 50, 55, 60, 33, 38, 43, -1, -1, 0, 4, 8, 12, 16, 20, 24, 28, 49, 54, 59, 32, 37, 42, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 48, 53, 58, 63, 36, 41, 46, 47}}},
      /* PS */
      {{0.714285714, 5}, {0.714285714, 10}}};
  }
}
class CathodeSegmentationCreatorRegisterCreateSegType19
{
 public:
  CathodeSegmentationCreatorRegisterCreateSegType19()
  {
    registerCathodeSegmentationCreator(19, createSegType19);
  }
} aCathodeSegmentationCreatorRegisterCreateSegType19;

} // namespace impl3
} // namespace mapping
} // namespace mch
} // namespace o2

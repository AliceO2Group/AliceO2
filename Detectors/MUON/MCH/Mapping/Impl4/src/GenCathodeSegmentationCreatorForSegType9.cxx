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
CathodeSegmentation* createSegType9(bool isBendingPlane)
{
  if (isBendingPlane) {
    return new CathodeSegmentation{
      9,
      true,
      /* PG */
      {{1, 2, 0, -80, -20}, {2, 8, 0, -77.5, -20}, {3, 4, 0, -72.5, -20}, {4, 9, 0, -70, -20}, {5, 3, 0, -65, -20}, {6, 2, 0, -60, -20}, {7, 8, 0, -57.5, -20}, {8, 4, 0, -52.5, -20}, {9, 9, 0, -50, -20}, {10, 3, 0, -45, -20}, {104, 2, 1, 40, -20}, {105, 8, 1, 45, -20}, {106, 4, 1, 55, -20}, {107, 9, 1, 60, -20}, {108, 3, 1, 70, -20}, {112, 2, 1, 0, -20}, {113, 8, 1, 5, -20}, {114, 4, 1, 15, -20}, {115, 9, 1, 20, -20}, {116, 3, 1, 30, -20}, {120, 2, 1, -40, -20}, {121, 8, 1, -35, -20}, {122, 4, 1, -25, -20}, {123, 9, 1, -20, -20}, {124, 3, 1, -10, -20}, {201, 1, 1, 70, 0}, {202, 7, 1, 60, 0}, {203, 5, 1, 55, 4}, {204, 6, 1, 45, 0}, {205, 0, 1, 40, 0}, {210, 1, 1, 30, 0}, {211, 7, 1, 20, 0}, {212, 5, 1, 15, 4}, {213, 6, 1, 5, 0}, {214, 0, 1, 0, 0}, {219, 1, 1, -10, 0}, {220, 7, 1, -20, 0}, {221, 5, 1, -25, 4}, {222, 6, 1, -35, 0}, {223, 0, 1, -40, 0}, {308, 1, 0, -45, 0}, {309, 7, 0, -50, 0}, {310, 5, 0, -52.5, 4}, {311, 6, 0, -57.5, 0}, {312, 0, 0, -60, 0}, {313, 1, 0, -65, 0}, {314, 7, 0, -70, 0}, {315, 5, 0, -72.5, 4}, {316, 6, 0, -77.5, 0}, {317, 0, 0, -80, 0}},
      /* PGT */
      {/* L5 */ {2, 40, {23, 20, 24, 21, 26, 16, 27, 19, 28, 12, 29, 14, 30, 11, 31, 13, 58, 7, 55, 8, 54, 5, 52, 2, 49, 6, 48, 1, 46, 3, 43, 0, 42, 4, 40, 9, 39, 10, 32, 15, 37, 17, 34, 18, 33, 22, 36, 25, 35, -1, 38, -1, 41, -1, 44, -1, 45, -1, 47, -1, 50, -1, 51, -1, 53, -1, 56, -1, 57, -1, 59, -1, 60, -1, 61, -1, 62, -1, 63, -1}},
       /* L6 */ {2, 40, {42, 43, 40, 46, 39, 48, 32, 49, 37, 52, 34, 54, 33, 55, 36, 58, 35, 31, 38, 30, 41, 29, 44, 28, 45, 27, 47, 26, 50, 24, 51, 23, 53, 20, 56, 21, 57, 16, 59, 19, 60, 12, 61, 14, 62, 11, 63, 13, -1, 7, -1, 8, -1, 5, -1, 2, -1, 6, -1, 1, -1, 3, -1, 0, -1, 4, -1, 9, -1, 10, -1, 15, -1, 17, -1, 18, -1, 22, -1, 25}},
       /* L7 */ {2, 40, {25, -1, 22, -1, 18, -1, 17, -1, 15, -1, 10, -1, 9, -1, 4, -1, 0, -1, 3, -1, 1, -1, 6, -1, 2, -1, 5, -1, 8, -1, 7, -1, 13, 63, 11, 62, 14, 61, 12, 60, 19, 59, 16, 57, 21, 56, 20, 53, 23, 51, 24, 50, 26, 47, 27, 45, 28, 44, 29, 41, 30, 38, 31, 35, 58, 36, 55, 33, 54, 34, 52, 37, 49, 32, 48, 39, 46, 40, 43, 42}},
       /* L8 */ {2, 40, {-1, 63, -1, 62, -1, 61, -1, 60, -1, 59, -1, 57, -1, 56, -1, 53, -1, 51, -1, 50, -1, 47, -1, 45, -1, 44, -1, 41, -1, 38, -1, 35, 25, 36, 22, 33, 18, 34, 17, 37, 15, 32, 10, 39, 9, 40, 4, 42, 0, 43, 3, 46, 1, 48, 6, 49, 2, 52, 5, 54, 8, 55, 7, 58, 13, 31, 11, 30, 14, 29, 12, 28, 19, 27, 16, 26, 21, 24, 20, 23}},
       /* O10 */ {2, 32, {31, 58, 30, 55, 29, 54, 28, 52, 27, 49, 26, 48, 24, 46, 23, 43, 20, 42, 21, 40, 16, 39, 19, 32, 12, 37, 14, 34, 11, 33, 13, 36, 7, 35, 8, 38, 5, 41, 2, 44, 6, 45, 1, 47, 3, 50, 0, 51, 4, 53, 9, 56, 10, 57, 15, 59, 17, 60, 18, 61, 22, 62, 25, 63}},
       /* O9 */ {2, 32, {63, 25, 62, 22, 61, 18, 60, 17, 59, 15, 57, 10, 56, 9, 53, 4, 51, 0, 50, 3, 47, 1, 45, 6, 44, 2, 41, 5, 38, 8, 35, 7, 36, 13, 33, 11, 34, 14, 37, 12, 32, 19, 39, 16, 40, 21, 42, 20, 43, 23, 46, 24, 48, 26, 49, 27, 52, 28, 54, 29, 55, 30, 58, 31}},
       /* Z1 */ {3, 40, {-1, 0, 4, -1, 3, 9, -1, 1, 10, -1, 6, 15, -1, 2, 17, -1, 5, 18, -1, 8, 22, -1, 7, 25, -1, 13, -1, -1, 11, -1, -1, 14, -1, -1, 12, -1, -1, 19, -1, -1, 16, -1, -1, 21, -1, -1, 20, -1, -1, 23, -1, -1, 24, -1, -1, 26, -1, -1, 27, -1, -1, 28, -1, -1, 29, -1, -1, 30, -1, -1, 31, -1, 63, 58, -1, 62, 55, -1, 61, 54, -1, 60, 52, -1, 59, 49, -1, 57, 48, -1, 56, 46, -1, 53, 43, -1, 51, 42, -1, 50, 40, -1, 47, 39, -1, 45, 32, -1, 44, 37, -1, 41, 34, -1, 38, 33, -1, 35, 36, -1}},
       /* Z2 */ {3, 40, {53, 51, -1, 56, 50, -1, 57, 47, -1, 59, 45, -1, 60, 44, -1, 61, 41, -1, 62, 38, -1, 63, 35, -1, -1, 36, -1, -1, 33, -1, -1, 34, -1, -1, 37, -1, -1, 32, -1, -1, 39, -1, -1, 40, -1, -1, 42, -1, -1, 43, -1, -1, 46, -1, -1, 48, -1, -1, 49, -1, -1, 52, -1, -1, 54, -1, -1, 55, -1, -1, 58, -1, -1, 31, 25, -1, 30, 22, -1, 29, 18, -1, 28, 17, -1, 27, 15, -1, 26, 10, -1, 24, 9, -1, 23, 4, -1, 20, 0, -1, 21, 3, -1, 16, 1, -1, 19, 6, -1, 12, 2, -1, 14, 5, -1, 11, 8, -1, 13, 7}},
       /* Z3 */ {3, 40, {7, 13, -1, 8, 11, -1, 5, 14, -1, 2, 12, -1, 6, 19, -1, 1, 16, -1, 3, 21, -1, 0, 20, -1, 4, 23, -1, 9, 24, -1, 10, 26, -1, 15, 27, -1, 17, 28, -1, 18, 29, -1, 22, 30, -1, 25, 31, -1, -1, 58, -1, -1, 55, -1, -1, 54, -1, -1, 52, -1, -1, 49, -1, -1, 48, -1, -1, 46, -1, -1, 43, -1, -1, 42, -1, -1, 40, -1, -1, 39, -1, -1, 32, -1, -1, 37, -1, -1, 34, -1, -1, 33, -1, -1, 36, -1, -1, 35, 63, -1, 38, 62, -1, 41, 61, -1, 44, 60, -1, 45, 59, -1, 47, 57, -1, 50, 56, -1, 51, 53}},
       /* Z4 */
       {3,
        40,
        {-1, 36, 35, -1, 33, 38, -1, 34, 41, -1, 37, 44, -1, 32, 45,
         -1, 39, 47, -1, 40, 50, -1, 42, 51, -1, 43, 53, -1, 46, 56,
         -1, 48, 57, -1, 49, 59, -1, 52, 60, -1, 54, 61, -1, 55, 62,
         -1, 58, 63, -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1,
         -1, 27, -1, -1, 26, -1, -1, 24, -1, -1, 23, -1, -1, 20, -1,
         -1, 21, -1, -1, 16, -1, -1, 19, -1, -1, 12, -1, -1, 14, -1,
         -1, 11, -1, -1, 13, -1, 25, 7, -1, 22, 8, -1, 18, 5, -1,
         17, 2, -1, 15, 6, -1, 10, 1, -1, 9, 3, -1, 4, 0, -1}}},
      /* PS */
      {{2.5, 0.5}, {5, 0.5}}};
  } else {
    return new CathodeSegmentation{
      9,
      false,
      /* PG */
      {{1035, 3, 0, -45.7142868, -20},
       {1036, 3, 0, -51.42856979, -20},
       {1037, 3, 0, -57.1428566, -20},
       {1038, 3, 0, -62.8571434, -20},
       {1039, 3, 0, -68.57142639, -20},
       {1040, 3, 0, -74.2857132, -20},
       {1041, 3, 0, -80, -20},
       {1125, 1, 1, 65.7142868, -20},
       {1126, 4, 1, 54.2857132, -20},
       {1127, 0, 1, 40, -20},
       {1133, 1, 1, 25.7142849, -20},
       {1134, 4, 1, 14.28571415, -20},
       {1135, 0, 1, 4.440892099e-15, -20},
       {1141, 1, 1, -14.28571415, -20},
       {1142, 4, 1, -25.7142849, -20},
       {1143, 0, 1, -40, -20},
       {1230, 7, 1, 40, -5},
       {1231, 5, 1, 48.57143021, -5},
       {1232, 6, 1, 60, -5},
       {1233, 8, 1, 70, -5},
       {1239, 7, 1, -7.105427358e-15, -5},
       {1240, 5, 1, 8.571428299, -5},
       {1241, 6, 1, 20, -5},
       {1242, 8, 1, 30, -5},
       {1248, 7, 1, -40, -5},
       {1249, 5, 1, -31.4285717, -5},
       {1250, 6, 1, -20, -5},
       {1251, 8, 1, -10, -5},
       {1325, 2, 0, -80, 0},
       {1326, 2, 0, -74.2857132, 0},
       {1327, 2, 0, -68.57142639, 0},
       {1328, 2, 0, -62.8571434, 0},
       {1329, 2, 0, -57.1428566, 0},
       {1330, 2, 0, -51.42856979, 0},
       {1331, 2, 0, -45.7142868, 0}},
      /* PGT */
      {/* L1 */ {20, 4, {60, 53, 45, 35, 34, 39, 43, 49, 55, 30, 27, 23, 16, 14, 7, 2, 3, 9, 17, 25, 61, 56, 47, 38, 33, 32, 42, 48, 54, 31, 28, 24, 21, 12, 13, 5, 1, 4, 15, 22, 62, 57, 50, 41, 36, 37, 40, 46, 52, 58, 29, 26, 20, 19, 11, 8, 6, 0, 10, 18, 63, 59, 51, 44, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}},
       /* L2 */ {20, 4, {61, 57, 51, 45, 38, 33, 32, 42, 48, 54, 31, 28, 24, 21, 12, 13, 2, 0, 15, 25, 62, 59, 53, 47, 41, 36, 37, 40, 46, 52, 58, 29, 26, 20, 19, 11, 5, 3, 10, 22, 63, 60, 56, 50, 44, 35, 34, 39, 43, 49, 55, 30, 27, 23, 16, 14, 8, 1, 9, 18, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 7, 6, 4, 17}},
       /* O1 */ {8, 8, {4, 7, 20, 31, 43, 36, 51, 63, 9, 8, 21, 30, 46, 33, 50, 62, 10, 5, 16, 29, 48, 34, 47, 61, 15, 2, 19, 28, 49, 37, 45, 60, 17, 6, 12, 27, 52, 32, 44, 59, 18, 1, 14, 26, 54, 39, 41, 57, 22, 3, 11, 24, 55, 40, 38, 56, 25, 0, 13, 23, 58, 42, 35, 53}},
       /* O2 */ {8, 8, {53, 35, 42, 58, 23, 13, 0, 25, 56, 38, 40, 55, 24, 11, 3, 22, 57, 41, 39, 54, 26, 14, 1, 18, 59, 44, 32, 52, 27, 12, 6, 17, 60, 45, 37, 49, 28, 19, 2, 15, 61, 47, 34, 48, 29, 16, 5, 10, 62, 50, 33, 46, 30, 21, 8, 9, 63, 51, 36, 43, 31, 20, 7, 4}},
       /* O3 */ {16, 4, {60, 53, 45, 35, 37, 42, 49, 58, 28, 23, 19, 13, 2, 0, 15, 25, 61, 56, 47, 38, 34, 40, 48, 55, 29, 24, 16, 11, 5, 3, 10, 22, 62, 57, 50, 41, 33, 39, 46, 54, 30, 26, 21, 14, 8, 1, 9, 18, 63, 59, 51, 44, 36, 32, 43, 52, 31, 27, 20, 12, 7, 6, 4, 17}},
       /* P1 */ {16, 5, {25, 22, 9, 6, 13, 16, 26, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 10, 1, 7, 19, 24, 30, 52, 43, 32, 36, 44, 51, 59, 63, -1, -1, 15, 3, 8, 12, 23, 29, 54, 46, 39, 33, 41, 50, 57, 62, -1, -1, 17, 0, 5, 14, 20, 28, 55, 48, 40, 34, 38, 47, 56, 61, -1, -1, 18, 4, 2, 11, 21, 27, 58, 49, 42, 37, 35, 45, 53, 60}},
       /* P2 */ {16, 5, {-1, -1, -1, -1, -1, -1, -1, -1, 49, 40, 33, 44, 53, 61, 62, 63, 17, 4, 6, 7, 12, 20, 27, 31, 52, 42, 34, 41, 51, 60, -1, -1, 18, 9, 1, 8, 14, 21, 26, 30, 54, 43, 37, 38, 50, 59, -1, -1, 22, 10, 3, 5, 11, 16, 24, 29, 55, 46, 32, 35, 47, 57, -1, -1, 25, 15, 0, 2, 13, 19, 23, 28, 58, 48, 39, 36, 45, 56, -1, -1}},
       /* Q1 */ {14, 5, {-1, -1, -1, -1, 19, 24, 30, 52, 42, 34, 41, 51, -1, -1, 17, 4, 6, 7, 12, 23, 29, 54, 43, 37, 38, 50, 59, 63, 18, 9, 1, 8, 14, 20, 28, 55, 46, 32, 35, 47, 57, 62, 22, 10, 3, 5, 11, 21, 27, 58, 48, 39, 36, 45, 56, 61, 25, 15, 0, 2, 13, 16, 26, 31, 49, 40, 33, 44, 53, 60}},
       /* Q2 */ {14, 5, {-1, -1, 2, 11, 21, 27, 58, 48, 39, 36, -1, -1, -1, -1, 17, 4, 6, 13, 16, 26, 31, 49, 40, 33, 44, 51, 59, 63, 18, 9, 1, 7, 19, 24, 30, 52, 42, 34, 41, 50, 57, 62, 22, 10, 3, 8, 12, 23, 29, 54, 43, 37, 38, 47, 56, 61, 25, 15, 0, 5, 14, 20, 28, 55, 46, 32, 35, 45, 53, 60}}},
      /* PS */
      {{0.714285714, 2.5}, {0.714285714, 5}}};
  }
}
class CathodeSegmentationCreatorRegisterCreateSegType9
{
 public:
  CathodeSegmentationCreatorRegisterCreateSegType9()
  {
    registerCathodeSegmentationCreator(9, createSegType9);
  }
} aCathodeSegmentationCreatorRegisterCreateSegType9;

} // namespace impl4
} // namespace mapping
} // namespace mch
} // namespace o2

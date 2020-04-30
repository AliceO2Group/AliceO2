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
CathodeSegmentation* createSegType18(bool isBendingPlane)
{
  if (isBendingPlane) {
    return new CathodeSegmentation{
      18,
      true,
      /* PG */
      {{1, 3, 0, -120, -20}, {2, 12, 0, -115, -20}, {3, 6, 0, -105, -20}, {4, 13, 0, -100, -20}, {5, 4, 0, -90, -20}, {10, 3, 0, -80, -20}, {11, 12, 0, -75, -20}, {12, 6, 0, -65, -20}, {13, 13, 0, -60, -20}, {14, 4, 0, -50, -20}, {19, 3, 0, -40, -20}, {20, 12, 0, -35, -20}, {21, 6, 0, -25, -20}, {22, 13, 0, -20, -20}, {23, 4, 0, -10, -20}, {103, 8, 1, 80, -20}, {104, 8, 1, 100, -20}, {107, 8, 1, 40, -20}, {108, 8, 1, 60, -20}, {111, 8, 1, 0, -20}, {112, 8, 1, 20, -20}, {201, 0, 1, 100, -4}, {202, 7, 1, 90, 4}, {203, 5, 1, 80, -4}, {206, 0, 1, 60, -4}, {207, 7, 1, 50, 4}, {208, 5, 1, 40, -4}, {211, 0, 1, 20, -4}, {212, 7, 1, 10, 4}, {213, 5, 1, 0, -4}, {304, 2, 0, -90, 0}, {305, 11, 0, -100, 0}, {306, 9, 0, -105, 4}, {307, 10, 0, -115, 0}, {308, 1, 0, -120, 0}, {312, 2, 0, -50, 0}, {313, 11, 0, -60, 0}, {314, 9, 0, -65, 4}, {315, 10, 0, -75, 0}, {316, 1, 0, -80, 0}, {320, 2, 0, -10, 0}, {321, 11, 0, -20, 0}, {322, 9, 0, -25, 4}, {323, 10, 0, -35, 0}, {324, 1, 0, -40, 0}},
      /* PGT */
      {/* L10 */ {2, 48, {35, 36, 38, 33, 41, 34, 44, 37, 45, 32, 47, 39, 50, 40, 51, 42, 53, 43, 56, 46, 57, 48, 59, 49, 60, 52, 61, 54, 62, 55, 63, 58, -1, 31, -1, 30, -1, 29, -1, 28, -1, 27, -1, 26, -1, 24, -1, 23, -1, 20, -1, 21, -1, 16, -1, 19, -1, 12, -1, 14, -1, 11, -1, 13, -1, 7, -1, 8, -1, 5, -1, 2, -1, 6, -1, 1, -1, 3, -1, 0, -1, 4, -1, 9, -1, 10, -1, 15, -1, 17, -1, 18, -1, 22, -1, 25}},
       /* L5 */ {2, 40, {23, 20, 24, 21, 26, 16, 27, 19, 28, 12, 29, 14, 30, 11, 31, 13, 58, 7, 55, 8, 54, 5, 52, 2, 49, 6, 48, 1, 46, 3, 43, 0, 42, 4, 40, 9, 39, 10, 32, 15, 37, 17, 34, 18, 33, 22, 36, 25, 35, -1, 38, -1, 41, -1, 44, -1, 45, -1, 47, -1, 50, -1, 51, -1, 53, -1, 56, -1, 57, -1, 59, -1, 60, -1, 61, -1, 62, -1, 63, -1}},
       /* L6 */ {2, 40, {42, 43, 40, 46, 39, 48, 32, 49, 37, 52, 34, 54, 33, 55, 36, 58, 35, 31, 38, 30, 41, 29, 44, 28, 45, 27, 47, 26, 50, 24, 51, 23, 53, 20, 56, 21, 57, 16, 59, 19, 60, 12, 61, 14, 62, 11, 63, 13, -1, 7, -1, 8, -1, 5, -1, 2, -1, 6, -1, 1, -1, 3, -1, 0, -1, 4, -1, 9, -1, 10, -1, 15, -1, 17, -1, 18, -1, 22, -1, 25}},
       /* L7 */ {2, 40, {25, -1, 22, -1, 18, -1, 17, -1, 15, -1, 10, -1, 9, -1, 4, -1, 0, -1, 3, -1, 1, -1, 6, -1, 2, -1, 5, -1, 8, -1, 7, -1, 13, 63, 11, 62, 14, 61, 12, 60, 19, 59, 16, 57, 21, 56, 20, 53, 23, 51, 24, 50, 26, 47, 27, 45, 28, 44, 29, 41, 30, 38, 31, 35, 58, 36, 55, 33, 54, 34, 52, 37, 49, 32, 48, 39, 46, 40, 43, 42}},
       /* L8 */ {2, 40, {-1, 63, -1, 62, -1, 61, -1, 60, -1, 59, -1, 57, -1, 56, -1, 53, -1, 51, -1, 50, -1, 47, -1, 45, -1, 44, -1, 41, -1, 38, -1, 35, 25, 36, 22, 33, 18, 34, 17, 37, 15, 32, 10, 39, 9, 40, 4, 42, 0, 43, 3, 46, 1, 48, 6, 49, 2, 52, 5, 54, 8, 55, 7, 58, 13, 31, 11, 30, 14, 29, 12, 28, 19, 27, 16, 26, 21, 24, 20, 23}},
       /* L9 */ {2, 48, {13, 7, 11, 8, 14, 5, 12, 2, 19, 6, 16, 1, 21, 3, 20, 0, 23, 4, 24, 9, 26, 10, 27, 15, 28, 17, 29, 18, 30, 22, 31, 25, 58, -1, 55, -1, 54, -1, 52, -1, 49, -1, 48, -1, 46, -1, 43, -1, 42, -1, 40, -1, 39, -1, 32, -1, 37, -1, 34, -1, 33, -1, 36, -1, 35, -1, 38, -1, 41, -1, 44, -1, 45, -1, 47, -1, 50, -1, 51, -1, 53, -1, 56, -1, 57, -1, 59, -1, 60, -1, 61, -1, 62, -1, 63, -1}},
       /* O10 */ {2, 32, {31, 58, 30, 55, 29, 54, 28, 52, 27, 49, 26, 48, 24, 46, 23, 43, 20, 42, 21, 40, 16, 39, 19, 32, 12, 37, 14, 34, 11, 33, 13, 36, 7, 35, 8, 38, 5, 41, 2, 44, 6, 45, 1, 47, 3, 50, 0, 51, 4, 53, 9, 56, 10, 57, 15, 59, 17, 60, 18, 61, 22, 62, 25, 63}},
       /* O11 */ {2, 32, {58, 31, 55, 30, 54, 29, 52, 28, 49, 27, 48, 26, 46, 24, 43, 23, 42, 20, 40, 21, 39, 16, 32, 19, 37, 12, 34, 14, 33, 11, 36, 13, 35, 7, 38, 8, 41, 5, 44, 2, 45, 6, 47, 1, 50, 3, 51, 0, 53, 4, 56, 9, 57, 10, 59, 15, 60, 17, 61, 18, 62, 22, 63, 25}},
       /* O12 */ {2, 32, {25, 63, 22, 62, 18, 61, 17, 60, 15, 59, 10, 57, 9, 56, 4, 53, 0, 51, 3, 50, 1, 47, 6, 45, 2, 44, 5, 41, 8, 38, 7, 35, 13, 36, 11, 33, 14, 34, 12, 37, 19, 32, 16, 39, 21, 40, 20, 42, 23, 43, 24, 46, 26, 48, 27, 49, 28, 52, 29, 54, 30, 55, 31, 58}},
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
      {{5, 0.5}, {10, 0.5}}};
  } else {
    return new CathodeSegmentation{
      18,
      false,
      /* PG */
      {{1030, 8, 0, -90, -20},
       {1031, 10, 0, -100, -20},
       {1032, 9, 0, -111.4285736, -20},
       {1033, 7, 0, -120, -20},
       {1039, 8, 0, -50, -20},
       {1040, 10, 0, -60, -20},
       {1041, 9, 0, -71.42857361, -20},
       {1042, 7, 0, -80, -20},
       {1048, 8, 0, -10, -20},
       {1049, 10, 0, -20, -20},
       {1050, 9, 0, -31.4285717, -20},
       {1051, 7, 0, -40, -20},
       {1125, 6, 1, 100, -20},
       {1126, 5, 1, 80, -20},
       {1129, 6, 1, 60, -20},
       {1130, 5, 1, 40, -20},
       {1133, 6, 1, 20, -20},
       {1134, 5, 1, -1.421085472e-14, -20},
       {1228, 3, 1, 80, 0},
       {1229, 4, 1, 100, 0},
       {1233, 3, 1, 40, 0},
       {1234, 4, 1, 60, 0},
       {1238, 3, 1, -1.421085472e-14, 0},
       {1239, 4, 1, 20, 0},
       {1325, 0, 0, -120, 0},
       {1326, 2, 0, -105.7142868, 0},
       {1327, 1, 0, -94.2857132, 0},
       {1333, 0, 0, -80, 0},
       {1334, 2, 0, -65.7142868, 0},
       {1335, 1, 0, -54.2857132, 0},
       {1341, 0, 0, -40, 0},
       {1342, 2, 0, -25.7142849, 0},
       {1343, 1, 0, -14.28571415, 0}},
      /* PGT */
      {/* L3 */ {20, 4, {17, 4, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 18, 9, 1, 8, 14, 16, 23, 27, 30, 55, 49, 43, 39, 34, 35, 44, 50, 56, 60, 63, 22, 10, 3, 5, 11, 19, 20, 26, 29, 58, 52, 46, 40, 37, 36, 41, 47, 53, 59, 62, 25, 15, 0, 2, 13, 12, 21, 24, 28, 31, 54, 48, 42, 32, 33, 38, 45, 51, 57, 61}},
       /* L4 */ {20, 4, {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 44, 51, 59, 63, 18, 10, 0, 6, 8, 11, 19, 20, 26, 29, 58, 52, 46, 40, 37, 36, 41, 50, 57, 62, 22, 15, 4, 1, 5, 13, 12, 21, 24, 28, 31, 54, 48, 42, 32, 33, 38, 47, 56, 61, 25, 17, 9, 3, 2, 7, 14, 16, 23, 27, 30, 55, 49, 43, 39, 34, 35, 45, 53, 60}},
       /* O4 */ {16, 4, {17, 4, 6, 7, 12, 20, 27, 31, 52, 43, 32, 36, 44, 51, 59, 63, 18, 9, 1, 8, 14, 21, 26, 30, 54, 46, 39, 33, 41, 50, 57, 62, 22, 10, 3, 5, 11, 16, 24, 29, 55, 48, 40, 34, 38, 47, 56, 61, 25, 15, 0, 2, 13, 19, 23, 28, 58, 49, 42, 37, 35, 45, 53, 60}},
       /* O5 */ {28, 2, {25, 18, 15, 9, 0, 1, 2, 8, 13, 14, 19, 21, 23, 26, 28, 30, 58, 54, 49, 46, 42, 39, 37, 33, 35, 41, 45, 50, 22, 17, 10, 4, 3, 6, 5, 7, 11, 12, 16, 20, 24, 27, 29, 31, 55, 52, 48, 43, 40, 32, 34, 36, 38, 44, 47, 51}},
       /* O6 */ {28, 2, {0, 1, 2, 8, 13, 14, 19, 21, 23, 26, 28, 30, 58, 54, 49, 46, 42, 39, 37, 33, 35, 41, 45, 50, 53, 57, 60, 62, 3, 6, 5, 7, 11, 12, 16, 20, 24, 27, 29, 31, 55, 52, 48, 43, 40, 32, 34, 36, 38, 44, 47, 51, 56, 59, 61, 63}},
       /* O7 */ {28, 2, {63, 61, 59, 56, 51, 47, 44, 38, 36, 34, 32, 40, 43, 48, 52, 55, 31, 29, 27, 24, 20, 16, 12, 11, 7, 5, 6, 3, 62, 60, 57, 53, 50, 45, 41, 35, 33, 37, 39, 42, 46, 49, 54, 58, 30, 28, 26, 23, 21, 19, 14, 13, 8, 2, 1, 0}},
       /* O8 */ {28, 2, {51, 47, 44, 38, 36, 34, 32, 40, 43, 48, 52, 55, 31, 29, 27, 24, 20, 16, 12, 11, 7, 5, 6, 3, 4, 10, 17, 22, 50, 45, 41, 35, 33, 37, 39, 42, 46, 49, 54, 58, 30, 28, 26, 23, 21, 19, 14, 13, 8, 2, 1, 0, 9, 15, 18, 25}},
       /* P3 */ {14, 5, {60, 53, 45, 35, 32, 46, 55, 28, 20, 14, 5, 0, 15, 25, 61, 56, 47, 38, 37, 43, 54, 29, 23, 12, 8, 3, 10, 22, 62, 57, 50, 41, 34, 42, 52, 30, 24, 19, 7, 1, 9, 18, 63, 59, 51, 44, 33, 40, 49, 31, 26, 16, 13, 6, 4, 17, -1, -1, -1, -1, 36, 39, 48, 58, 27, 21, 11, 2, -1, -1}},
       /* P4 */ {14, 5, {60, 53, 44, 33, 40, 49, 31, 26, 16, 13, 2, 0, 15, 25, 61, 56, 45, 36, 39, 48, 58, 27, 21, 11, 5, 3, 10, 22, 62, 57, 47, 35, 32, 46, 55, 28, 20, 14, 8, 1, 9, 18, 63, 59, 50, 38, 37, 43, 54, 29, 23, 12, 7, 6, 4, 17, -1, -1, 51, 41, 34, 42, 52, 30, 24, 19, -1, -1, -1, -1}},
       /* Q3 */ {16, 5, {-1, -1, 56, 45, 36, 39, 48, 58, 28, 23, 19, 13, 2, 0, 15, 25, -1, -1, 57, 47, 35, 32, 46, 55, 29, 24, 16, 11, 5, 3, 10, 22, -1, -1, 59, 50, 38, 37, 43, 54, 30, 26, 21, 14, 8, 1, 9, 18, -1, -1, 60, 51, 41, 34, 42, 52, 31, 27, 20, 12, 7, 6, 4, 17, 63, 62, 61, 53, 44, 33, 40, 49, -1, -1, -1, -1, -1, -1, -1, -1}},
       /* Q4 */ {16, 5, {60, 53, 45, 35, 37, 42, 49, 58, 27, 21, 11, 2, 4, 18, -1, -1, 61, 56, 47, 38, 34, 40, 48, 55, 28, 20, 14, 5, 0, 17, -1, -1, 62, 57, 50, 41, 33, 39, 46, 54, 29, 23, 12, 8, 3, 15, -1, -1, 63, 59, 51, 44, 36, 32, 43, 52, 30, 24, 19, 7, 1, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 31, 26, 16, 13, 6, 9, 22, 25}}},
      /* PS */
      {{0.714285714, 5}, {0.714285714, 10}}};
  }
}
class CathodeSegmentationCreatorRegisterCreateSegType18
{
 public:
  CathodeSegmentationCreatorRegisterCreateSegType18()
  {
    registerCathodeSegmentationCreator(18, createSegType18);
  }
} aCathodeSegmentationCreatorRegisterCreateSegType18;

} // namespace impl4
} // namespace mapping
} // namespace mch
} // namespace o2

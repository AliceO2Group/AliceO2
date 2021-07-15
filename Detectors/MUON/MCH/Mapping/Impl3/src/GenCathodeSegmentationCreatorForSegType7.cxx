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
namespace impl3
{
CathodeSegmentation* createSegType7(bool isBendingPlane)
{
  if (isBendingPlane) {
    return new CathodeSegmentation{
      7,
      true,
      /* PG */
      {{1, 0, 0, 50, 2}, {2, 10, 0, 45, 4}, {3, 10, 0, 40, 4}, {6, 6, 0, 35, 0}, {7, 13, 0, 30, 0}, {8, 11, 0, 27.5, 4}, {9, 12, 0, 22.5, 0}, {10, 5, 0, 20, 0}, {11, 6, 0, 15, 0}, {12, 13, 0, 10, 0}, {13, 11, 0, 7.5, 4}, {14, 12, 0, 2.5, 0}, {15, 5, 0, 0, 0}, {104, 6, 1, -50, 0}, {105, 13, 1, -60, 0}, {106, 11, 1, -65, 4}, {107, 12, 1, -75, 0}, {108, 5, 1, -80, 0}, {112, 6, 1, -10, 0}, {113, 13, 1, -20, 0}, {114, 11, 1, -25, 4}, {115, 12, 1, -35, 0}, {116, 5, 1, -40, 0}, {201, 7, 1, -80, -20}, {202, 14, 1, -75, -20}, {203, 9, 1, -65, -20}, {204, 15, 1, -60, -20}, {205, 8, 1, -50, -20}, {210, 7, 1, -40, -20}, {211, 14, 1, -35, -20}, {212, 9, 1, -25, -20}, {213, 15, 1, -20, -20}, {214, 8, 1, -10, -20}, {304, 1, 0, 40, -20}, {305, 2, 0, 42.5, -20}, {306, 3, 0, 45, -20}, {307, 4, 0, 50, -20}, {315, 7, 0, 0, -20}, {316, 14, 0, 2.5, -20}, {317, 9, 0, 7.5, -20}, {318, 15, 0, 10, -20}, {319, 8, 0, 15, -20}, {320, 7, 0, 20, -20}, {321, 14, 0, 22.5, -20}, {322, 9, 0, 27.5, -20}, {323, 15, 0, 30, -20}, {324, 8, 0, 35, -20}},
      /* PGT */
      {/* C10 */ {3, 36, {51, -1, -1, 50, -1, -1, 49, -1, -1, 48, -1, -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1, -1, 27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1, -1, 23, -1, -1, 22, -1, -1, 21, -1, -1, 20, 40, -1, 19, 39, -1, 18, 38, -1, 17, 37, -1, 16, 36, -1, 15, 35, -1, 14, 34, -1, 13, 33, -1, 12, 32, -1, 11, 63, -1, 10, 62, -1, 9, 61, -1, 8, 60, -1, 7, 59, -1, 6, 58, 47, 5, 57, 46, 4, 56, 45, 3, 55, 44, 2, 54, 43, 1, 53, 42, 0, 52, 41}},
       /* C6 */ {2, 48, {47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41, 9, 40, 8, 39, 7, 38, 6, 37, 5, 36, 4, 35, 3, 34, 2, 33, 1, 32, 0, 63, -1, 62, -1, 61, -1, 60, -1, 59, -1, 58, -1, 57, -1, 56, -1, 55, -1, 54, -1, 53, -1, 52, -1, 51, -1, 50, -1, 49, -1, 48, -1, 31, -1, 30, -1, 29, -1, 28, -1, 27, -1, 26, -1, 25, -1, 24, -1, 23, -1, 22, -1, 21, -1, 20, -1, 19, -1, 18, -1, 17, -1, 16, -1}},
       /* C7 */ {2, 48, {-1, 31, -1, 30, -1, 29, -1, 28, -1, 27, -1, 26, -1, 25, -1, 24, -1, 23, -1, 22, -1, 21, -1, 20, -1, 19, -1, 18, -1, 17, -1, 16, 47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41, 9, 40, 8, 39, 7, 38, 6, 37, 5, 36, 4, 35, 3, 34, 2, 33, 1, 32, 0, 63, -1, 62, -1, 61, -1, 60, -1, 59, -1, 58, -1, 57, -1, 56, -1, 55, -1, 54, -1, 53, -1, 52, -1, 51, -1, 50, -1, 49, -1, 48, -1}},
       /* C8 */ {2, 48, {-1, 63, -1, 62, -1, 61, -1, 60, -1, 59, -1, 58, -1, 57, -1, 56, -1, 55, -1, 54, -1, 53, -1, 52, -1, 51, -1, 50, -1, 49, -1, 48, -1, 31, -1, 30, -1, 29, -1, 28, -1, 27, -1, 26, -1, 25, -1, 24, -1, 23, -1, 22, -1, 21, -1, 20, -1, 19, -1, 18, -1, 17, -1, 16, 47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41, 9, 40, 8, 39, 7, 38, 6, 37, 5, 36, 4, 35, 3, 34, 2, 33, 1, 32, 0}},
       /* C9 */ {3, 36, {47, 27, 6, 46, 26, 5, 45, 25, 4, 44, 24, 3, 43, 23, 2, 42, 22, 1, 41, 21, 0, 40, 20, -1, 39, 19, -1, 38, 18, -1, 37, 17, -1, 36, 16, -1, 35, 15, -1, 34, 14, -1, 33, 13, -1, 32, 12, -1, 63, 11, -1, 62, 10, -1, 61, 9, -1, 60, 8, -1, 59, 7, -1, 58, -1, -1, 57, -1, -1, 56, -1, -1, 55, -1, -1, 54, -1, -1, 53, -1, -1, 52, -1, -1, 51, -1, -1, 50, -1, -1, 49, -1, -1, 48, -1, -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1, -1}},
       /* L5 */ {2, 40, {55, 56, 54, 57, 53, 58, 52, 59, 51, 60, 50, 61, 49, 62, 48, 63, 31, 32, 30, 33, 29, 34, 28, 35, 27, 36, 26, 37, 25, 38, 24, 39, 23, 40, 22, 41, 21, 42, 20, 43, 19, 44, 18, 45, 17, 46, 16, 47, 15, -1, 14, -1, 13, -1, 12, -1, 11, -1, 10, -1, 9, -1, 8, -1, 7, -1, 6, -1, 5, -1, 4, -1, 3, -1, 2, -1, 1, -1, 0, -1}},
       /* L6 */ {2, 40, {23, 24, 22, 25, 21, 26, 20, 27, 19, 28, 18, 29, 17, 30, 16, 31, 15, 48, 14, 49, 13, 50, 12, 51, 11, 52, 10, 53, 9, 54, 8, 55, 7, 56, 6, 57, 5, 58, 4, 59, 3, 60, 2, 61, 1, 62, 0, 63, -1, 32, -1, 33, -1, 34, -1, 35, -1, 36, -1, 37, -1, 38, -1, 39, -1, 40, -1, 41, -1, 42, -1, 43, -1, 44, -1, 45, -1, 46, -1, 47}},
       /* L7 */ {2, 40, {47, -1, 46, -1, 45, -1, 44, -1, 43, -1, 42, -1, 41, -1, 40, -1, 39, -1, 38, -1, 37, -1, 36, -1, 35, -1, 34, -1, 33, -1, 32, -1, 63, 0, 62, 1, 61, 2, 60, 3, 59, 4, 58, 5, 57, 6, 56, 7, 55, 8, 54, 9, 53, 10, 52, 11, 51, 12, 50, 13, 49, 14, 48, 15, 31, 16, 30, 17, 29, 18, 28, 19, 27, 20, 26, 21, 25, 22, 24, 23}},
       /* L8 */ {2, 40, {-1, 0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1, 8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, 47, 16, 46, 17, 45, 18, 44, 19, 43, 20, 42, 21, 41, 22, 40, 23, 39, 24, 38, 25, 37, 26, 36, 27, 35, 28, 34, 29, 33, 30, 32, 31, 63, 48, 62, 49, 61, 50, 60, 51, 59, 52, 58, 53, 57, 54, 56, 55}},
       /* O10 */ {2, 32, {48, 31, 49, 30, 50, 29, 51, 28, 52, 27, 53, 26, 54, 25, 55, 24, 56, 23, 57, 22, 58, 21, 59, 20, 60, 19, 61, 18, 62, 17, 63, 16, 32, 15, 33, 14, 34, 13, 35, 12, 36, 11, 37, 10, 38, 9, 39, 8, 40, 7, 41, 6, 42, 5, 43, 4, 44, 3, 45, 2, 46, 1, 47, 0}},
       /* O25 */ {2, 32, {31, 47, 30, 46, 29, 45, 28, 44, 27, 43, 26, 42, 25, 41, 24, 40, 23, 39, 22, 38, 21, 37, 20, 36, 19, 35, 18, 34, 17, 33, 16, 32, 15, 63, 14, 62, 13, 61, 12, 60, 11, 59, 10, 58, 9, 57, 8, 56, 7, 55, 6, 54, 5, 53, 4, 52, 3, 51, 2, 50, 1, 49, 0, 48}},
       /* O9 */ {2, 32, {0, 47, 1, 46, 2, 45, 3, 44, 4, 43, 5, 42, 6, 41, 7, 40, 8, 39, 9, 38, 10, 37, 11, 36, 12, 35, 13, 34, 14, 33, 15, 32, 16, 63, 17, 62, 18, 61, 19, 60, 20, 59, 21, 58, 22, 57, 23, 56, 24, 55, 25, 54, 26, 53, 27, 52, 28, 51, 29, 50, 30, 49, 31, 48}},
       /* Z1 */ {3, 40, {-1, 39, 40, -1, 38, 41, -1, 37, 42, -1, 36, 43, -1, 35, 44, -1, 34, 45, -1, 33, 46, -1, 32, 47, -1, 63, -1, -1, 62, -1, -1, 61, -1, -1, 60, -1, -1, 59, -1, -1, 58, -1, -1, 57, -1, -1, 56, -1, -1, 55, -1, -1, 54, -1, -1, 53, -1, -1, 52, -1, -1, 51, -1, -1, 50, -1, -1, 49, -1, -1, 48, -1, 0, 31, -1, 1, 30, -1, 2, 29, -1, 3, 28, -1, 4, 27, -1, 5, 26, -1, 6, 25, -1, 7, 24, -1, 8, 23, -1, 9, 22, -1, 10, 21, -1, 11, 20, -1, 12, 19, -1, 13, 18, -1, 14, 17, -1, 15, 16, -1}},
       /* Z2 */ {3, 40, {7, 8, -1, 6, 9, -1, 5, 10, -1, 4, 11, -1, 3, 12, -1, 2, 13, -1, 1, 14, -1, 0, 15, -1, -1, 16, -1, -1, 17, -1, -1, 18, -1, -1, 19, -1, -1, 20, -1, -1, 21, -1, -1, 22, -1, -1, 23, -1, -1, 24, -1, -1, 25, -1, -1, 26, -1, -1, 27, -1, -1, 28, -1, -1, 29, -1, -1, 30, -1, -1, 31, -1, -1, 48, 47, -1, 49, 46, -1, 50, 45, -1, 51, 44, -1, 52, 43, -1, 53, 42, -1, 54, 41, -1, 55, 40, -1, 56, 39, -1, 57, 38, -1, 58, 37, -1, 59, 36, -1, 60, 35, -1, 61, 34, -1, 62, 33, -1, 63, 32}},
       /* Z3 */ {3, 40, {32, 63, -1, 33, 62, -1, 34, 61, -1, 35, 60, -1, 36, 59, -1, 37, 58, -1, 38, 57, -1, 39, 56, -1, 40, 55, -1, 41, 54, -1, 42, 53, -1, 43, 52, -1, 44, 51, -1, 45, 50, -1, 46, 49, -1, 47, 48, -1, -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1, -1, 27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1, -1, 23, -1, -1, 22, -1, -1, 21, -1, -1, 20, -1, -1, 19, -1, -1, 18, -1, -1, 17, -1, -1, 16, -1, -1, 15, 0, -1, 14, 1, -1, 13, 2, -1, 12, 3, -1, 11, 4, -1, 10, 5, -1, 9, 6, -1, 8, 7}},
       /* Z4 */ {3, 40, {-1, 16, 15, -1, 17, 14, -1, 18, 13, -1, 19, 12, -1, 20, 11, -1, 21, 10, -1, 22, 9, -1, 23, 8, -1, 24, 7, -1, 25, 6, -1, 26, 5, -1, 27, 4, -1, 28, 3, -1, 29, 2, -1, 30, 1, -1, 31, 0, -1, 48, -1, -1, 49, -1, -1, 50, -1, -1, 51, -1, -1, 52, -1, -1, 53, -1, -1, 54, -1, -1, 55, -1, -1, 56, -1, -1, 57, -1, -1, 58, -1, -1, 59, -1, -1, 60, -1, -1, 61, -1, -1, 62, -1, -1, 63, -1, 47, 32, -1, 46, 33, -1, 45, 34, -1, 44, 35, -1, 43, 36, -1, 42, 37, -1, 41, 38, -1, 40, 39, -1}}},
      /* PS */
      {{2.5, 0.5}, {5, 0.5}}};
  } else {
    return new CathodeSegmentation{
      7,
      false,
      /* PG */
      {{1028, 4, 0, 40, 2.5},
       {1029, 3, 0, 47.1428566, 5},
       {1040, 7, 0, -3.996802889e-15, 0},
       {1041, 7, 0, 5.714285851, 0},
       {1042, 7, 0, 11.4285717, 0},
       {1043, 7, 0, 17.1428566, 0},
       {1044, 7, 0, 22.8571434, 0},
       {1045, 7, 0, 28.5714283, 0},
       {1046, 7, 0, 34.2857132, 0},
       {1125, 5, 1, -80, 0},
       {1126, 9, 1, -65.7142868, 0},
       {1127, 6, 1, -54.2857132, 0},
       {1133, 5, 1, -40, 0},
       {1134, 9, 1, -25.7142849, 0},
       {1135, 6, 1, -14.28571415, 0},
       {1230, 11, 1, -50, -20},
       {1231, 13, 1, -60, -20},
       {1232, 12, 1, -71.42857361, -20},
       {1233, 10, 1, -80, -20},
       {1239, 11, 1, -10, -20},
       {1240, 13, 1, -20, -20},
       {1241, 12, 1, -31.4285717, -20},
       {1242, 10, 1, -40, -20},
       {1325, 2, 0, 49.2857132, -20},
       {1326, 1, 0, 44.2857132, -20},
       {1327, 0, 0, 40, -20},
       {1332, 8, 0, 34.2857132, -20},
       {1333, 8, 0, 28.5714283, -20},
       {1334, 8, 0, 22.8571434, -20},
       {1335, 8, 0, 17.1428566, -20},
       {1336, 8, 0, 11.4285717, -20},
       {1337, 8, 0, 5.714285851, -20},
       {1338, 8, 0, -3.996802889e-15, -20}},
      /* PGT */
      {/* C1 */ {7, 10, {8, 17, 27, 53, 63, 41, -1, 7, 16, 26, 52, 62, 40, -1, 6, 15, 25, 51, 61, 39, -1, 5, 14, 24, 50, 60, 38, -1, 4, 13, 23, 49, 59, 37, 47, 3, 12, 22, 48, 58, 36, 46, 2, 11, 21, 31, 57, 35, 45, 1, 10, 20, 30, 56, 34, 44, 0, 9, 19, 29, 55, 33, 43, -1, -1, 18, 28, 54, 32, 42}},
       /* C2 */ {7, 10, {3, 13, 23, 49, 59, 37, 47, 2, 12, 22, 48, 58, 36, 46, 1, 11, 21, 31, 57, 35, 45, 0, 10, 20, 30, 56, 34, 44, -1, 9, 19, 29, 55, 33, 43, -1, 8, 18, 28, 54, 32, 42, -1, 7, 17, 27, 53, 63, 41, -1, 6, 16, 26, 52, 62, 40, -1, 5, 15, 25, 51, 61, 39, -1, 4, 14, 24, 50, 60, 38}},
       /* C3 */ {13, 10, {9, 19, 29, 55, 61, 33, 37, 40, 42, 44, 45, 46, 47, 8, 18, 28, 54, 60, 32, 36, 39, 41, 43, -1, -1, -1, 7, 17, 27, 53, 59, 63, 35, 38, -1, -1, -1, -1, -1, 6, 16, 26, 52, 58, 62, 34, -1, -1, -1, -1, -1, -1, 5, 15, 25, 51, 57, -1, -1, -1, -1, -1, -1, -1, -1, 4, 14, 24, 50, 56, -1, -1, -1, -1, -1, -1, -1, -1, 3, 13, 23, 49, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 12, 22, 48, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 11, 21, 31, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 10, 20, 30, -1, -1, -1, -1, -1, -1, -1, -1, -1}},
       /* C4 */ {16, 6, {-1, 43, 37, 63, 57, 51, 29, 23, -1, -1, -1, -1, -1, -1, -1, -1, -1, 42, 36, 62, 56, 50, 28, 22, -1, -1, -1, -1, -1, -1, -1, -1, 47, 41, 35, 61, 55, 49, 27, 21, 17, 13, -1, -1, -1, -1, -1, -1, 46, 40, 34, 60, 54, 48, 26, 20, 16, 12, 9, -1, -1, -1, -1, -1, 45, 39, 33, 59, 53, 31, 25, 19, 15, 11, 8, 6, 4, -1, -1, -1, 44, 38, 32, 58, 52, 30, 24, 18, 14, 10, 7, 5, 3, 2, 1, 0}},
       /* C5 */ {11, 7, {47, 40, -1, -1, -1, -1, -1, -1, -1, -1, -1, 46, 39, 33, 59, 53, 31, 25, 19, 13, 7, 1, 45, 38, 32, 58, 52, 30, 24, 18, 12, 6, 0, 44, 37, 63, 57, 51, 29, 23, 17, 11, 5, -1, 43, 36, 62, 56, 50, 28, 22, 16, 10, 4, -1, 42, 35, 61, 55, 49, 27, 21, 15, 9, 3, -1, 41, 34, 60, 54, 48, 26, 20, 14, 8, 2, -1}},
       /* L3 */ {20, 4, {44, 40, 36, 32, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 45, 41, 37, 33, 61, 58, 55, 52, 49, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0, 46, 42, 38, 34, 62, 59, 56, 53, 50, 31, 28, 25, 22, 19, 16, 13, 10, 7, 4, 1, 47, 43, 39, 35, 63, 60, 57, 54, 51, 48, 29, 26, 23, 20, 17, 14, 11, 8, 5, 2}},
       /* L4 */ {20, 4, {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 8, 4, 0, 45, 42, 39, 36, 33, 62, 59, 56, 53, 50, 31, 28, 25, 22, 19, 16, 13, 9, 5, 1, 46, 43, 40, 37, 34, 63, 60, 57, 54, 51, 48, 29, 26, 23, 20, 17, 14, 10, 6, 2, 47, 44, 41, 38, 35, 32, 61, 58, 55, 52, 49, 30, 27, 24, 21, 18, 15, 11, 7, 3}},
       /* O1 */ {8, 8, {40, 32, 56, 48, 24, 16, 8, 0, 41, 33, 57, 49, 25, 17, 9, 1, 42, 34, 58, 50, 26, 18, 10, 2, 43, 35, 59, 51, 27, 19, 11, 3, 44, 36, 60, 52, 28, 20, 12, 4, 45, 37, 61, 53, 29, 21, 13, 5, 46, 38, 62, 54, 30, 22, 14, 6, 47, 39, 63, 55, 31, 23, 15, 7}},
       /* O2 */ {8, 8, {7, 15, 23, 31, 55, 63, 39, 47, 6, 14, 22, 30, 54, 62, 38, 46, 5, 13, 21, 29, 53, 61, 37, 45, 4, 12, 20, 28, 52, 60, 36, 44, 3, 11, 19, 27, 51, 59, 35, 43, 2, 10, 18, 26, 50, 58, 34, 42, 1, 9, 17, 25, 49, 57, 33, 41, 0, 8, 16, 24, 48, 56, 32, 40}},
       /* O4 */ {16, 4, {44, 40, 36, 32, 60, 56, 52, 48, 28, 24, 20, 16, 12, 8, 4, 0, 45, 41, 37, 33, 61, 57, 53, 49, 29, 25, 21, 17, 13, 9, 5, 1, 46, 42, 38, 34, 62, 58, 54, 50, 30, 26, 22, 18, 14, 10, 6, 2, 47, 43, 39, 35, 63, 59, 55, 51, 31, 27, 23, 19, 15, 11, 7, 3}},
       /* P3 */ {14, 5, {3, 7, 11, 15, 20, 25, 30, 51, 56, 61, 34, 39, 43, 47, 2, 6, 10, 14, 19, 24, 29, 50, 55, 60, 33, 38, 42, 46, 1, 5, 9, 13, 18, 23, 28, 49, 54, 59, 32, 37, 41, 45, 0, 4, 8, 12, 17, 22, 27, 48, 53, 58, 63, 36, 40, 44, -1, -1, -1, -1, 16, 21, 26, 31, 52, 57, 62, 35, -1, -1}},
       /* P4 */ {14, 5, {3, 7, 12, 17, 22, 27, 48, 53, 58, 63, 35, 39, 43, 47, 2, 6, 11, 16, 21, 26, 31, 52, 57, 62, 34, 38, 42, 46, 1, 5, 10, 15, 20, 25, 30, 51, 56, 61, 33, 37, 41, 45, 0, 4, 9, 14, 19, 24, 29, 50, 55, 60, 32, 36, 40, 44, -1, -1, 8, 13, 18, 23, 28, 49, 54, 59, -1, -1, -1, -1}},
       /* Q3 */ {16, 5, {-1, -1, 6, 11, 16, 21, 26, 31, 51, 55, 59, 63, 35, 39, 43, 47, -1, -1, 5, 10, 15, 20, 25, 30, 50, 54, 58, 62, 34, 38, 42, 46, -1, -1, 4, 9, 14, 19, 24, 29, 49, 53, 57, 61, 33, 37, 41, 45, -1, -1, 3, 8, 13, 18, 23, 28, 48, 52, 56, 60, 32, 36, 40, 44, 0, 1, 2, 7, 12, 17, 22, 27, -1, -1, -1, -1, -1, -1, -1, -1}},
       /* Q4 */ {16, 5, {3, 7, 11, 15, 19, 23, 27, 31, 52, 57, 62, 35, 40, 45, -1, -1, 2, 6, 10, 14, 18, 22, 26, 30, 51, 56, 61, 34, 39, 44, -1, -1, 1, 5, 9, 13, 17, 21, 25, 29, 50, 55, 60, 33, 38, 43, -1, -1, 0, 4, 8, 12, 16, 20, 24, 28, 49, 54, 59, 32, 37, 42, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 48, 53, 58, 63, 36, 41, 46, 47}}},
      /* PS */
      {{0.714285714, 2.5}, {0.714285714, 5}}};
  }
}
class CathodeSegmentationCreatorRegisterCreateSegType7
{
 public:
  CathodeSegmentationCreatorRegisterCreateSegType7()
  {
    registerCathodeSegmentationCreator(7, createSegType7);
  }
} aCathodeSegmentationCreatorRegisterCreateSegType7;

} // namespace impl3
} // namespace mapping
} // namespace mch
} // namespace o2

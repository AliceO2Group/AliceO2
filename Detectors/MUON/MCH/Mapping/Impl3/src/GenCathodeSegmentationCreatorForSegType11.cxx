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

namespace o2 {
namespace mch {
namespace mapping {
namespace impl3 {
CathodeSegmentation *createSegType11(bool isBendingPlane) {
  if (isBendingPlane) {
    return new CathodeSegmentation{
        11,
        true,
        /* PG */
        {{1, 10, 0, 80, -20},      {9, 17, 0, 40, -20},
         {10, 25, 0, 42.5, -20},   {11, 19, 0, 47.5, -20},
         {12, 26, 0, 50, -20},     {13, 18, 0, 55, -20},
         {14, 17, 0, 60, -20},     {15, 25, 0, 62.5, -20},
         {16, 19, 0, 67.5, -20},   {17, 26, 0, 70, -20},
         {18, 18, 0, 75, -20},     {22, 17, 1, 0, -20},
         {23, 25, 1, 5, -20},      {24, 19, 1, 15, -20},
         {25, 26, 1, 20, -20},     {26, 18, 1, 30, -20},
         {101, 13, 2, -120, -20},  {102, 21, 2, -110, -20},
         {103, 14, 2, -100, -20},  {106, 13, 2, -80, -20},
         {107, 21, 2, -70, -20},   {108, 14, 2, -60, -20},
         {111, 17, 1, -40, -20},   {112, 25, 1, -35, -20},
         {113, 19, 1, -25, -20},   {114, 26, 1, -20, -20},
         {115, 18, 1, -10, -20},   {203, 20, 2, -100, 4},
         {204, 20, 2, -120, 4},    {207, 20, 2, -60, 4},
         {208, 20, 2, -80, 4},     {212, 16, 1, -10, 0},
         {213, 24, 1, -20, 0},     {214, 22, 1, -25, 4},
         {215, 23, 1, -35, 0},     {216, 15, 1, -40, 0},
         {311, 16, 0, 75, 0},      {312, 24, 0, 70, 0},
         {313, 22, 0, 67.5, 4},    {314, 23, 0, 62.5, 0},
         {315, 15, 0, 60, 0},      {316, 16, 0, 55, 0},
         {317, 24, 0, 50, 0},      {318, 22, 0, 47.5, 4},
         {319, 23, 0, 42.5, 0},    {320, 15, 0, 40, 0},
         {328, 16, 1, 30, 0},      {329, 24, 1, 20, 0},
         {330, 22, 1, 15, 4},      {331, 23, 1, 5, 0},
         {332, 15, 1, 0, 0},       {401, 11, 0, 112.5, 0.5},
         {402, 12, 0, 110, 0},     {403, 0, 0, 105, -1.5},
         {404, 1, 0, 102.5, -2.5}, {405, 2, 0, 100, -3.5},
         {406, 3, 0, 95, -6.5},    {407, 4, 0, 95, -5},
         {407, 27, 0, 95, -8.5},   {407, 28, 0, 92.5, -9.5},
         {408, 5, 0, 92.5, -8.5},  {408, 29, 0, 90, -11},
         {409, 6, 0, 90, -10},     {410, 7, 0, 87.5, -12},
         {411, 7, 0, 85, -12},     {412, 8, 0, 82.5, -11.5},
         {413, 9, 0, 80, -11.5}},
        /* PGT */
        {/* E10 */ {
             4, 43, {-1, 18, -1, -1, -1, 19, -1, -1, -1, 20, 45, -1, -1, 21, -1,
                     -1, -1, 22, -1, -1, -1, 23, -1, -1, -1, 26, -1, -1, -1, 27,
                     -1, -1, -1, 28, -1, -1, -1, 29, -1, -1, -1, 59, -1, -1, -1,
                     60, -1, -1, -1, 61, -1, -1, 0,  62, -1, -1, 1,  63, -1, -1,
                     2,  32, -1, -1, 3,  33, -1, -1, 7,  34, -1, -1, 8,  38, -1,
                     -1, 9,  39, -1, -1, 10, 40, -1, -1, 11, 41, -1, -1, 12, 44,
                     -1, -1, 17, 43, -1, -1, 16, 42, -1, -1, 13, 35, -1, -1, 4,
                     30, -1, -1, -1, 24, 46, -1, -1, 14, 36, -1, -1, 5,  31, -1,
                     -1, -1, 25, 47, -1, -1, 15, 37, -1, -1, 6,  58, -1, -1, -1,
                     57, -1, -1, -1, 56, -1, -1, -1, 55, -1, -1, -1, 54, -1, -1,
                     -1, 53, -1, -1, -1, 52, -1, -1, -1, 51, -1, -1, -1, 50, -1,
                     -1, -1, 49, -1, -1, -1, 48}},
         /* E11 */ {4, 45, {-1, 60, -1, -1, -1, 61, -1, -1, -1, 62, -1, -1, -1,
                            63, -1, -1, -1, 32, -1, -1, -1, 35, -1, -1, -1, 36,
                            -1, -1, -1, 37, -1, -1, -1, 38, -1, -1, -1, 39, -1,
                            -1, 8,  40, -1, -1, 9,  41, -1, -1, 13, 45, -1, -1,
                            14, 46, -1, -1, 15, 47, -1, -1, 16, -1, -1, -1, 17,
                            -1, -1, -1, 18, -1, -1, -1, 19, -1, -1, -1, 20, -1,
                            -1, -1, 24, -1, -1, -1, 25, -1, -1, -1, 26, -1, -1,
                            -1, 27, -1, -1, -1, 28, -1, -1, -1, 57, -1, -1, -1,
                            58, -1, -1, -1, 59, -1, -1, -1, 29, -1, -1, -1, 21,
                            42, -1, -1, 10, 33, -1, -1, -1, 30, -1, -1, -1, 22,
                            43, -1, -1, 11, 34, -1, -1, -1, 31, -1, -1, -1, 23,
                            44, -1, -1, 12, 56, -1, -1, 7,  55, -1, -1, 6,  54,
                            -1, -1, 5,  53, -1, -1, 4,  52, -1, -1, 3,  51, -1,
                            -1, 2,  50, -1, -1, 1,  49, -1, -1, 0,  48}},
         /* E12 */ {3, 47, {6,  34, -1, 7,  35, -1, 8,  36, -1, 9,  37, -1, 10,
                            38, -1, 13, 39, -1, 14, 42, -1, 15, 43, -1, 16, 44,
                            -1, 17, 45, -1, 18, 46, -1, 19, 47, -1, 23, -1, -1,
                            24, -1, -1, 25, -1, -1, 26, -1, -1, 27, -1, -1, 28,
                            -1, -1, 29, -1, -1, 49, -1, -1, 50, -1, -1, 54, -1,
                            -1, 55, -1, -1, 56, -1, -1, 57, -1, -1, 58, -1, -1,
                            59, -1, -1, 33, -1, -1, 32, -1, -1, 61, -1, -1, 60,
                            -1, -1, 51, -1, -1, 30, -1, -1, 20, 40, -1, 11, 62,
                            -1, -1, 52, -1, -1, 31, 41, -1, 21, 63, -1, 12, 53,
                            -1, -1, 48, -1, -1, 22, -1, -1, 5,  -1, -1, 4,  -1,
                            -1, 3,  -1, -1, 2,  -1, -1, 1,  -1, -1, 0}},
         /* E13 */ {4, 53, {-1, 3,  -1, -1, 2,  4,  -1, -1, 1,  5,  -1, -1, -1,
                            6,  43, -1, -1, 9,  46, -1, -1, 10, 47, -1, -1, 11,
                            -1, -1, -1, 12, -1, -1, -1, 13, -1, -1, -1, 14, -1,
                            -1, -1, 15, -1, -1, -1, 18, -1, -1, -1, 42, -1, -1,
                            -1, 41, -1, -1, -1, 40, -1, -1, -1, 39, -1, -1, -1,
                            38, -1, -1, -1, 37, -1, -1, -1, 34, -1, -1, -1, 33,
                            -1, -1, -1, 32, -1, -1, -1, 63, -1, -1, -1, 62, -1,
                            -1, -1, 61, -1, -1, -1, 58, -1, -1, -1, 57, -1, -1,
                            -1, 56, -1, -1, -1, 55, -1, -1, -1, 54, -1, -1, -1,
                            53, -1, -1, -1, 30, -1, -1, -1, 29, -1, -1, -1, 28,
                            -1, -1, -1, 27, -1, -1, -1, 26, -1, -1, -1, 24, -1,
                            -1, -1, 23, -1, -1, -1, 22, -1, -1, -1, 21, -1, -1,
                            -1, 20, -1, -1, -1, 19, -1, -1, -1, 16, 44, -1, -1,
                            7,  35, -1, -1, -1, 59, -1, -1, -1, 31, -1, -1, -1,
                            25, 45, -1, -1, 17, 36, -1, -1, 8,  60, -1, -1, 0,
                            52, -1, -1, -1, 51, -1, -1, -1, 50, -1, -1, -1, 49,
                            -1, -1, -1, 48}},
         /* E14 */ {3, 50, {46, -1, -1, 45, -1, -1, 44, -1, -1, 43, -1, -1, 42,
                            -1, -1, 41, -1, -1, 38, -1, -1, 37, -1, -1, 36, -1,
                            -1, 35, -1, -1, 34, -1, -1, 33, -1, -1, 63, -1, -1,
                            62, -1, -1, 61, -1, -1, 60, -1, -1, 59, -1, -1, 58,
                            -1, -1, 56, -1, -1, 55, -1, -1, 54, -1, -1, 53, -1,
                            -1, 52, -1, -1, 51, -1, -1, 30, -1, -1, 28, -1, -1,
                            27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1, -1, 23,
                            -1, -1, 22, -1, -1, 20, -1, -1, 19, -1, -1, 18, -1,
                            -1, 17, -1, -1, 16, -1, -1, 15, -1, -1, 14, -1, -1,
                            13, -1, -1, 12, 47, -1, 11, 39, -1, 7,  32, -1, -1,
                            57, -1, -1, 31, -1, -1, 29, -1, -1, 21, 40, -1, 10,
                            50, -1, 9,  49, -1, 8,  48}},
         /* E15 */ {2, 56, {47, -1, 45, -1, 44, -1, 43, -1, 42, -1, 41, -1, 40,
                            -1, 39, -1, 37, -1, 36, -1, 35, -1, 34, -1, 33, -1,
                            32, -1, 62, -1, 61, -1, 60, -1, 59, -1, 58, -1, 57,
                            -1, 55, -1, 54, -1, 53, -1, 52, -1, 51, -1, 50, -1,
                            48, -1, 31, -1, 30, -1, 29, -1, 28, -1, 27, -1, 25,
                            -1, 24, -1, 23, -1, 22, -1, 21, -1, 20, -1, 19, -1,
                            18, -1, 17, -1, 16, -1, 15, -1, 14, -1, 13, -1, 12,
                            -1, 11, -1, 10, -1, 9,  -1, 8,  -1, 7,  46, 6,  38,
                            5,  63, 4,  56, -1, 49, -1, 26}},
         /* E16 */ {3, 60, {47, -1, -1, 44, -1, -1, 43, -1, -1, 42, -1, -1, 41,
                            -1, -1, 40, -1, -1, 39, -1, -1, 37, -1, -1, 36, -1,
                            -1, 35, -1, -1, 34, -1, -1, 33, -1, -1, 32, -1, -1,
                            62, -1, -1, 61, -1, -1, 60, -1, -1, 59, -1, -1, 58,
                            -1, -1, 57, -1, -1, 56, -1, -1, 55, -1, -1, 54, -1,
                            -1, 53, -1, -1, 52, -1, -1, 51, -1, -1, 50, -1, -1,
                            49, -1, -1, 48, -1, -1, 31, -1, -1, 30, -1, -1, 29,
                            -1, -1, 28, -1, -1, 27, -1, -1, 26, -1, -1, 25, -1,
                            -1, 24, -1, -1, 23, -1, -1, 22, -1, -1, 21, -1, -1,
                            20, -1, -1, 19, -1, -1, 18, -1, -1, 17, -1, -1, 16,
                            -1, -1, 15, -1, -1, 14, -1, -1, 13, -1, -1, 12, -1,
                            -1, 11, -1, -1, 10, -1, -1, 9,  -1, -1, 8,  -1, -1,
                            7,  -1, -1, 6,  -1, -1, 5,  -1, -1, 4,  -1, -1, 3,
                            -1, -1, 2,  45, -1, 1,  38, -1, 0,  63, 46}},
         /* E17 */ {2, 64, {-1, 45, -1, 46, -1, 47, 44, -1, 43, -1, 42, -1, 41,
                            -1, 40, -1, 39, -1, 38, -1, 37, -1, 36, -1, 35, -1,
                            34, -1, 33, -1, 32, -1, 63, -1, 62, -1, 61, -1, 60,
                            -1, 59, -1, 58, -1, 57, -1, 56, -1, 55, -1, 54, -1,
                            53, -1, 52, -1, 51, -1, 50, -1, 49, -1, 48, -1, 31,
                            -1, 30, -1, 29, -1, 28, -1, 27, -1, 26, -1, 25, -1,
                            24, -1, 23, -1, 22, -1, 21, -1, 20, -1, 19, -1, 18,
                            -1, 17, -1, 16, -1, 15, -1, 14, -1, 13, -1, 12, -1,
                            11, -1, 10, -1, 9,  -1, 8,  -1, 7,  -1, 6,  -1, 5,
                            -1, 4,  -1, 3,  -1, 2,  -1, 1,  -1, 0,  -1}},
         /* E18 */ {2, 63, {-1, 46, 45, 47, 44, -1, 43, -1, 42, -1, 41, -1, 40,
                            -1, 39, -1, 38, -1, 37, -1, 36, -1, 35, -1, 34, -1,
                            33, -1, 32, -1, 63, -1, 62, -1, 61, -1, 60, -1, 59,
                            -1, 58, -1, 57, -1, 56, -1, 55, -1, 54, -1, 53, -1,
                            52, -1, 51, -1, 50, -1, 49, -1, 48, -1, 31, -1, 30,
                            -1, 29, -1, 28, -1, 27, -1, 26, -1, 25, -1, 24, -1,
                            23, -1, 22, -1, 21, -1, 20, -1, 19, -1, 18, -1, 17,
                            -1, 16, -1, 15, -1, 14, -1, 13, -1, 12, -1, 11, -1,
                            10, -1, 9,  -1, 8,  -1, 7,  -1, 6,  -1, 5,  -1, 4,
                            -1, 3,  -1, 2,  -1, 1,  -1, 0,  -1}},
         /* E19 */ {2, 63, {46, 47, 45, -1, 44, -1, 43, -1, 42, -1, 41, -1, 40,
                            -1, 39, -1, 38, -1, 37, -1, 36, -1, 35, -1, 34, -1,
                            33, -1, 32, -1, 63, -1, 62, -1, 61, -1, 60, -1, 59,
                            -1, 58, -1, 57, -1, 56, -1, 55, -1, 54, -1, 53, -1,
                            52, -1, 51, -1, 50, -1, 49, -1, 48, -1, 31, -1, 30,
                            -1, 29, -1, 28, -1, 27, -1, 26, -1, 25, -1, 24, -1,
                            23, -1, 22, -1, 21, -1, 20, -1, 19, -1, 18, -1, 17,
                            -1, 16, -1, 15, -1, 14, -1, 13, -1, 12, -1, 11, -1,
                            10, -1, 9,  -1, 8,  -1, 7,  -1, 6,  -1, 5,  -1, 4,
                            -1, 3,  -1, 2,  -1, 1,  -1, 0,  -1}},
         /* E7 */ {5, 17, {47, 30, 13, -1, -1, 46, 31, 14, -1, -1, 45, 48, 15,
                           -1, -1, 44, 49, 16, -1, -1, 43, 50, 17, -1, -1, 42,
                           51, 18, -1, -1, 41, 52, 19, 3,  -1, 40, 53, 20, 4,
                           -1, 39, 54, 21, 5,  -1, 38, 55, 22, 6,  -1, 37, 56,
                           23, 7,  -1, 36, 57, 24, 8,  -1, 35, 58, 25, 9,  -1,
                           34, 59, 26, 10, 0,  33, 60, 27, 11, 1,  32, 61, 28,
                           12, 2,  63, 62, 29, -1, -1}},
         /* E8 */ {3, 28, {-1, 16, -1, -1, 17, 58, -1, 18, 59, -1, 19, 60,
                           -1, 20, 61, -1, 21, 62, -1, 22, 63, -1, 23, 32,
                           -1, 26, 33, -1, 27, 34, 1,  28, 36, 2,  29, 37,
                           5,  48, 38, 6,  49, 39, 7,  50, 40, 8,  51, 41,
                           9,  52, 42, 10, 53, 43, 15, 54, 45, 14, 57, 47,
                           11, 55, 46, 3,  30, 44, -1, 24, 35, -1, 12, 56,
                           -1, 4,  31, -1, -1, 25, -1, -1, 13, -1, -1, 0}},
         /* E9 */ {4, 40, {11, 35, -1, -1, 12, 36, -1, -1, 13, 37, -1, -1, 14,
                           38, -1, -1, 15, 39, -1, -1, 16, 40, -1, -1, 17, 43,
                           -1, -1, 21, 44, -1, -1, 22, 45, -1, -1, 23, 46, -1,
                           -1, 24, 47, -1, -1, 25, -1, -1, -1, 26, -1, -1, -1,
                           27, -1, -1, -1, 30, -1, -1, -1, 31, -1, -1, -1, 59,
                           -1, -1, -1, 60, -1, -1, -1, 61, -1, -1, -1, 34, -1,
                           -1, -1, 33, -1, -1, -1, 62, -1, -1, -1, 28, -1, -1,
                           -1, 18, 41, -1, -1, -1, 63, -1, -1, -1, 29, -1, -1,
                           -1, 19, 42, -1, -1, -1, 32, -1, -1, -1, 20, -1, -1,
                           -1, 10, 58, -1, -1, 9,  57, -1, -1, 8,  56, -1, -1,
                           7,  55, -1, -1, 6,  54, -1, -1, 5,  53, -1, -1, 4,
                           52, -1, -1, 3,  51, -1, -1, 2,  50, -1, -1, 1,  49,
                           -1, -1, 0,  48}},
         /* L19 */ {2, 48, {47, -1, 46, -1, 45, -1, 44, -1, 43, -1, 42, -1,
                            41, -1, 40, -1, 39, -1, 38, -1, 37, -1, 36, -1,
                            35, -1, 34, -1, 33, -1, 32, -1, 63, -1, 62, -1,
                            61, -1, 60, -1, 59, -1, 58, -1, 57, -1, 56, -1,
                            55, -1, 54, -1, 53, -1, 52, -1, 51, -1, 50, -1,
                            49, -1, 48, -1, 31, 0,  30, 1,  29, 2,  28, 3,
                            27, 4,  26, 5,  25, 6,  24, 7,  23, 8,  22, 9,
                            21, 10, 20, 11, 19, 12, 18, 13, 17, 14, 16, 15}},
         /* L20 */ {2, 48, {-1, 0,  -1, 1,  -1, 2,  -1, 3,  -1, 4,  -1, 5,
                            -1, 6,  -1, 7,  -1, 8,  -1, 9,  -1, 10, -1, 11,
                            -1, 12, -1, 13, -1, 14, -1, 15, -1, 16, -1, 17,
                            -1, 18, -1, 19, -1, 20, -1, 21, -1, 22, -1, 23,
                            -1, 24, -1, 25, -1, 26, -1, 27, -1, 28, -1, 29,
                            -1, 30, -1, 31, 47, 48, 46, 49, 45, 50, 44, 51,
                            43, 52, 42, 53, 41, 54, 40, 55, 39, 56, 38, 57,
                            37, 58, 36, 59, 35, 60, 34, 61, 33, 62, 32, 63}},
         /* L5 */ {2, 40, {55, 56, 54, 57, 53, 58, 52, 59, 51, 60, 50, 61,
                           49, 62, 48, 63, 31, 32, 30, 33, 29, 34, 28, 35,
                           27, 36, 26, 37, 25, 38, 24, 39, 23, 40, 22, 41,
                           21, 42, 20, 43, 19, 44, 18, 45, 17, 46, 16, 47,
                           15, -1, 14, -1, 13, -1, 12, -1, 11, -1, 10, -1,
                           9,  -1, 8,  -1, 7,  -1, 6,  -1, 5,  -1, 4,  -1,
                           3,  -1, 2,  -1, 1,  -1, 0,  -1}},
         /* L6 */ {2, 40, {23, 24, 22, 25, 21, 26, 20, 27, 19, 28, 18, 29,
                           17, 30, 16, 31, 15, 48, 14, 49, 13, 50, 12, 51,
                           11, 52, 10, 53, 9,  54, 8,  55, 7,  56, 6,  57,
                           5,  58, 4,  59, 3,  60, 2,  61, 1,  62, 0,  63,
                           -1, 32, -1, 33, -1, 34, -1, 35, -1, 36, -1, 37,
                           -1, 38, -1, 39, -1, 40, -1, 41, -1, 42, -1, 43,
                           -1, 44, -1, 45, -1, 46, -1, 47}},
         /* L7 */ {2, 40, {47, -1, 46, -1, 45, -1, 44, -1, 43, -1, 42, -1,
                           41, -1, 40, -1, 39, -1, 38, -1, 37, -1, 36, -1,
                           35, -1, 34, -1, 33, -1, 32, -1, 63, 0,  62, 1,
                           61, 2,  60, 3,  59, 4,  58, 5,  57, 6,  56, 7,
                           55, 8,  54, 9,  53, 10, 52, 11, 51, 12, 50, 13,
                           49, 14, 48, 15, 31, 16, 30, 17, 29, 18, 28, 19,
                           27, 20, 26, 21, 25, 22, 24, 23}},
         /* L8 */ {2, 40, {-1, 0,  -1, 1,  -1, 2,  -1, 3,  -1, 4,  -1, 5,
                           -1, 6,  -1, 7,  -1, 8,  -1, 9,  -1, 10, -1, 11,
                           -1, 12, -1, 13, -1, 14, -1, 15, 47, 16, 46, 17,
                           45, 18, 44, 19, 43, 20, 42, 21, 41, 22, 40, 23,
                           39, 24, 38, 25, 37, 26, 36, 27, 35, 28, 34, 29,
                           33, 30, 32, 31, 63, 48, 62, 49, 61, 50, 60, 51,
                           59, 52, 58, 53, 57, 54, 56, 55}},
         /* O10 */ {2, 32, {48, 31, 49, 30, 50, 29, 51, 28, 52, 27, 53, 26, 54,
                            25, 55, 24, 56, 23, 57, 22, 58, 21, 59, 20, 60, 19,
                            61, 18, 62, 17, 63, 16, 32, 15, 33, 14, 34, 13, 35,
                            12, 36, 11, 37, 10, 38, 9,  39, 8,  40, 7,  41, 6,
                            42, 5,  43, 4,  44, 3,  45, 2,  46, 1,  47, 0}},
         /* O11 */ {2, 32, {31, 48, 30, 49, 29, 50, 28, 51, 27, 52, 26, 53, 25,
                            54, 24, 55, 23, 56, 22, 57, 21, 58, 20, 59, 19, 60,
                            18, 61, 17, 62, 16, 63, 15, 32, 14, 33, 13, 34, 12,
                            35, 11, 36, 10, 37, 9,  38, 8,  39, 7,  40, 6,  41,
                            5,  42, 4,  43, 3,  44, 2,  45, 1,  46, 0,  47}},
         /* O12 */ {2, 32, {47, 0,  46, 1,  45, 2,  44, 3,  43, 4,  42, 5,  41,
                            6,  40, 7,  39, 8,  38, 9,  37, 10, 36, 11, 35, 12,
                            34, 13, 33, 14, 32, 15, 63, 16, 62, 17, 61, 18, 60,
                            19, 59, 20, 58, 21, 57, 22, 56, 23, 55, 24, 54, 25,
                            53, 26, 52, 27, 51, 28, 50, 29, 49, 30, 48, 31}},
         /* O9 */ {2, 32, {0,  47, 1,  46, 2,  45, 3,  44, 4,  43, 5,  42, 6,
                           41, 7,  40, 8,  39, 9,  38, 10, 37, 11, 36, 12, 35,
                           13, 34, 14, 33, 15, 32, 16, 63, 17, 62, 18, 61, 19,
                           60, 20, 59, 21, 58, 22, 57, 23, 56, 24, 55, 25, 54,
                           26, 53, 27, 52, 28, 51, 29, 50, 30, 49, 31, 48}},
         /* Z1 */ {3, 40, {-1, 39, 40, -1, 38, 41, -1, 37, 42, -1, 36, 43,
                           -1, 35, 44, -1, 34, 45, -1, 33, 46, -1, 32, 47,
                           -1, 63, -1, -1, 62, -1, -1, 61, -1, -1, 60, -1,
                           -1, 59, -1, -1, 58, -1, -1, 57, -1, -1, 56, -1,
                           -1, 55, -1, -1, 54, -1, -1, 53, -1, -1, 52, -1,
                           -1, 51, -1, -1, 50, -1, -1, 49, -1, -1, 48, -1,
                           0,  31, -1, 1,  30, -1, 2,  29, -1, 3,  28, -1,
                           4,  27, -1, 5,  26, -1, 6,  25, -1, 7,  24, -1,
                           8,  23, -1, 9,  22, -1, 10, 21, -1, 11, 20, -1,
                           12, 19, -1, 13, 18, -1, 14, 17, -1, 15, 16, -1}},
         /* Z2 */ {3, 40, {7,  8,  -1, 6,  9,  -1, 5,  10, -1, 4,  11, -1,
                           3,  12, -1, 2,  13, -1, 1,  14, -1, 0,  15, -1,
                           -1, 16, -1, -1, 17, -1, -1, 18, -1, -1, 19, -1,
                           -1, 20, -1, -1, 21, -1, -1, 22, -1, -1, 23, -1,
                           -1, 24, -1, -1, 25, -1, -1, 26, -1, -1, 27, -1,
                           -1, 28, -1, -1, 29, -1, -1, 30, -1, -1, 31, -1,
                           -1, 48, 47, -1, 49, 46, -1, 50, 45, -1, 51, 44,
                           -1, 52, 43, -1, 53, 42, -1, 54, 41, -1, 55, 40,
                           -1, 56, 39, -1, 57, 38, -1, 58, 37, -1, 59, 36,
                           -1, 60, 35, -1, 61, 34, -1, 62, 33, -1, 63, 32}},
         /* Z3 */ {3, 40, {32, 63, -1, 33, 62, -1, 34, 61, -1, 35, 60, -1,
                           36, 59, -1, 37, 58, -1, 38, 57, -1, 39, 56, -1,
                           40, 55, -1, 41, 54, -1, 42, 53, -1, 43, 52, -1,
                           44, 51, -1, 45, 50, -1, 46, 49, -1, 47, 48, -1,
                           -1, 31, -1, -1, 30, -1, -1, 29, -1, -1, 28, -1,
                           -1, 27, -1, -1, 26, -1, -1, 25, -1, -1, 24, -1,
                           -1, 23, -1, -1, 22, -1, -1, 21, -1, -1, 20, -1,
                           -1, 19, -1, -1, 18, -1, -1, 17, -1, -1, 16, -1,
                           -1, 15, 0,  -1, 14, 1,  -1, 13, 2,  -1, 12, 3,
                           -1, 11, 4,  -1, 10, 5,  -1, 9,  6,  -1, 8,  7}},
         /* Z4 */ {3, 40, {-1, 16, 15, -1, 17, 14, -1, 18, 13, -1, 19, 12,
                           -1, 20, 11, -1, 21, 10, -1, 22, 9,  -1, 23, 8,
                           -1, 24, 7,  -1, 25, 6,  -1, 26, 5,  -1, 27, 4,
                           -1, 28, 3,  -1, 29, 2,  -1, 30, 1,  -1, 31, 0,
                           -1, 48, -1, -1, 49, -1, -1, 50, -1, -1, 51, -1,
                           -1, 52, -1, -1, 53, -1, -1, 54, -1, -1, 55, -1,
                           -1, 56, -1, -1, 57, -1, -1, 58, -1, -1, 59, -1,
                           -1, 60, -1, -1, 61, -1, -1, 62, -1, -1, 63, -1,
                           47, 32, -1, 46, 33, -1, 45, 34, -1, 44, 35, -1,
                           43, 36, -1, 42, 37, -1, 41, 38, -1, 40, 39, -1}},
         /* E14 */ {1, 5, {2, 3, 4, 5, 6}},
         /* E14 */ {1, 2, {1, 0}},
         /* E15 */ {2, 3, {-1, 1, 0, 2, -1, 3}}},
        /* PS */
        {{2.5, 0.5}, {5, 0.5}, {10, 0.5}}};
  } else {
    return new CathodeSegmentation{
        11,
        false,
        /* PG */
        {{1026, 11, 0, 74.2857132, -20},
         {1027, 11, 0, 68.57142639, -20},
         {1028, 11, 0, 62.8571434, -20},
         {1029, 11, 0, 57.1428566, -20},
         {1030, 11, 0, 51.42856979, -20},
         {1031, 11, 0, 45.7142868, -20},
         {1032, 11, 0, 40, -20},
         {1043, 7, 1, 25.7142849, -20},
         {1044, 14, 1, 14.28571415, -20},
         {1045, 6, 1, 4.440892099e-15, -20},
         {1128, 19, 2, -100, -20},
         {1129, 18, 2, -120, -20},
         {1133, 19, 2, -60, -20},
         {1134, 18, 2, -80, -20},
         {1140, 23, 1, -10, -20},
         {1141, 27, 1, -20, -20},
         {1142, 26, 1, -31.4285717, -20},
         {1143, 22, 1, -40, -20},
         {1225, 16, 2, -120, 0},
         {1226, 17, 2, -100, 0},
         {1229, 16, 2, -80, 0},
         {1230, 17, 2, -60, 0},
         {1233, 8, 1, -40, 0},
         {1234, 15, 1, -25.7142849, 0},
         {1235, 9, 1, -14.28571415, 0},
         {1325, 12, 0, 80, -20},
         {1326, 12, 0, 82.85713959, -20},
         {1327, 12, 0, 85.7142868, -20},
         {1328, 0, 0, 88.57142639, -17.5},
         {1329, 1, 0, 91.42857361, -15},
         {1330, 2, 0, 95, -10},
         {1331, 3, 0, 98.57142639, -7.5},
         {1332, 4, 0, 103.5714264, -5},
         {1333, 5, 0, 108.5714264, -2.5},
         {1334, 13, 0, 114.2857132, 0},
         {1345, 10, 0, 40, 0},
         {1346, 10, 0, 45.7142868, 0},
         {1347, 10, 0, 51.42856979, 0},
         {1348, 10, 0, 57.1428566, 0},
         {1349, 10, 0, 62.8571434, 0},
         {1350, 10, 0, 68.57142639, 0},
         {1351, 10, 0, 74.2857132, 0},
         {1357, 24, 1, -7.105427358e-15, -5},
         {1358, 20, 1, 8.571428299, -5},
         {1359, 21, 1, 20, -5},
         {1360, 25, 1, 30, -5}},
        /* PGT */
        {/* E1 */ {5, 15, {33, 50, -1, -1, -1, 34, 51, 20, 6,  -1, 35, 52, 21,
                           7,  -1, 36, 53, 22, 8,  -1, 37, 54, 23, 9,  -1, 38,
                           55, 24, 10, -1, 39, 56, 25, 11, -1, 40, 57, 26, 12,
                           -1, 41, 58, 27, 13, -1, 42, 59, 28, 14, 0,  43, 60,
                           29, 15, 1,  44, 61, 30, 16, 2,  45, 62, 31, 17, 3,
                           46, 63, 48, 18, 4,  47, 32, 49, 19, 5}},
         /* E2 */ {8, 14, {39, -1, -1, -1, -1, -1, -1, -1, 40, 58, 29, 15, -1,
                           -1, -1, -1, 41, 59, 30, 16, 3,  -1, -1, -1, 42, 60,
                           31, 17, 4,  -1, -1, -1, 43, 61, 48, 18, 5,  -1, -1,
                           -1, 44, 62, 49, 19, 6,  -1, -1, -1, 45, 63, 50, 20,
                           7,  -1, -1, -1, 46, 32, 51, 21, 8,  -1, -1, -1, -1,
                           33, 52, 22, 9,  -1, -1, -1, -1, 34, 53, 23, 10, -1,
                           -1, -1, -1, 35, 54, 24, 11, -1, -1, -1, -1, 36, 55,
                           25, 12, 0,  -1, -1, -1, 37, 56, 26, 13, 1,  -1, -1,
                           -1, 47, 38, 57, 28, 27, 14, 2}},
         /* E3 */ {9, 12, {39, 59, 31, -1, -1, -1, -1, -1, -1, 40, 60, 48,
                           20, 8,  -1, -1, -1, -1, 41, 61, 49, 21, 9,  -1,
                           -1, -1, -1, 42, 62, 50, 22, 10, -1, -1, -1, -1,
                           43, 63, 51, 23, 11, 0,  -1, -1, -1, 44, 32, 52,
                           24, 12, 1,  -1, -1, -1, 45, 33, 53, 25, 13, 2,
                           -1, -1, -1, 46, 34, 54, 26, 14, 3,  -1, -1, -1,
                           47, 35, 55, 27, 15, 4,  -1, -1, -1, -1, 36, 56,
                           28, 16, 5,  -1, -1, -1, -1, 37, 57, 29, 17, 6,
                           -1, -1, -1, -1, -1, -1, 38, 58, 30, 19, 18, 7}},
         /* E4 */ {10, 11, {45, 34, -1, -1, -1, -1, -1, -1, -1, -1, 46, 35, 56,
                            30, 20, 10, 0,  -1, -1, -1, 47, 36, 57, 31, 21, 11,
                            1,  -1, -1, -1, -1, 37, 58, 48, 22, 12, 2,  -1, -1,
                            -1, -1, 38, 59, 49, 23, 13, 3,  -1, -1, -1, -1, 39,
                            60, 50, 24, 14, 4,  -1, -1, -1, -1, 40, 61, 51, 25,
                            15, 5,  -1, -1, -1, -1, 41, 62, 52, 26, 16, 6,  -1,
                            -1, -1, -1, 42, 63, 53, 27, 17, 7,  -1, -1, -1, -1,
                            43, 32, 54, 28, 18, 8,  -1, -1, -1, -1, -1, -1, -1,
                            44, 33, 55, 29, 19, 9}},
         /* E5 */ {10, 10, {38, -1, -1, -1, -1, -1, -1, -1, -1, -1, 39, 61, 52,
                            27, 18, 9,  0,  -1, -1, -1, 40, 62, 53, 28, 19, 10,
                            1,  -1, -1, -1, 41, 63, 54, 29, 20, 11, 2,  -1, -1,
                            -1, 42, 32, 55, 30, 21, 12, 3,  -1, -1, -1, 43, 33,
                            56, 31, 22, 13, 4,  -1, -1, -1, 44, 34, 57, 48, 23,
                            14, 5,  -1, -1, -1, 45, 35, 58, 49, 24, 15, 6,  -1,
                            -1, -1, 46, 36, 59, 50, 25, 16, 7,  -1, -1, -1, -1,
                            -1, -1, 47, 37, 60, 51, 26, 17, 8}},
         /* E6 */ {8, 9, {39, 63, 55, -1, -1, -1, -1, -1, 40, 32, 56, 48,
                          24, 16, 8,  0,  41, 33, 57, 49, 25, 17, 9,  1,
                          42, 34, 58, 50, 26, 18, 10, 2,  43, 35, 59, 51,
                          27, 19, 11, 3,  44, 36, 60, 52, 28, 20, 12, 4,
                          45, 37, 61, 53, 29, 21, 13, 5,  46, 38, 62, 54,
                          30, 22, 14, 6,  -1, -1, -1, 47, 31, 23, 15, 7}},
         /* L1 */ {20, 4, {3,  7,  11, 15, 18, 21, 24, 27, 30, 49, 52, 55,
                           58, 61, 32, 35, 38, 41, 44, 47, 2,  6,  10, 14,
                           17, 20, 23, 26, 29, 48, 51, 54, 57, 60, 63, 34,
                           37, 40, 43, 46, 1,  5,  9,  13, 16, 19, 22, 25,
                           28, 31, 50, 53, 56, 59, 62, 33, 36, 39, 42, 45,
                           0,  4,  8,  12, -1, -1, -1, -1, -1, -1, -1, -1,
                           -1, -1, -1, -1, -1, -1, -1, -1}},
         /* L2 */ {20, 4, {2,  5,  8,  11, 14, 17, 20, 23, 26, 29, 48, 51,
                           54, 57, 60, 63, 35, 39, 43, 47, 1,  4,  7,  10,
                           13, 16, 19, 22, 25, 28, 31, 50, 53, 56, 59, 62,
                           34, 38, 42, 46, 0,  3,  6,  9,  12, 15, 18, 21,
                           24, 27, 30, 49, 52, 55, 58, 61, 33, 37, 41, 45,
                           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                           -1, -1, -1, -1, 32, 36, 40, 44}},
         /* L3 */ {20, 4, {44, 40, 36, 32, -1, -1, -1, -1, -1, -1, -1, -1,
                           -1, -1, -1, -1, -1, -1, -1, -1, 45, 41, 37, 33,
                           61, 58, 55, 52, 49, 30, 27, 24, 21, 18, 15, 12,
                           9,  6,  3,  0,  46, 42, 38, 34, 62, 59, 56, 53,
                           50, 31, 28, 25, 22, 19, 16, 13, 10, 7,  4,  1,
                           47, 43, 39, 35, 63, 60, 57, 54, 51, 48, 29, 26,
                           23, 20, 17, 14, 11, 8,  5,  2}},
         /* L4 */ {20, 4, {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                           -1, -1, -1, -1, 12, 8,  4,  0,  45, 42, 39, 36,
                           33, 62, 59, 56, 53, 50, 31, 28, 25, 22, 19, 16,
                           13, 9,  5,  1,  46, 43, 40, 37, 34, 63, 60, 57,
                           54, 51, 48, 29, 26, 23, 20, 17, 14, 10, 6,  2,
                           47, 44, 41, 38, 35, 32, 61, 58, 55, 52, 49, 30,
                           27, 24, 21, 18, 15, 11, 7,  3}},
         /* O1 */ {8, 8, {40, 32, 56, 48, 24, 16, 8,  0,  41, 33, 57, 49, 25,
                          17, 9,  1,  42, 34, 58, 50, 26, 18, 10, 2,  43, 35,
                          59, 51, 27, 19, 11, 3,  44, 36, 60, 52, 28, 20, 12,
                          4,  45, 37, 61, 53, 29, 21, 13, 5,  46, 38, 62, 54,
                          30, 22, 14, 6,  47, 39, 63, 55, 31, 23, 15, 7}},
         /* O2 */ {8, 8, {7,  15, 23, 31, 55, 63, 39, 47, 6,  14, 22, 30, 54,
                          62, 38, 46, 5,  13, 21, 29, 53, 61, 37, 45, 4,  12,
                          20, 28, 52, 60, 36, 44, 3,  11, 19, 27, 51, 59, 35,
                          43, 2,  10, 18, 26, 50, 58, 34, 42, 1,  9,  17, 25,
                          49, 57, 33, 41, 0,  8,  16, 24, 48, 56, 32, 40}},
         /* O26 */ {4, 16, {32, 48, 16, 0,  33, 49, 17, 1,  34, 50, 18, 2,  35,
                            51, 19, 3,  36, 52, 20, 4,  37, 53, 21, 5,  38, 54,
                            22, 6,  39, 55, 23, 7,  40, 56, 24, 8,  41, 57, 25,
                            9,  42, 58, 26, 10, 43, 59, 27, 11, 44, 60, 28, 12,
                            45, 61, 29, 13, 46, 62, 30, 14, 47, 63, 31, 15}},
         /* O27 */ {8, 8, {40, 32, 56, 48, 24, 16, 8,  0,  41, 33, 57, 49, 25,
                           17, 9,  1,  42, 34, 58, 50, 26, 18, 10, 2,  43, 35,
                           59, 51, 27, 19, 11, 3,  44, 36, 60, 52, 28, 20, 12,
                           4,  45, 37, 61, 53, 29, 21, 13, 5,  46, 38, 62, 54,
                           30, 22, 14, 6,  47, 39, 63, 55, 31, 23, 15, 7}},
         /* O3 */ {16, 4, {3,  7,  11, 15, 19, 23, 27, 31, 51, 55, 59, 63, 35,
                           39, 43, 47, 2,  6,  10, 14, 18, 22, 26, 30, 50, 54,
                           58, 62, 34, 38, 42, 46, 1,  5,  9,  13, 17, 21, 25,
                           29, 49, 53, 57, 61, 33, 37, 41, 45, 0,  4,  8,  12,
                           16, 20, 24, 28, 48, 52, 56, 60, 32, 36, 40, 44}},
         /* O4 */ {16, 4, {44, 40, 36, 32, 60, 56, 52, 48, 28, 24, 20, 16, 12,
                           8,  4,  0,  45, 41, 37, 33, 61, 57, 53, 49, 29, 25,
                           21, 17, 13, 9,  5,  1,  46, 42, 38, 34, 62, 58, 54,
                           50, 30, 26, 22, 18, 14, 10, 6,  2,  47, 43, 39, 35,
                           63, 59, 55, 51, 31, 27, 23, 19, 15, 11, 7,  3}},
         /* O5 */ {28, 2, {47, 45, 43, 41, 39, 37, 35, 33, 63, 61, 59, 57,
                           55, 53, 51, 49, 31, 29, 27, 25, 23, 21, 19, 17,
                           15, 13, 11, 9,  46, 44, 42, 40, 38, 36, 34, 32,
                           62, 60, 58, 56, 54, 52, 50, 48, 30, 28, 26, 24,
                           22, 20, 18, 16, 14, 12, 10, 8}},
         /* O6 */ {28, 2, {39, 37, 35, 33, 63, 61, 59, 57, 55, 53, 51, 49,
                           31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9,
                           7,  5,  3,  1,  38, 36, 34, 32, 62, 60, 58, 56,
                           54, 52, 50, 48, 30, 28, 26, 24, 22, 20, 18, 16,
                           14, 12, 10, 8,  6,  4,  2,  0}},
         /* O7 */ {28, 2, {0,  2,  4,  6,  8,  10, 12, 14, 16, 18, 20, 22,
                           24, 26, 28, 30, 48, 50, 52, 54, 56, 58, 60, 62,
                           32, 34, 36, 38, 1,  3,  5,  7,  9,  11, 13, 15,
                           17, 19, 21, 23, 25, 27, 29, 31, 49, 51, 53, 55,
                           57, 59, 61, 63, 33, 35, 37, 39}},
         /* O8 */ {28, 2, {8,  10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30,
                           48, 50, 52, 54, 56, 58, 60, 62, 32, 34, 36, 38,
                           40, 42, 44, 46, 9,  11, 13, 15, 17, 19, 21, 23,
                           25, 27, 29, 31, 49, 51, 53, 55, 57, 59, 61, 63,
                           33, 35, 37, 39, 41, 43, 45, 47}},
         /* P1 */ {16, 5, {47, 46, 41, 36, 63, 58, 53, 48, -1, -1, -1, -1,
                           -1, -1, -1, -1, -1, -1, 42, 37, 32, 59, 54, 49,
                           28, 24, 20, 16, 12, 8,  4,  0,  -1, -1, 43, 38,
                           33, 60, 55, 50, 29, 25, 21, 17, 13, 9,  5,  1,
                           -1, -1, 44, 39, 34, 61, 56, 51, 30, 26, 22, 18,
                           14, 10, 6,  2,  -1, -1, 45, 40, 35, 62, 57, 52,
                           31, 27, 23, 19, 15, 11, 7,  3}},
         /* P2 */ {16, 5, {-1, -1, -1, -1, -1, -1, -1, -1, 27, 22, 17, 12,
                           7,  2,  1,  0,  44, 40, 36, 32, 60, 56, 52, 48,
                           28, 23, 18, 13, 8,  3,  -1, -1, 45, 41, 37, 33,
                           61, 57, 53, 49, 29, 24, 19, 14, 9,  4,  -1, -1,
                           46, 42, 38, 34, 62, 58, 54, 50, 30, 25, 20, 15,
                           10, 5,  -1, -1, 47, 43, 39, 35, 63, 59, 55, 51,
                           31, 26, 21, 16, 11, 6,  -1, -1}},
         /* P3 */ {14, 5, {3,  7,  11, 15, 20, 25, 30, 51, 56, 61, 34, 39,
                           43, 47, 2,  6,  10, 14, 19, 24, 29, 50, 55, 60,
                           33, 38, 42, 46, 1,  5,  9,  13, 18, 23, 28, 49,
                           54, 59, 32, 37, 41, 45, 0,  4,  8,  12, 17, 22,
                           27, 48, 53, 58, 63, 36, 40, 44, -1, -1, -1, -1,
                           16, 21, 26, 31, 52, 57, 62, 35, -1, -1}},
         /* P4 */ {14, 5, {3,  7,  12, 17, 22, 27, 48, 53, 58, 63, 35, 39,
                           43, 47, 2,  6,  11, 16, 21, 26, 31, 52, 57, 62,
                           34, 38, 42, 46, 1,  5,  10, 15, 20, 25, 30, 51,
                           56, 61, 33, 37, 41, 45, 0,  4,  9,  14, 19, 24,
                           29, 50, 55, 60, 32, 36, 40, 44, -1, -1, 8,  13,
                           18, 23, 28, 49, 54, 59, -1, -1, -1, -1}},
         /* Q1 */ {14, 5, {-1, -1, -1, -1, 59, 54, 49, 28, 23, 18, 13, 8,
                           -1, -1, 44, 40, 36, 32, 60, 55, 50, 29, 24, 19,
                           14, 9,  4,  0,  45, 41, 37, 33, 61, 56, 51, 30,
                           25, 20, 15, 10, 5,  1,  46, 42, 38, 34, 62, 57,
                           52, 31, 26, 21, 16, 11, 6,  2,  47, 43, 39, 35,
                           63, 58, 53, 48, 27, 22, 17, 12, 7,  3}},
         /* Q2 */ {14, 5, {-1, -1, 35, 62, 57, 52, 31, 26, 21, 16, -1, -1,
                           -1, -1, 44, 40, 36, 63, 58, 53, 48, 27, 22, 17,
                           12, 8,  4,  0,  45, 41, 37, 32, 59, 54, 49, 28,
                           23, 18, 13, 9,  5,  1,  46, 42, 38, 33, 60, 55,
                           50, 29, 24, 19, 14, 10, 6,  2,  47, 43, 39, 34,
                           61, 56, 51, 30, 25, 20, 15, 11, 7,  3}},
         /* Q3 */ {16, 5, {-1, -1, 6,  11, 16, 21, 26, 31, 51, 55, 59, 63,
                           35, 39, 43, 47, -1, -1, 5,  10, 15, 20, 25, 30,
                           50, 54, 58, 62, 34, 38, 42, 46, -1, -1, 4,  9,
                           14, 19, 24, 29, 49, 53, 57, 61, 33, 37, 41, 45,
                           -1, -1, 3,  8,  13, 18, 23, 28, 48, 52, 56, 60,
                           32, 36, 40, 44, 0,  1,  2,  7,  12, 17, 22, 27,
                           -1, -1, -1, -1, -1, -1, -1, -1}},
         /* Q4 */ {16, 5, {3,  7,  11, 15, 19, 23, 27, 31, 52, 57, 62, 35,
                           40, 45, -1, -1, 2,  6,  10, 14, 18, 22, 26, 30,
                           51, 56, 61, 34, 39, 44, -1, -1, 1,  5,  9,  13,
                           17, 21, 25, 29, 50, 55, 60, 33, 38, 43, -1, -1,
                           0,  4,  8,  12, 16, 20, 24, 28, 49, 54, 59, 32,
                           37, 42, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                           48, 53, 58, 63, 36, 41, 46, 47}}},
        /* PS */
        {{0.714285714, 2.5}, {0.714285714, 5}, {0.714285714, 10}}};
  }
}
class CathodeSegmentationCreatorRegisterCreateSegType11 {
public:
  CathodeSegmentationCreatorRegisterCreateSegType11() {
    registerCathodeSegmentationCreator(11, createSegType11);
  }
} aCathodeSegmentationCreatorRegisterCreateSegType11;

} // namespace impl3
} // namespace mapping
} // namespace mch
} // namespace o2

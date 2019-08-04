// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GBTFrame.cxx
/// \author Sebastian Klewin

#include "TPCReconstruction/GBTFrame.h"
#include <iostream>

using namespace o2::tpc;

GBTFrame::GBTFrame(short s0hw0l, short s0hw1l, short s0hw2l, short s0hw3l,
                   short s0hw0h, short s0hw1h, short s0hw2h, short s0hw3h,
                   short s1hw0l, short s1hw1l, short s1hw2l, short s1hw3l,
                   short s1hw0h, short s1hw1h, short s1hw2h, short s1hw3h,
                   short s2hw0, short s2hw1, short s2hw2, short s2hw3,
                   short s0adc, short s1adc, short s2adc, unsigned marker)
{

  mWords[0] =
    (BIT02(s0hw0h) << 31) | (BIT02(s0hw1h) << 30) | (BIT02(s0hw2h) << 29) | (BIT02(s0hw3h) << 28) |
    (BIT01(s0hw0h) << 27) | (BIT01(s0hw1h) << 26) | (BIT01(s0hw2h) << 25) | (BIT01(s0hw3h) << 24) |
    (BIT00(s0hw0h) << 23) | (BIT00(s0hw1h) << 22) | (BIT00(s0hw2h) << 21) | (BIT00(s0hw3h) << 20) |
    (BIT04(s0hw0l) << 19) | (BIT04(s0hw1l) << 18) | (BIT04(s0hw2l) << 17) | (BIT04(s0hw3l) << 16) |
    (BIT03(s0hw0l) << 15) | (BIT03(s0hw1l) << 14) | (BIT03(s0hw2l) << 13) | (BIT03(s0hw3l) << 12) |
    (BIT02(s0hw0l) << 11) | (BIT02(s0hw1l) << 10) | (BIT02(s0hw2l) << 9) | (BIT02(s0hw3l) << 8) |
    (BIT01(s0hw0l) << 7) | (BIT01(s0hw1l) << 6) | (BIT01(s0hw2l) << 5) | (BIT01(s0hw3l) << 4) |
    (BIT00(s0hw0l) << 3) | (BIT00(s0hw1l) << 2) | (BIT00(s0hw2l) << 1) | (BIT00(s0hw3l));

  mWords[1] =
    (BIT04(s1hw0l) << 31) | (BIT04(s1hw1l) << 30) | (BIT04(s1hw2l) << 29) | (BIT04(s1hw3l) << 28) |
    (BIT03(s1hw0l) << 27) | (BIT03(s1hw1l) << 26) | (BIT03(s1hw2l) << 25) | (BIT03(s1hw3l) << 24) |
    (BIT02(s1hw0l) << 23) | (BIT02(s1hw1l) << 22) | (BIT02(s1hw2l) << 21) | (BIT02(s1hw3l) << 20) |
    (BIT01(s1hw0l) << 19) | (BIT01(s1hw1l) << 18) | (BIT01(s1hw2l) << 17) | (BIT01(s1hw3l) << 16) |
    (BIT00(s1hw0l) << 15) | (BIT00(s1hw1l) << 14) | (BIT00(s1hw2l) << 13) | (BIT00(s1hw3l) << 12) |
    (BIT03(s0adc) << 11) | (BIT02(s0adc) << 10) | (BIT01(s0adc) << 9) | (BIT00(s0adc) << 8) |
    (BIT04(s0hw0h) << 7) | (BIT04(s0hw1h) << 6) | (BIT04(s0hw2h) << 5) | (BIT04(s0hw3h) << 4) |
    (BIT03(s0hw0h) << 3) | (BIT03(s0hw1h) << 2) | (BIT03(s0hw2h) << 1) | (BIT03(s0hw3h));

  mWords[2] =
    (BIT01(s2hw0) << 31) | (BIT01(s2hw1) << 30) | (BIT01(s2hw2) << 29) | (BIT01(s2hw3) << 28) |
    (BIT00(s2hw0) << 27) | (BIT00(s2hw1) << 26) | (BIT00(s2hw2) << 25) | (BIT00(s2hw3) << 24) |
    (BIT03(s1adc) << 23) | (BIT02(s1adc) << 22) | (BIT01(s1adc) << 21) | (BIT00(s1adc) << 20) |
    (BIT04(s1hw0h) << 19) | (BIT04(s1hw1h) << 18) | (BIT04(s1hw2h) << 17) | (BIT04(s1hw3h) << 16) |
    (BIT03(s1hw0h) << 15) | (BIT03(s1hw1h) << 14) | (BIT03(s1hw2h) << 13) | (BIT03(s1hw3h) << 12) |
    (BIT02(s1hw0h) << 11) | (BIT02(s1hw1h) << 10) | (BIT02(s1hw2h) << 9) | (BIT02(s1hw3h) << 8) |
    (BIT01(s1hw0h) << 7) | (BIT01(s1hw1h) << 6) | (BIT01(s1hw2h) << 5) | (BIT01(s1hw3h) << 4) |
    (BIT00(s1hw0h) << 3) | (BIT00(s1hw1h) << 2) | (BIT00(s1hw2h) << 1) | (BIT00(s1hw3h));

  mWords[3] =
    (BIT15(marker) << 31) | (BIT14(marker) << 30) | (BIT13(marker) << 29) | (BIT12(marker) << 28) |
    (BIT11(marker) << 27) | (BIT10(marker) << 26) | (BIT09(marker) << 25) | (BIT08(marker) << 24) |
    (BIT07(marker) << 23) | (BIT06(marker) << 22) | (BIT05(marker) << 21) | (BIT04(marker) << 20) |
    (BIT03(marker) << 19) | (BIT02(marker) << 18) | (BIT01(marker) << 17) | (BIT00(marker) << 16) |
    (BIT03(s2adc) << 15) | (BIT02(s2adc) << 14) | (BIT01(s2adc) << 13) | (BIT00(s2adc) << 12) |
    (BIT04(s2hw0) << 11) | (BIT04(s2hw1) << 10) | (BIT04(s2hw2) << 9) | (BIT04(s2hw3) << 8) |
    (BIT03(s2hw0) << 7) | (BIT03(s2hw1) << 6) | (BIT03(s2hw2) << 5) | (BIT03(s2hw3) << 4) |
    (BIT02(s2hw0) << 3) | (BIT02(s2hw1) << 2) | (BIT02(s2hw2) << 1) | (BIT02(s2hw3));

  mHalfWords[0][0][0] = s0hw0l;
  mHalfWords[0][0][1] = s0hw1l;
  mHalfWords[0][0][2] = s0hw2l;
  mHalfWords[0][0][3] = s0hw3l;

  mHalfWords[0][1][0] = s0hw0h;
  mHalfWords[0][1][1] = s0hw1h;
  mHalfWords[0][1][2] = s0hw2h;
  mHalfWords[0][1][3] = s0hw3h;

  mHalfWords[1][0][0] = s1hw0l;
  mHalfWords[1][0][1] = s1hw1l;
  mHalfWords[1][0][2] = s1hw2l;
  mHalfWords[1][0][3] = s1hw3l;

  mHalfWords[1][1][0] = s1hw0h;
  mHalfWords[1][1][1] = s1hw1h;
  mHalfWords[1][1][2] = s1hw2h;
  mHalfWords[1][1][3] = s1hw3h;

  mHalfWords[2][0][0] = s2hw0;
  mHalfWords[2][0][1] = s2hw1;
  mHalfWords[2][0][2] = s2hw2;
  mHalfWords[2][0][3] = s2hw3;

  mHalfWords[2][1][0] = mHalfWords[2][0][0];
  mHalfWords[2][1][1] = mHalfWords[2][0][1];
  mHalfWords[2][1][2] = mHalfWords[2][0][2];
  mHalfWords[2][1][3] = mHalfWords[2][0][3];

  mAdcClock[0] = s0adc;
  mAdcClock[1] = s0adc;
  mAdcClock[2] = s0adc;
}

void GBTFrame::setData(short s0hw0l, short s0hw1l, short s0hw2l, short s0hw3l,
                       short s0hw0h, short s0hw1h, short s0hw2h, short s0hw3h,
                       short s1hw0l, short s1hw1l, short s1hw2l, short s1hw3l,
                       short s1hw0h, short s1hw1h, short s1hw2h, short s1hw3h,
                       short s2hw0, short s2hw1, short s2hw2, short s2hw3,
                       short s0adc, short s1adc, short s2adc, unsigned marker)
{
  mWords[0] =
    (BIT02(s0hw0h) << 31) | (BIT02(s0hw1h) << 30) | (BIT02(s0hw2h) << 29) | (BIT02(s0hw3h) << 28) |
    (BIT01(s0hw0h) << 27) | (BIT01(s0hw1h) << 26) | (BIT01(s0hw2h) << 25) | (BIT01(s0hw3h) << 24) |
    (BIT00(s0hw0h) << 23) | (BIT00(s0hw1h) << 22) | (BIT00(s0hw2h) << 21) | (BIT00(s0hw3h) << 20) |
    (BIT04(s0hw0l) << 19) | (BIT04(s0hw1l) << 18) | (BIT04(s0hw2l) << 17) | (BIT04(s0hw3l) << 16) |
    (BIT03(s0hw0l) << 15) | (BIT03(s0hw1l) << 14) | (BIT03(s0hw2l) << 13) | (BIT03(s0hw3l) << 12) |
    (BIT02(s0hw0l) << 11) | (BIT02(s0hw1l) << 10) | (BIT02(s0hw2l) << 9) | (BIT02(s0hw3l) << 8) |
    (BIT01(s0hw0l) << 7) | (BIT01(s0hw1l) << 6) | (BIT01(s0hw2l) << 5) | (BIT01(s0hw3l) << 4) |
    (BIT00(s0hw0l) << 3) | (BIT00(s0hw1l) << 2) | (BIT00(s0hw2l) << 1) | (BIT00(s0hw3l));

  mWords[1] =
    (BIT04(s1hw0l) << 31) | (BIT04(s1hw1l) << 30) | (BIT04(s1hw2l) << 29) | (BIT04(s1hw3l) << 28) |
    (BIT03(s1hw0l) << 27) | (BIT03(s1hw1l) << 26) | (BIT03(s1hw2l) << 25) | (BIT03(s1hw3l) << 24) |
    (BIT02(s1hw0l) << 23) | (BIT02(s1hw1l) << 22) | (BIT02(s1hw2l) << 21) | (BIT02(s1hw3l) << 20) |
    (BIT01(s1hw0l) << 19) | (BIT01(s1hw1l) << 18) | (BIT01(s1hw2l) << 17) | (BIT01(s1hw3l) << 16) |
    (BIT00(s1hw0l) << 15) | (BIT00(s1hw1l) << 14) | (BIT00(s1hw2l) << 13) | (BIT00(s1hw3l) << 12) |
    (BIT03(s0adc) << 11) | (BIT02(s0adc) << 10) | (BIT01(s0adc) << 9) | (BIT00(s0adc) << 8) |
    (BIT04(s0hw0h) << 7) | (BIT04(s0hw1h) << 6) | (BIT04(s0hw2h) << 5) | (BIT04(s0hw3h) << 4) |
    (BIT03(s0hw0h) << 3) | (BIT03(s0hw1h) << 2) | (BIT03(s0hw2h) << 1) | (BIT03(s0hw3h));

  mWords[2] =
    (BIT01(s2hw0) << 31) | (BIT01(s2hw1) << 30) | (BIT01(s2hw2) << 29) | (BIT01(s2hw3) << 28) |
    (BIT00(s2hw0) << 27) | (BIT00(s2hw1) << 26) | (BIT00(s2hw2) << 25) | (BIT00(s2hw3) << 24) |
    (BIT03(s1adc) << 23) | (BIT02(s1adc) << 22) | (BIT01(s1adc) << 21) | (BIT00(s1adc) << 20) |
    (BIT04(s1hw0h) << 19) | (BIT04(s1hw1h) << 18) | (BIT04(s1hw2h) << 17) | (BIT04(s1hw3h) << 16) |
    (BIT03(s1hw0h) << 15) | (BIT03(s1hw1h) << 14) | (BIT03(s1hw2h) << 13) | (BIT03(s1hw3h) << 12) |
    (BIT02(s1hw0h) << 11) | (BIT02(s1hw1h) << 10) | (BIT02(s1hw2h) << 9) | (BIT02(s1hw3h) << 8) |
    (BIT01(s1hw0h) << 7) | (BIT01(s1hw1h) << 6) | (BIT01(s1hw2h) << 5) | (BIT01(s1hw3h) << 4) |
    (BIT00(s1hw0h) << 3) | (BIT00(s1hw1h) << 2) | (BIT00(s1hw2h) << 1) | (BIT00(s1hw3h));

  mWords[3] =
    (BIT15(marker) << 31) | (BIT14(marker) << 30) | (BIT13(marker) << 29) | (BIT12(marker) << 28) |
    (BIT11(marker) << 27) | (BIT10(marker) << 26) | (BIT09(marker) << 25) | (BIT08(marker) << 24) |
    (BIT07(marker) << 23) | (BIT06(marker) << 22) | (BIT05(marker) << 21) | (BIT04(marker) << 20) |
    (BIT03(marker) << 19) | (BIT02(marker) << 18) | (BIT01(marker) << 17) | (BIT00(marker) << 16) |
    (BIT03(s2adc) << 15) | (BIT02(s2adc) << 14) | (BIT01(s2adc) << 13) | (BIT00(s2adc) << 12) |
    (BIT04(s2hw0) << 11) | (BIT04(s2hw1) << 10) | (BIT04(s2hw2) << 9) | (BIT04(s2hw3) << 8) |
    (BIT03(s2hw0) << 7) | (BIT03(s2hw1) << 6) | (BIT03(s2hw2) << 5) | (BIT03(s2hw3) << 4) |
    (BIT02(s2hw0) << 3) | (BIT02(s2hw1) << 2) | (BIT02(s2hw2) << 1) | (BIT02(s2hw3));

  mHalfWords[0][0][0] = s0hw0l;
  mHalfWords[0][0][1] = s0hw1l;
  mHalfWords[0][0][2] = s0hw2l;
  mHalfWords[0][0][3] = s0hw3l;

  mHalfWords[0][1][0] = s0hw0h;
  mHalfWords[0][1][1] = s0hw1h;
  mHalfWords[0][1][2] = s0hw2h;
  mHalfWords[0][1][3] = s0hw3h;

  mHalfWords[1][0][0] = s1hw0l;
  mHalfWords[1][0][1] = s1hw1l;
  mHalfWords[1][0][2] = s1hw2l;
  mHalfWords[1][0][3] = s1hw3l;

  mHalfWords[1][1][0] = s1hw0h;
  mHalfWords[1][1][1] = s1hw1h;
  mHalfWords[1][1][2] = s1hw2h;
  mHalfWords[1][1][3] = s1hw3h;

  mHalfWords[2][0][0] = s2hw0;
  mHalfWords[2][0][1] = s2hw1;
  mHalfWords[2][0][2] = s2hw2;
  mHalfWords[2][0][3] = s2hw3;

  mHalfWords[2][1][0] = mHalfWords[2][0][0];
  mHalfWords[2][1][1] = mHalfWords[2][0][1];
  mHalfWords[2][1][2] = mHalfWords[2][0][2];
  mHalfWords[2][1][3] = mHalfWords[2][0][3];

  mAdcClock[0] = s0adc;
  mAdcClock[1] = s0adc;
  mAdcClock[2] = s0adc;
}

void GBTFrame::setAdcClock(int sampa, int clock)
{
  switch (sampa) {
    case 0:
      mWords[1] = (mWords[1] & 0xFFFFF0FF) | ((clock & 0xF) << 8);
      break;
    case 1:
      mWords[2] = (mWords[2] & 0xFF0FFFFF) | ((clock & 0xF) << 20);
      break;
    case 2:
      mWords[3] = (mWords[3] & 0xFFFF0FFF) | ((clock & 0xF) << 12);
      break;
    case -1:
      mWords[1] = (mWords[1] & 0xFFFFF0FF) | ((clock & 0xF) << 8);
      mWords[2] = (mWords[2] & 0xFF0FFFFF) | ((clock & 0xF) << 20);
      mWords[3] = (mWords[3] & 0xFFFF0FFF) | ((clock & 0xF) << 12);
      break;
    default:
      std::cout << "don't know SAMPA " << sampa << std::endl;
      break;
  }
  calculateAdcClock();
}

std::ostream& GBTFrame::Print(std::ostream& output) const
{
  output << "0x" << std::hex
         << std::setfill('0') << std::right << std::setw(8) << mWords[3]
         << std::setfill('0') << std::right << std::setw(8) << mWords[2]
         << std::setfill('0') << std::right << std::setw(8) << mWords[1]
         << std::setfill('0') << std::right << std::setw(8) << mWords[0]
         << std::dec;
  return output;
}

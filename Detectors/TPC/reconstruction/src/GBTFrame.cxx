/// \file GBTFrame.cxx
/// \author Sebastian Klewin

#include "TPCReconstruction/GBTFrame.h"
using namespace AliceO2::TPC;

GBTFrame::GBTFrame()
  : GBTFrame(0,0,0,0)
{}

GBTFrame::GBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0)
{
  mWords[3] = word3;
  mWords[2] = word2;
  mWords[1] = word1;
  mWords[0] = word0;

  calculateHalfWords();
}

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
      (BIT02(s0hw0l) << 11) | (BIT02(s0hw1l) << 10) | (BIT02(s0hw2l) <<  9) | (BIT02(s0hw3l) <<  8) | 
      (BIT01(s0hw0l) <<  7) | (BIT01(s0hw1l) <<  6) | (BIT01(s0hw2l) <<  5) | (BIT01(s0hw3l) <<  4) | 
      (BIT00(s0hw0l) <<  3) | (BIT00(s0hw1l) <<  2) | (BIT00(s0hw2l) <<  1) | (BIT00(s0hw3l));

  mWords[1] =
      (BIT04(s1hw0l) << 31) | (BIT04(s1hw1l) << 30) | (BIT04(s1hw2l) << 29) | (BIT04(s1hw3l) << 28) | 
      (BIT03(s1hw0l) << 27) | (BIT03(s1hw1l) << 26) | (BIT03(s1hw2l) << 25) | (BIT03(s1hw3l) << 24) |
      (BIT02(s1hw0l) << 23) | (BIT02(s1hw1l) << 22) | (BIT02(s1hw2l) << 21) | (BIT02(s1hw3l) << 20) | 
      (BIT01(s1hw0l) << 19) | (BIT01(s1hw1l) << 18) | (BIT01(s1hw2l) << 17) | (BIT01(s1hw3l) << 16) | 
      (BIT00(s1hw0l) << 15) | (BIT00(s1hw1l) << 14) | (BIT00(s1hw2l) << 13) | (BIT00(s1hw3l) << 12) |
      (BIT03(s0adc)  << 11) | (BIT02(s0adc)  << 10) | (BIT01(s0adc)  <<  9) | (BIT00(s0adc)  <<  8) |
      (BIT04(s0hw0h) <<  7) | (BIT04(s0hw1h) <<  6) | (BIT04(s0hw2h) <<  5) | (BIT04(s0hw3h) <<  4) | 
      (BIT03(s0hw0h) <<  3) | (BIT03(s0hw1h) <<  2) | (BIT03(s0hw2h) <<  1) | (BIT03(s0hw3h));

  mWords[2] =
      (BIT01(s2hw0)  << 31) | (BIT01(s2hw1)  << 30) | (BIT01(s2hw2)  << 29) | (BIT01(s2hw3)  << 28) | 
      (BIT00(s2hw0)  << 27) | (BIT00(s2hw1)  << 26) | (BIT00(s2hw2)  << 25) | (BIT00(s2hw3)  << 24) | 
      (BIT03(s1adc)  << 23) | (BIT02(s1adc)  << 22) | (BIT01(s1adc)  << 21) | (BIT00(s1adc)  << 20) |
      (BIT04(s1hw0h) << 19) | (BIT04(s1hw1h) << 18) | (BIT04(s1hw2h) << 17) | (BIT04(s1hw3h) << 16) | 
      (BIT03(s1hw0h) << 15) | (BIT03(s1hw1h) << 14) | (BIT03(s1hw2h) << 13) | (BIT03(s1hw3h) << 12) |
      (BIT02(s1hw0h) << 11) | (BIT02(s1hw1h) << 10) | (BIT02(s1hw2h) <<  9) | (BIT02(s1hw3h) <<  8) | 
      (BIT01(s1hw0h) <<  7) | (BIT01(s1hw1h) <<  6) | (BIT01(s1hw2h) <<  5) | (BIT01(s1hw3h) <<  4) | 
      (BIT00(s1hw0h) <<  3) | (BIT00(s1hw1h) <<  2) | (BIT00(s1hw2h) <<  1) | (BIT00(s1hw3h));

  mWords[3] =
      (BIT15(marker) << 31) | (BIT14(marker) << 30) | (BIT13(marker) << 29) | (BIT12(marker) << 28) |
      (BIT11(marker) << 27) | (BIT10(marker) << 26) | (BIT09(marker) << 25) | (BIT08(marker) << 24) | 
      (BIT07(marker) << 23) | (BIT06(marker) << 22) | (BIT05(marker) << 21) | (BIT04(marker) << 20) | 
      (BIT03(marker) << 19) | (BIT02(marker) << 18) | (BIT01(marker) << 17) | (BIT00(marker) << 16) |
      (BIT03(s2adc)  << 15) | (BIT02(s2adc)  << 14) | (BIT01(s2adc)  << 13) | (BIT00(s2adc ) << 12) |
      (BIT04(s2hw0)  << 11) | (BIT04(s2hw1)  << 10) | (BIT04(s2hw2)  <<  9) | (BIT04(s2hw3 ) <<  8) | 
      (BIT03(s2hw0)  <<  7) | (BIT03(s2hw1)  <<  6) | (BIT03(s2hw2)  <<  5) | (BIT03(s2hw3 ) <<  4) | 
      (BIT02(s2hw0)  <<  3) | (BIT02(s2hw1)  <<  2) | (BIT02(s2hw2)  <<  1) | (BIT02(s2hw3 ));

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

GBTFrame::GBTFrame(const GBTFrame& other)
  : mWords(other.mWords)
  , mHalfWords(other.mHalfWords)
  , mAdcClock(other.mAdcClock)
{
}

//short GBTFrame::getHalfWord(short sampa, short halfword, short chan) const
//{
////  sampa %= 3;
////  halfword %= 5;
////  chan %= 2;
//
//  return mHalfWords[sampa][chan][halfword];
//}

void GBTFrame::calculateHalfWords() 
{

  mHalfWords[0][0][0] = (BIT19(mWords[0]) << 4) | (BIT15(mWords[0]) << 3) | (BIT11(mWords[0]) << 2) | (BIT07(mWords[0]) << 1) | BIT03(mWords[0]);
  mHalfWords[0][0][1] = (BIT18(mWords[0]) << 4) | (BIT14(mWords[0]) << 3) | (BIT10(mWords[0]) << 2) | (BIT06(mWords[0]) << 1) | BIT02(mWords[0]);
  mHalfWords[0][0][2] = (BIT17(mWords[0]) << 4) | (BIT13(mWords[0]) << 3) | (BIT09(mWords[0]) << 2) | (BIT05(mWords[0]) << 1) | BIT01(mWords[0]);
  mHalfWords[0][0][3] = (BIT16(mWords[0]) << 4) | (BIT12(mWords[0]) << 3) | (BIT08(mWords[0]) << 2) | (BIT04(mWords[0]) << 1) | BIT00(mWords[0]);

  mHalfWords[0][1][0] = (BIT07(mWords[1]) << 4) | (BIT03(mWords[1]) << 3) | (BIT31(mWords[0]) << 2) | (BIT27(mWords[0]) << 1) | BIT23(mWords[0]);
  mHalfWords[0][1][1] = (BIT06(mWords[1]) << 4) | (BIT02(mWords[1]) << 3) | (BIT30(mWords[0]) << 2) | (BIT26(mWords[0]) << 1) | BIT22(mWords[0]);
  mHalfWords[0][1][2] = (BIT05(mWords[1]) << 4) | (BIT01(mWords[1]) << 3) | (BIT29(mWords[0]) << 2) | (BIT25(mWords[0]) << 1) | BIT21(mWords[0]);
  mHalfWords[0][1][3] = (BIT04(mWords[1]) << 4) | (BIT00(mWords[1]) << 3) | (BIT28(mWords[0]) << 2) | (BIT24(mWords[0]) << 1) | BIT20(mWords[0]);

  mHalfWords[1][0][0] = (BIT31(mWords[1]) << 4) | (BIT27(mWords[1]) << 3) | (BIT23(mWords[1]) << 2) | (BIT19(mWords[1]) << 1) | BIT15(mWords[1]);
  mHalfWords[1][0][1] = (BIT30(mWords[1]) << 4) | (BIT26(mWords[1]) << 3) | (BIT22(mWords[1]) << 2) | (BIT18(mWords[1]) << 1) | BIT14(mWords[1]);
  mHalfWords[1][0][2] = (BIT29(mWords[1]) << 4) | (BIT25(mWords[1]) << 3) | (BIT21(mWords[1]) << 2) | (BIT17(mWords[1]) << 1) | BIT13(mWords[1]);
  mHalfWords[1][0][3] = (BIT28(mWords[1]) << 4) | (BIT24(mWords[1]) << 3) | (BIT20(mWords[1]) << 2) | (BIT16(mWords[1]) << 1) | BIT12(mWords[1]);

  mHalfWords[1][1][0] = (BIT19(mWords[2]) << 4) | (BIT15(mWords[2]) << 3) | (BIT11(mWords[2]) << 2) | (BIT07(mWords[2]) << 1) | BIT03(mWords[2]);
  mHalfWords[1][1][1] = (BIT18(mWords[2]) << 4) | (BIT14(mWords[2]) << 3) | (BIT10(mWords[2]) << 2) | (BIT06(mWords[2]) << 1) | BIT02(mWords[2]);
  mHalfWords[1][1][2] = (BIT17(mWords[2]) << 4) | (BIT13(mWords[2]) << 3) | (BIT09(mWords[2]) << 2) | (BIT05(mWords[2]) << 1) | BIT01(mWords[2]);
  mHalfWords[1][1][3] = (BIT16(mWords[2]) << 4) | (BIT12(mWords[2]) << 3) | (BIT08(mWords[2]) << 2) | (BIT04(mWords[2]) << 1) | BIT00(mWords[2]);

  mHalfWords[2][0][0] = (BIT11(mWords[3]) << 4) | (BIT07(mWords[3]) << 3) | (BIT03(mWords[3]) << 2) | (BIT31(mWords[2]) << 1) | BIT27(mWords[2]);
  mHalfWords[2][0][1] = (BIT10(mWords[3]) << 4) | (BIT06(mWords[3]) << 3) | (BIT02(mWords[3]) << 2) | (BIT30(mWords[2]) << 1) | BIT26(mWords[2]);
  mHalfWords[2][0][2] = (BIT09(mWords[3]) << 4) | (BIT05(mWords[3]) << 3) | (BIT01(mWords[3]) << 2) | (BIT29(mWords[2]) << 1) | BIT25(mWords[2]);
  mHalfWords[2][0][3] = (BIT08(mWords[3]) << 4) | (BIT04(mWords[3]) << 3) | (BIT00(mWords[3]) << 2) | (BIT28(mWords[2]) << 1) | BIT24(mWords[2]);

  mHalfWords[2][1][0] = mHalfWords[2][0][0];
  mHalfWords[2][1][1] = mHalfWords[2][0][1];
  mHalfWords[2][1][2] = mHalfWords[2][0][2];
  mHalfWords[2][1][3] = mHalfWords[2][0][3];

  calculateAdcClock();
}

void GBTFrame::calculateAdcClock()
{
  mAdcClock[0] = (mWords[1] >> 8) & 0xF;
  mAdcClock[1] = (mWords[2] >> 20) & 0xF;
  mAdcClock[2] = (mWords[3] >> 12) & 0xF;

}

void GBTFrame::setData(unsigned word3, unsigned word2, unsigned word1, unsigned word0)
{
  mWords[3] = word3;
  mWords[2] = word2;
  mWords[1] = word1;
  mWords[0] = word0;

  calculateHalfWords();
}

void GBTFrame::setData(short s0hw0l, short s0hw1l, short s0hw2l, short s0hw3l,
                       short s0hw0h, short s0hw1h, short s0hw2h, short s0hw3h,
                       short s1hw0l, short s1hw1l, short s1hw2l, short s1hw3l,
                       short s1hw0h, short s1hw1h, short s1hw2h, short s1hw3h,
                       short s2hw0,  short s2hw1,  short s2hw2,  short s2hw3, 
                       short s0adc,  short s1adc,  short s2adc,  unsigned marker)
{
  mWords[0] =
      (BIT02(s0hw0h) << 31) | (BIT02(s0hw1h) << 30) | (BIT02(s0hw2h) << 29) | (BIT02(s0hw3h) << 28) | 
      (BIT01(s0hw0h) << 27) | (BIT01(s0hw1h) << 26) | (BIT01(s0hw2h) << 25) | (BIT01(s0hw3h) << 24) | 
      (BIT00(s0hw0h) << 23) | (BIT00(s0hw1h) << 22) | (BIT00(s0hw2h) << 21) | (BIT00(s0hw3h) << 20) |
      (BIT04(s0hw0l) << 19) | (BIT04(s0hw1l) << 18) | (BIT04(s0hw2l) << 17) | (BIT04(s0hw3l) << 16) | 
      (BIT03(s0hw0l) << 15) | (BIT03(s0hw1l) << 14) | (BIT03(s0hw2l) << 13) | (BIT03(s0hw3l) << 12) |
      (BIT02(s0hw0l) << 11) | (BIT02(s0hw1l) << 10) | (BIT02(s0hw2l) <<  9) | (BIT02(s0hw3l) <<  8) | 
      (BIT01(s0hw0l) <<  7) | (BIT01(s0hw1l) <<  6) | (BIT01(s0hw2l) <<  5) | (BIT01(s0hw3l) <<  4) | 
      (BIT00(s0hw0l) <<  3) | (BIT00(s0hw1l) <<  2) | (BIT00(s0hw2l) <<  1) | (BIT00(s0hw3l));

  mWords[1] =
      (BIT04(s1hw0l) << 31) | (BIT04(s1hw1l) << 30) | (BIT04(s1hw2l) << 29) | (BIT04(s1hw3l) << 28) | 
      (BIT03(s1hw0l) << 27) | (BIT03(s1hw1l) << 26) | (BIT03(s1hw2l) << 25) | (BIT03(s1hw3l) << 24) |
      (BIT02(s1hw0l) << 23) | (BIT02(s1hw1l) << 22) | (BIT02(s1hw2l) << 21) | (BIT02(s1hw3l) << 20) | 
      (BIT01(s1hw0l) << 19) | (BIT01(s1hw1l) << 18) | (BIT01(s1hw2l) << 17) | (BIT01(s1hw3l) << 16) | 
      (BIT00(s1hw0l) << 15) | (BIT00(s1hw1l) << 14) | (BIT00(s1hw2l) << 13) | (BIT00(s1hw3l) << 12) |
      (BIT03(s0adc)  << 11) | (BIT02(s0adc)  << 10) | (BIT01(s0adc)  <<  9) | (BIT00(s0adc)  <<  8) |
      (BIT04(s0hw0h) <<  7) | (BIT04(s0hw1h) <<  6) | (BIT04(s0hw2h) <<  5) | (BIT04(s0hw3h) <<  4) | 
      (BIT03(s0hw0h) <<  3) | (BIT03(s0hw1h) <<  2) | (BIT03(s0hw2h) <<  1) | (BIT03(s0hw3h));

  mWords[2] =
      (BIT01(s2hw0)  << 31) | (BIT01(s2hw1)  << 30) | (BIT01(s2hw2)  << 29) | (BIT01(s2hw3)  << 28) | 
      (BIT00(s2hw0)  << 27) | (BIT00(s2hw1)  << 26) | (BIT00(s2hw2)  << 25) | (BIT00(s2hw3)  << 24) | 
      (BIT03(s1adc)  << 23) | (BIT02(s1adc)  << 22) | (BIT01(s1adc)  << 21) | (BIT00(s1adc)  << 20) |
      (BIT04(s1hw0h) << 19) | (BIT04(s1hw1h) << 18) | (BIT04(s1hw2h) << 17) | (BIT04(s1hw3h) << 16) | 
      (BIT03(s1hw0h) << 15) | (BIT03(s1hw1h) << 14) | (BIT03(s1hw2h) << 13) | (BIT03(s1hw3h) << 12) |
      (BIT02(s1hw0h) << 11) | (BIT02(s1hw1h) << 10) | (BIT02(s1hw2h) <<  9) | (BIT02(s1hw3h) <<  8) | 
      (BIT01(s1hw0h) <<  7) | (BIT01(s1hw1h) <<  6) | (BIT01(s1hw2h) <<  5) | (BIT01(s1hw3h) <<  4) | 
      (BIT00(s1hw0h) <<  3) | (BIT00(s1hw1h) <<  2) | (BIT00(s1hw2h) <<  1) | (BIT00(s1hw3h));

  mWords[3] =
      (BIT15(marker) << 31) | (BIT14(marker) << 30) | (BIT13(marker) << 29) | (BIT12(marker) << 28) |
      (BIT11(marker) << 27) | (BIT10(marker) << 26) | (BIT09(marker) << 25) | (BIT08(marker) << 24) | 
      (BIT07(marker) << 23) | (BIT06(marker) << 22) | (BIT05(marker) << 21) | (BIT04(marker) << 20) | 
      (BIT03(marker) << 19) | (BIT02(marker) << 18) | (BIT01(marker) << 17) | (BIT00(marker) << 16) |
      (BIT03(s2adc)  << 15) | (BIT02(s2adc)  << 14) | (BIT01(s2adc)  << 13) | (BIT00(s2adc ) << 12) |
      (BIT04(s2hw0)  << 11) | (BIT04(s2hw1)  << 10) | (BIT04(s2hw2)  <<  9) | (BIT04(s2hw3 ) <<  8) | 
      (BIT03(s2hw0)  <<  7) | (BIT03(s2hw1)  <<  6) | (BIT03(s2hw2)  <<  5) | (BIT03(s2hw3 ) <<  4) | 
      (BIT02(s2hw0)  <<  3) | (BIT02(s2hw1)  <<  2) | (BIT02(s2hw2)  <<  1) | (BIT02(s2hw3 ));

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
    case  0: mWords[1] = (mWords[1] & 0xFFFFF0FF) | ((clock & 0xF) <<  8); break;
    case  1: mWords[2] = (mWords[2] & 0xFF0FFFFF) | ((clock & 0xF) << 20); break;
    case  2: mWords[3] = (mWords[3] & 0xFFFF0FFF) | ((clock & 0xF) << 12); break;
    case -1: mWords[1] = (mWords[1] & 0xFFFFF0FF) | ((clock & 0xF) <<  8);
             mWords[2] = (mWords[2] & 0xFF0FFFFF) | ((clock & 0xF) << 20);
             mWords[3] = (mWords[3] & 0xFFFF0FFF) | ((clock & 0xF) << 12); break;
    default: std::cout << "don't know SAMPA " << sampa << std::endl; break; 
  }
  calculateAdcClock();
}

void GBTFrame::getGBTFrame(unsigned& word3, unsigned& word2, unsigned& word1, unsigned& word0) const
{
  word3 = mWords[3];
  word2 = mWords[2];
  word1 = mWords[1];
  word0 = mWords[0];
}

std::ostream& GBTFrame::Print(std::ostream& output) const
{
  output << "0x" << std::hex 
    << std::setfill('0') << std::setw(8) << mWords[3]
    << std::setfill('0') << std::setw(8) << mWords[2]
    << std::setfill('0') << std::setw(8) << mWords[1]
    << std::setfill('0') << std::setw(8) << mWords[0]
    << std::dec;
  return output;
}

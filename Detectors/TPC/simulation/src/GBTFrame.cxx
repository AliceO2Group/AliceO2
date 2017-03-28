/// \file GBTFrame.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/GBTFrame.h"
#define BIT00  (uint32_t) (0x1 << 0 )                                               
#define BIT01  (uint32_t) (0x1 << 1 )                                               
#define BIT02  (uint32_t) (0x1 << 2 )                                               
#define BIT03  (uint32_t) (0x1 << 3 )                                               
#define BIT04  (uint32_t) (0x1 << 4 )                                               
#define BIT05  (uint32_t) (0x1 << 5 )                                               
#define BIT06  (uint32_t) (0x1 << 6 )                                               
#define BIT07  (uint32_t) (0x1 << 7 )                                               
#define BIT08  (uint32_t) (0x1 << 8 )                                               
#define BIT09  (uint32_t) (0x1 << 9 )                                               
#define BIT10  (uint32_t) (0x1 << 10 )                                              
#define BIT11  (uint32_t) (0x1 << 11 )                                              
#define BIT12  (uint32_t) (0x1 << 12 )                                              
#define BIT13  (uint32_t) (0x1 << 13 )                                              
#define BIT14  (uint32_t) (0x1 << 14 )                                              
#define BIT15  (uint32_t) (0x1 << 15 )                                              
#define BIT16  (uint32_t) (0x1 << 16 )                                              
#define BIT17  (uint32_t) (0x1 << 17 )                                              
#define BIT18  (uint32_t) (0x1 << 18 )                                              
#define BIT19  (uint32_t) (0x1 << 19 )                                              
#define BIT20  (uint32_t) (0x1 << 20 )                                              
#define BIT21  (uint32_t) (0x1 << 21 )                                              
#define BIT22  (uint32_t) (0x1 << 22 )                                              
#define BIT23  (uint32_t) (0x1 << 23 )                                              
#define BIT24  (uint32_t) (0x1 << 24 )                                              
#define BIT25  (uint32_t) (0x1 << 25 )                                              
#define BIT26  (uint32_t) (0x1 << 26 )                                              
#define BIT27  (uint32_t) (0x1 << 27 )                                              
#define BIT28  (uint32_t) (0x1 << 28 )                                              
#define BIT29  (uint32_t) (0x1 << 29 )                                              
#define BIT30  (uint32_t) (0x1 << 30 )                                              
#define BIT31  (uint32_t) (0x1 << 31 )

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
      ((s0hw0h & BIT02) << 29) | ((s0hw1h & BIT02) << 28) | ((s0hw2h & BIT02) << 27) | ((s0hw3h & BIT02) << 26) | 
      ((s0hw0h & BIT01) << 26) | ((s0hw1h & BIT01) << 25) | ((s0hw2h & BIT01) << 24) | ((s0hw3h & BIT01) << 23) | 
      ((s0hw0h & BIT00) << 23) | ((s0hw1h & BIT00) << 22) | ((s0hw2h & BIT00) << 21) | ((s0hw3h & BIT00) << 20) |
      ((s0hw0l & BIT04) << 15) | ((s0hw1l & BIT04) << 14) | ((s0hw2l & BIT04) << 13) | ((s0hw3l & BIT04) << 12) | 
      ((s0hw0l & BIT03) << 12) | ((s0hw1l & BIT03) << 11) | ((s0hw2l & BIT03) << 10) | ((s0hw3l & BIT03) << 9) |
      ((s0hw0l & BIT02) <<  9) | ((s0hw1l & BIT02) <<  8) | ((s0hw2l & BIT02) <<  7) | ((s0hw3l & BIT02) << 6) | 
      ((s0hw0l & BIT01) <<  6) | ((s0hw1l & BIT01) <<  5) | ((s0hw2l & BIT01) <<  4) | ((s0hw3l & BIT01) << 3) | 
      ((s0hw0l & BIT00) <<  3) | ((s0hw1l & BIT00) <<  2) | ((s0hw2l & BIT00) <<  1) | (s0hw3l & BIT00);

//  mWords[0] = combineBits(std::vector<bool>{
//      getBit(s0hw0h,2), getBit(s0hw1h,2), getBit(s0hw2h,2), getBit(s0hw3h,2), 
//      getBit(s0hw0h,1), getBit(s0hw1h,1), getBit(s0hw2h,1), getBit(s0hw3h,1), 
//      getBit(s0hw0h,0), getBit(s0hw1h,0), getBit(s0hw2h,0), getBit(s0hw3h,0),
//      getBit(s0hw0l,4), getBit(s0hw1l,4), getBit(s0hw2l,4), getBit(s0hw3l,4), 
//      getBit(s0hw0l,3), getBit(s0hw1l,3), getBit(s0hw2l,3), getBit(s0hw3l,3),
//      getBit(s0hw0l,2), getBit(s0hw1l,2), getBit(s0hw2l,2), getBit(s0hw3l,2), 
//      getBit(s0hw0l,1), getBit(s0hw1l,1), getBit(s0hw2l,1), getBit(s0hw3l,1), 
//      getBit(s0hw0l,0), getBit(s0hw1l,0), getBit(s0hw2l,0), getBit(s0hw3l,0)
//      });

  mWords[1] = combineBits(std::vector<bool>{
      getBit(s1hw0l,4), getBit(s1hw1l,4), getBit(s1hw2l,4), getBit(s1hw3l,4), 
      getBit(s1hw0l,3), getBit(s1hw1l,3), getBit(s1hw2l,3), getBit(s1hw3l,3),
      getBit(s1hw0l,2), getBit(s1hw1l,2), getBit(s1hw2l,2), getBit(s1hw3l,2), 
      getBit(s1hw0l,1), getBit(s1hw1l,1), getBit(s1hw2l,1), getBit(s1hw3l,1), 
      getBit(s1hw0l,0), getBit(s1hw1l,0), getBit(s1hw2l,0), getBit(s1hw3l,0),
      getBit(s0adc,3),  getBit(s0adc,2),  getBit(s0adc,1),  getBit(s0adc,0),
      getBit(s0hw0h,4), getBit(s0hw1h,4), getBit(s0hw2h,4), getBit(s0hw3h,4), 
      getBit(s0hw0h,3), getBit(s0hw1h,3), getBit(s0hw2h,3), getBit(s0hw3h,3)
      });

  mWords[2] = combineBits(std::vector<bool>{
      getBit(s2hw0,1), getBit(s2hw1,1), getBit(s2hw2,1), getBit(s2hw3,1), 
      getBit(s2hw0,0), getBit(s2hw1,0), getBit(s2hw2,0), getBit(s2hw3,0), 
      getBit(s1adc,3), getBit(s1adc,2), getBit(s1adc,1), getBit(s1adc,0),
      getBit(s1hw0h,4), getBit(s1hw1h,4), getBit(s1hw2h,4), getBit(s1hw3h,4), 
      getBit(s1hw0h,3), getBit(s1hw1h,3), getBit(s1hw2h,3), getBit(s1hw3h,3),
      getBit(s1hw0h,2), getBit(s1hw1h,2), getBit(s1hw2h,2), getBit(s1hw3h,2), 
      getBit(s1hw0h,1), getBit(s1hw1h,1), getBit(s1hw2h,1), getBit(s1hw3h,1), 
      getBit(s1hw0h,0), getBit(s1hw1h,0), getBit(s1hw2h,0), getBit(s1hw3h,0)
      });

  mWords[3] = combineBits(std::vector<bool>{
      getBit(marker,15), getBit(marker,14), getBit(marker,13), getBit(marker,12),
      getBit(marker,11), getBit(marker,10), getBit(marker, 9), getBit(marker, 8), 
      getBit(marker, 7), getBit(marker, 6), getBit(marker, 5), getBit(marker, 4), 
      getBit(marker, 3), getBit(marker, 2), getBit(marker, 1), getBit(marker, 0),
      getBit(s2adc,3), getBit(s2adc,2), getBit(s2adc,1), getBit(s2adc,0),
      getBit(s2hw0,4), getBit(s2hw1,4), getBit(s2hw2,4), getBit(s2hw3,4), 
      getBit(s2hw0,3), getBit(s2hw1,3), getBit(s2hw2,3), getBit(s2hw3,3), 
      getBit(s2hw0,2), getBit(s2hw1,2), getBit(s2hw2,2), getBit(s2hw3,2) 
      });

  calculateHalfWords();
}

GBTFrame::GBTFrame(const GBTFrame& other)
  : mWords(other.mWords)
  , mHalfWords(other.mHalfWords)
  , mAdcClock(other.mAdcClock)
{
}

GBTFrame::~GBTFrame()
{}

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

  mHalfWords[0][0][0] = ((mWords[0] & BIT19) >> 15) | ((mWords[0] & BIT15) >> 12) | ((mWords[0] & BIT11) >> 9) | ((mWords[0] & BIT07) >> 6) | ((mWords[0] & BIT03) >> 3);
  mHalfWords[0][0][1] = ((mWords[0] & BIT18) >> 14) | ((mWords[0] & BIT14) >> 11) | ((mWords[0] & BIT10) >> 8) | ((mWords[0] & BIT06) >> 5) | ((mWords[0] & BIT02) >> 2);
  mHalfWords[0][0][2] = ((mWords[0] & BIT17) >> 13) | ((mWords[0] & BIT13) >> 10) | ((mWords[0] & BIT09) >> 7) | ((mWords[0] & BIT05) >> 4) | ((mWords[0] & BIT01) >> 1);
  mHalfWords[0][0][3] = ((mWords[0] & BIT16) >> 12) | ((mWords[0] & BIT12) >> 9) | ((mWords[0] & BIT08) >> 6) | ((mWords[0] & BIT04) >> 3) | ((mWords[0] & BIT00) >> 0);

  mHalfWords[0][1][0] = ((mWords[1] & BIT07) >> 3) | ((mWords[1] & BIT03) >> 0) | ((mWords[0] & BIT31) >> 29) | ((mWords[0] & BIT27) >> 26) | ((mWords[0] & BIT23) >> 23);
  mHalfWords[0][1][1] = ((mWords[1] & BIT06) >> 2) | ((mWords[1] & BIT02) << 1) | ((mWords[0] & BIT30) >> 28) | ((mWords[0] & BIT26) >> 25) | ((mWords[0] & BIT22) >> 22);
  mHalfWords[0][1][2] = ((mWords[1] & BIT05) >> 1) | ((mWords[1] & BIT01) << 2) | ((mWords[0] & BIT29) >> 27) | ((mWords[0] & BIT25) >> 24) | ((mWords[0] & BIT21) >> 21);
  mHalfWords[0][1][3] = ((mWords[1] & BIT04) >> 0) | ((mWords[1] & BIT00) << 3) | ((mWords[0] & BIT28) >> 26) | ((mWords[0] & BIT24) >> 23) | ((mWords[0] & BIT20) >> 20);
                     
  mHalfWords[1][0][0] = ((mWords[1] & BIT31) >> 27) | ((mWords[1] & BIT27) >> 24) | ((mWords[1] & BIT23) >> 21) | ((mWords[1] & BIT19) >> 18) | ((mWords[1] & BIT15) >> 15);
  mHalfWords[1][0][1] = ((mWords[1] & BIT30) >> 26) | ((mWords[1] & BIT26) >> 23) | ((mWords[1] & BIT22) >> 20) | ((mWords[1] & BIT18) >> 17) | ((mWords[1] & BIT14) >> 14);
  mHalfWords[1][0][2] = ((mWords[1] & BIT29) >> 25) | ((mWords[1] & BIT25) >> 22) | ((mWords[1] & BIT21) >> 19) | ((mWords[1] & BIT17) >> 16) | ((mWords[1] & BIT13) >> 13);
  mHalfWords[1][0][3] = ((mWords[1] & BIT28) >> 24) | ((mWords[1] & BIT24) >> 21) | ((mWords[1] & BIT20) >> 18) | ((mWords[1] & BIT16) >> 15) | ((mWords[1] & BIT12) >> 12);
                     
  mHalfWords[1][1][0] = ((mWords[2] & BIT19) >> 15) | ((mWords[2] & BIT15) >> 12) | ((mWords[2] & BIT11) >> 9) | ((mWords[2] & BIT07) >> 6) | ((mWords[2] & BIT03) >> 3);
  mHalfWords[1][1][1] = ((mWords[2] & BIT18) >> 14) | ((mWords[2] & BIT14) >> 11) | ((mWords[2] & BIT10) >> 8) | ((mWords[2] & BIT06) >> 5) | ((mWords[2] & BIT02) >> 2);
  mHalfWords[1][1][2] = ((mWords[2] & BIT17) >> 13) | ((mWords[2] & BIT13) >> 10) | ((mWords[2] & BIT09) >> 7) | ((mWords[2] & BIT05) >> 4) | ((mWords[2] & BIT01) >> 1);
  mHalfWords[1][1][3] = ((mWords[2] & BIT16) >> 12) | ((mWords[2] & BIT12) >> 9) | ((mWords[2] & BIT08) >> 6) | ((mWords[2] & BIT04) >> 3) | ((mWords[2] & BIT00) >> 0);
                     
  mHalfWords[2][0][0] = ((mWords[3] & BIT11) >> 7) | ((mWords[3] & BIT07) >> 4) | ((mWords[3] & BIT03) >> 1) | ((mWords[2] & BIT31) >> 30) | ((mWords[2] & BIT27) >> 27);
  mHalfWords[2][0][1] = ((mWords[3] & BIT10) >> 6) | ((mWords[3] & BIT06) >> 3) | ((mWords[3] & BIT02) >> 0) | ((mWords[2] & BIT30) >> 29) | ((mWords[2] & BIT26) >> 26);
  mHalfWords[2][0][2] = ((mWords[3] & BIT09) >> 5) | ((mWords[3] & BIT05) >> 2) | ((mWords[3] & BIT01) << 1) | ((mWords[2] & BIT29) >> 28) | ((mWords[2] & BIT25) >> 25);
  mHalfWords[2][0][3] = ((mWords[3] & BIT08) >> 4) | ((mWords[3] & BIT04) >> 1) | ((mWords[3] & BIT00) << 2) | ((mWords[2] & BIT28) >> 27) | ((mWords[2] & BIT24) >> 24);

//  mHalfWords[0][0][0] = combineBitsOfFrame(19,15,11,7,3);
//  mHalfWords[0][0][1] = combineBitsOfFrame(18,14,10,6,2);
//  mHalfWords[0][0][2] = combineBitsOfFrame(17,13, 9,5,1);
//  mHalfWords[0][0][3] = combineBitsOfFrame(16,12, 8,4,0);

//  mHalfWords[0][1][0] = combineBitsOfFrame(39,35,31,27,23);
//  mHalfWords[0][1][1] = combineBitsOfFrame(38,34,30,26,22);
//  mHalfWords[0][1][2] = combineBitsOfFrame(37,33,29,25,21);
//  mHalfWords[0][1][3] = combineBitsOfFrame(36,32,28,24,20);
//
//  mHalfWords[1][0][0] = combineBitsOfFrame(63,59,55,51,47);
//  mHalfWords[1][0][1] = combineBitsOfFrame(62,58,54,50,46);
//  mHalfWords[1][0][2] = combineBitsOfFrame(61,57,53,49,45);
//  mHalfWords[1][0][3] = combineBitsOfFrame(60,56,52,48,44);
//
//  mHalfWords[1][1][0] = combineBitsOfFrame(83,79,75,71,67);
//  mHalfWords[1][1][1] = combineBitsOfFrame(82,78,74,70,66);
//  mHalfWords[1][1][2] = combineBitsOfFrame(81,77,73,69,65);
//  mHalfWords[1][1][3] = combineBitsOfFrame(80,76,72,68,64);
//
//  mHalfWords[2][0][0] = combineBitsOfFrame(107,103,99,95,91);
//  mHalfWords[2][0][1] = combineBitsOfFrame(106,102,98,94,90);
//  mHalfWords[2][0][2] = combineBitsOfFrame(105,101,97,93,89);
//  mHalfWords[2][0][3] = combineBitsOfFrame(104,100,96,92,88);

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

void GBTFrame::setData(unsigned& word3, unsigned& word2, unsigned& word1, unsigned& word0)
{
  mWords[3] = word3;
  mWords[2] = word2;
  mWords[1] = word1;
  mWords[0] = word0;

  calculateHalfWords();
}

void GBTFrame::setData(short& s0hw0l, short& s0hw1l, short& s0hw2l, short& s0hw3l,
                       short& s0hw0h, short& s0hw1h, short& s0hw2h, short& s0hw3h,
                       short& s1hw0l, short& s1hw1l, short& s1hw2l, short& s1hw3l,
                       short& s1hw0h, short& s1hw1h, short& s1hw2h, short& s1hw3h,
                       short& s2hw0,  short& s2hw1,  short& s2hw2,  short& s2hw3, 
                       short& s0adc,  short& s1adc,  short& s2adc,  unsigned marker)
{
  mWords[0] = combineBits(std::vector<bool>{
      getBit(s0hw0h,2), getBit(s0hw1h,2), getBit(s0hw2h,2), getBit(s0hw3h,2), 
      getBit(s0hw0h,1), getBit(s0hw1h,1), getBit(s0hw2h,1), getBit(s0hw3h,1), 
      getBit(s0hw0h,0), getBit(s0hw1h,0), getBit(s0hw2h,0), getBit(s0hw3h,0),
      getBit(s0hw0l,4), getBit(s0hw1l,4), getBit(s0hw2l,4), getBit(s0hw3l,4), 
      getBit(s0hw0l,3), getBit(s0hw1l,3), getBit(s0hw2l,3), getBit(s0hw3l,3),
      getBit(s0hw0l,2), getBit(s0hw1l,2), getBit(s0hw2l,2), getBit(s0hw3l,2), 
      getBit(s0hw0l,1), getBit(s0hw1l,1), getBit(s0hw2l,1), getBit(s0hw3l,1), 
      getBit(s0hw0l,0), getBit(s0hw1l,0), getBit(s0hw2l,0), getBit(s0hw3l,0)
      });

  mWords[1] = combineBits(std::vector<bool>{
      getBit(s1hw0l,4), getBit(s1hw1l,4), getBit(s1hw2l,4), getBit(s1hw3l,4), 
      getBit(s1hw0l,3), getBit(s1hw1l,3), getBit(s1hw2l,3), getBit(s1hw3l,3),
      getBit(s1hw0l,2), getBit(s1hw1l,2), getBit(s1hw2l,2), getBit(s1hw3l,2), 
      getBit(s1hw0l,1), getBit(s1hw1l,1), getBit(s1hw2l,1), getBit(s1hw3l,1), 
      getBit(s1hw0l,0), getBit(s1hw1l,0), getBit(s1hw2l,0), getBit(s1hw3l,0),
      getBit(s0adc,3),  getBit(s0adc,2),  getBit(s0adc,1),  getBit(s0adc,0),
      getBit(s0hw0h,4), getBit(s0hw1h,4), getBit(s0hw2h,4), getBit(s0hw3h,4), 
      getBit(s0hw0h,3), getBit(s0hw1h,3), getBit(s0hw2h,3), getBit(s0hw3h,3)
      });

  mWords[2] = combineBits(std::vector<bool>{
      getBit(s2hw0,1), getBit(s2hw1,1), getBit(s2hw2,1), getBit(s2hw3,1), 
      getBit(s2hw0,0), getBit(s2hw1,0), getBit(s2hw2,0), getBit(s2hw3,0), 
      getBit(s1adc,3), getBit(s1adc,2), getBit(s1adc,1), getBit(s1adc,0),
      getBit(s1hw0h,4), getBit(s1hw1h,4), getBit(s1hw2h,4), getBit(s1hw3h,4), 
      getBit(s1hw0h,3), getBit(s1hw1h,3), getBit(s1hw2h,3), getBit(s1hw3h,3),
      getBit(s1hw0h,2), getBit(s1hw1h,2), getBit(s1hw2h,2), getBit(s1hw3h,2), 
      getBit(s1hw0h,1), getBit(s1hw1h,1), getBit(s1hw2h,1), getBit(s1hw3h,1), 
      getBit(s1hw0h,0), getBit(s1hw1h,0), getBit(s1hw2h,0), getBit(s1hw3h,0)
      });

  mWords[3] = combineBits(std::vector<bool>{
      getBit(marker,15), getBit(marker,14), getBit(marker,13), getBit(marker,12),
      getBit(marker,11), getBit(marker,10), getBit(marker, 9), getBit(marker, 8), 
      getBit(marker, 7), getBit(marker, 6), getBit(marker, 5), getBit(marker, 4), 
      getBit(marker, 3), getBit(marker, 2), getBit(marker, 1), getBit(marker, 0),
      getBit(s2adc,3), getBit(s2adc,2), getBit(s2adc,1), getBit(s2adc,0),
      getBit(s2hw0,4), getBit(s2hw1,4), getBit(s2hw2,4), getBit(s2hw3,4), 
      getBit(s2hw0,3), getBit(s2hw1,3), getBit(s2hw2,3), getBit(s2hw3,3), 
      getBit(s2hw0,2), getBit(s2hw1,2), getBit(s2hw2,2), getBit(s2hw3,2) 
      });

  calculateHalfWords();
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

inline bool GBTFrame::getBit(unsigned word, unsigned lsb) const
{
  return (word >> lsb) & 0x1;
}

short GBTFrame::getBits(short word, unsigned width, unsigned lsb) const
{
  return ((width >= 8) ? word : (word >> lsb) & (((1 << width) - 1)));
}

unsigned GBTFrame::getBits(unsigned word, unsigned width, unsigned lsb) const
{
  return (word >> lsb) & (((1 << width) - 1));
}

short GBTFrame::combineBitsOfFrame(short bit0, short bit1, short bit2, short bit3, short bit4) const
{
//
//  ">> 5"      is equivalent to "/ 32"
//  "& 0x1F"    is equivalent to "% 32"
  return    ( getBit(mWords[bit0 >> 5],bit0 & 0x1F) << 4 ) |
            ( getBit(mWords[bit1 >> 5],bit1 & 0x1F) << 3 ) |
            ( getBit(mWords[bit2 >> 5],bit2 & 0x1F) << 2 ) |
            ( getBit(mWords[bit3 >> 5],bit3 & 0x1F) << 1 ) |
            ( getBit(mWords[bit4 >> 5],bit4 & 0x1F) );
}

unsigned GBTFrame::combineBits(std::vector<bool> bits) const
{
  unsigned res = 0;
  for (std::vector<bool>::iterator it = bits.begin(); it != bits.end(); it++) {
    res = (res << 1) + (unsigned)*it;
  }
  return res;
}

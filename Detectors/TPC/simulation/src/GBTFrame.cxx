/// \file GBTFrame.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/GBTFrame.h"

using namespace AliceO2::TPC;

GBTFrame::GBTFrame()
  : GBTFrame(0,0,0,0)
{}

GBTFrame::GBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0)
  : mWords(4)
  , mAdcClock(3)
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
  : mWords(4)
  , mAdcClock(3)
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

  mHalfWords[0][0][0] = combineBitsOfFrame(19,15,11,7,3);
  mHalfWords[0][0][1] = combineBitsOfFrame(18,14,10,6,2);
  mHalfWords[0][0][2] = combineBitsOfFrame(17,13, 9,5,1);
  mHalfWords[0][0][3] = combineBitsOfFrame(16,12, 8,4,0);

  mHalfWords[0][1][0] = combineBitsOfFrame(39,35,31,27,23);
  mHalfWords[0][1][1] = combineBitsOfFrame(38,34,30,26,22);
  mHalfWords[0][1][2] = combineBitsOfFrame(37,33,29,25,21);
  mHalfWords[0][1][3] = combineBitsOfFrame(36,32,28,24,20);

  mHalfWords[1][0][0] = combineBitsOfFrame(63,59,55,51,47);
  mHalfWords[1][0][1] = combineBitsOfFrame(62,58,54,50,46);
  mHalfWords[1][0][2] = combineBitsOfFrame(61,57,53,49,45);
  mHalfWords[1][0][3] = combineBitsOfFrame(60,56,52,48,44);

  mHalfWords[1][1][0] = combineBitsOfFrame(83,79,75,71,67);
  mHalfWords[1][1][1] = combineBitsOfFrame(82,78,74,70,66);
  mHalfWords[1][1][2] = combineBitsOfFrame(81,77,73,69,65);
  mHalfWords[1][1][3] = combineBitsOfFrame(80,76,72,68,64);

  mHalfWords[2][0][0] = combineBitsOfFrame(107,103,99,95,91);
  mHalfWords[2][0][1] = combineBitsOfFrame(106,102,98,94,90);
  mHalfWords[2][0][2] = combineBitsOfFrame(105,101,97,93,89);
  mHalfWords[2][0][3] = combineBitsOfFrame(104,100,96,92,88);

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

void GBTFrame::setData(short s0hw0l, short s0hw1l, short s0hw2l, short s0hw3l,
                       short s0hw0h, short s0hw1h, short s0hw2h, short s0hw3h,
                       short s1hw0l, short s1hw1l, short s1hw2l, short s1hw3l,
                       short s1hw0h, short s1hw1h, short s1hw2h, short s1hw3h,
                       short s2hw0, short s2hw1, short s2hw2, short s2hw3, 
                       short s0adc, short s1adc, short s2adc, unsigned marker)
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

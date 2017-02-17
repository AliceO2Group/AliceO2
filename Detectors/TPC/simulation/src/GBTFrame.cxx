/// \file GBTFrame.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/GBTFrame.h"

ClassImp(AliceO2::TPC::GBTFrame)

using namespace AliceO2::TPC;

GBTFrame::GBTFrame()
  : TObject()
{
  mWords[3] = 0;
  mWords[2] = 0;
  mWords[1] = 0;
  mWords[0] = 0;
}

GBTFrame::GBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0)
  : TObject()
{
  mWords[3] = word3;
  mWords[2] = word2;
  mWords[1] = word1;
  mWords[0] = word0;
}

GBTFrame::GBTFrame(char s0hw0l, char s0hw1l, char s0hw2l, char s0hw3l,
                   char s0hw0h, char s0hw1h, char s0hw2h, char s0hw3h,
                   char s1hw0l, char s1hw1l, char s1hw2l, char s1hw3l,
                   char s1hw0h, char s1hw1h, char s1hw2h, char s1hw3h,
                   char s2hw0, char s2hw1, char s2hw2, char s2hw3, 
                   char s0adc, char s1adc, char s2adc, unsigned marker)
  : TObject()
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

}

GBTFrame::GBTFrame(const GBTFrame& other)
  : TObject(other)
{
  mWords[3] = other.mWords[3];
  mWords[2] = other.mWords[2];
  mWords[1] = other.mWords[1];
  mWords[0] = other.mWords[0];
}

GBTFrame::~GBTFrame()
{}

char GBTFrame::getHalfWord(char sampa, char hw, char chan)
{
  sampa %= 3;
  hw %= 5;
  chan %= 2;

  char result = 0;
  if (sampa == 0 && chan == 0) {
    switch(hw) {
      case 0: return combineBitsOfFrame(std::vector<char>{19,15,11,7,3});
      case 1: return combineBitsOfFrame(std::vector<char>{18,14,10,6,2});
      case 2: return combineBitsOfFrame(std::vector<char>{17,13, 9,5,1});
      case 3: return combineBitsOfFrame(std::vector<char>{16,12, 8,4,0});
    }
  }
  if (sampa == 0 && chan == 1) {
    switch(hw) {
      case 0: return combineBitsOfFrame(std::vector<char>{39,35,31,27,23});
      case 1: return combineBitsOfFrame(std::vector<char>{38,34,30,26,22});
      case 2: return combineBitsOfFrame(std::vector<char>{37,33,29,25,21});
      case 3: return combineBitsOfFrame(std::vector<char>{36,32,28,24,20});
    }
  }
  if (sampa == 1 && chan == 0) {
    switch(hw) {
      case 0: return combineBitsOfFrame(std::vector<char>{63,59,55,51,47});
      case 1: return combineBitsOfFrame(std::vector<char>{62,58,54,50,46});
      case 2: return combineBitsOfFrame(std::vector<char>{61,57,53,49,45});
      case 3: return combineBitsOfFrame(std::vector<char>{60,56,52,48,44});
    }
  }
  if (sampa == 1 && chan == 1) {
    switch(hw) {
      case 0: return combineBitsOfFrame(std::vector<char>{83,79,75,71,67});
      case 1: return combineBitsOfFrame(std::vector<char>{82,78,74,70,66});
      case 2: return combineBitsOfFrame(std::vector<char>{81,77,73,69,65});
      case 3: return combineBitsOfFrame(std::vector<char>{80,76,72,68,64});
    }
  }
  if (sampa == 2) {
    switch(hw) {
      case 0: return combineBitsOfFrame(std::vector<char>{107,103,99,95,91});
      case 1: return combineBitsOfFrame(std::vector<char>{106,102,98,94,90});
      case 2: return combineBitsOfFrame(std::vector<char>{105,101,97,93,89});
      case 3: return combineBitsOfFrame(std::vector<char>{104,100,96,92,88});
    }
  }

  return 0;
}

char GBTFrame::getAdcClock(char sampa)
{
  sampa = sampa % 3;

  switch (sampa) {
    case 0: return (mWords[1] >> 8) & 0xF;
    case 1: return (mWords[2] >> 20) & 0xF;
    case 2: return (mWords[3] >> 12) & 0xF;
  }
}

void GBTFrame::getGBTFrame(unsigned& word3, unsigned& word2, unsigned& word1, unsigned& word0)
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

bool GBTFrame::getBit(unsigned word, unsigned lsb)
{
  return (word >> lsb) & 0x1;
}

char GBTFrame::getBits(char word, unsigned width, unsigned lsb)
{
  return ((width >= 8) ? word : (word >> lsb) & (((1 << width) - 1)));
}

unsigned GBTFrame::getBits(unsigned word, unsigned width, unsigned lsb)
{
  return ((width >= 32) ? word : (word >> lsb) & (((1 << width) - 1)));
}

unsigned GBTFrame::combineBitsOfFrame(std::vector<char> bits)
{
  unsigned res = 0;
  for (std::vector<char>::iterator it = bits.begin(); it != bits.end(); it++) {
    res = (res << 1) + getBits(mWords[(*it)/32],1,*it);
  }
  return res;
}
unsigned GBTFrame::combineBits(std::vector<bool> bits)
{
  unsigned res = 0;
  for (std::vector<bool>::iterator it = bits.begin(); it != bits.end(); it++) {
    res = (res << 1) + (unsigned)*it;
  }
  return res;
}

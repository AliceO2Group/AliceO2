/// \file GBTFrameContainer.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/GBTFrameContainer.h"

using namespace AliceO2::TPC;

GBTFrameContainer::GBTFrameContainer()
{
  mAdcClock = new AdcClockMonitor [3];
}

GBTFrameContainer::GBTFrameContainer(int amount)
{
  mAdcClock = new AdcClockMonitor [3];
  mGBTFrames.reserve(amount);
}

GBTFrameContainer::~GBTFrameContainer()
{
  delete[] mAdcClock;
}

void GBTFrameContainer::addGBTFrame(GBTFrame& frame) 
{
  mGBTFrames.emplace_back(frame);
  processLastFrame();
}

void GBTFrameContainer::addGBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0)
{
  mGBTFrames.emplace_back(word3, word2, word1, word0);
  processLastFrame();
}

void GBTFrameContainer::addGBTFrame(char s0hw0l, char s0hw1l, char s0hw2l, char s0hw3l,
                                    char s0hw0h, char s0hw1h, char s0hw2h, char s0hw3h,
                                    char s1hw0l, char s1hw1l, char s1hw2l, char s1hw3l,
                                    char s1hw0h, char s1hw1h, char s1hw2h, char s1hw3h,
                                    char s2hw0, char s2hw1, char s2hw2, char s2hw3, 
                                    char s0adc, char s1adc, char s2adc, unsigned marker)
{
  mGBTFrames.emplace_back(s0hw0l, s0hw1l, s0hw2l, s0hw3l, s0hw0h, s0hw1h, s0hw2h, s0hw3h,
                          s1hw0l, s1hw1l, s1hw2l, s1hw3l, s1hw0h, s1hw1h, s1hw2h, s1hw3h,
                          s2hw0, s2hw1, s2hw2, s2hw3, s0adc, s1adc, s2adc, marker);
  processLastFrame();
}


void GBTFrameContainer::fillOutputContainer(TClonesArray* output)
{

  TClonesArray &clref = *output;
  for (auto &aGBTFrame : mGBTFrames) {
    new(clref[clref.GetEntriesFast()]) GBTFrame(aGBTFrame);
  }
}

void GBTFrameContainer::processLastFrame()
{
  if (mAdcClock[0].addSequence(mGBTFrames.back().getAdcClock(0))) 
    LOG(WARNING) << "ADC clock error of SAMPA 0 in GBT Frame " << getNentries() << FairLogger::endl;
  if (mAdcClock[1].addSequence(mGBTFrames.back().getAdcClock(1))) 
    LOG(WARNING) << "ADC clock error of SAMPA 1 in GBT Frame " << getNentries() << FairLogger::endl;
  if (mAdcClock[2].addSequence(mGBTFrames.back().getAdcClock(2))) 
    LOG(WARNING) << "ADC clock error of SAMPA 2 in GBT Frame " << getNentries() << FairLogger::endl;
}

/// \file GBTFrameContainer.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/GBTFrameContainer.h"

using namespace AliceO2::TPC;

GBTFrameContainer::GBTFrameContainer()
  : mEnableAdcClockWarning(true)
  , mEnableSyncPatternWarning(true)
{
  for (int i = 0; i < 3; ++i){
    mAdcClock.emplace_back(i);
  }
  
  mSyncPattern.emplace_back(0,0);
  mSyncPattern.emplace_back(0,1);
  mSyncPattern.emplace_back(1,0);
  mSyncPattern.emplace_back(1,1);
  mSyncPattern.emplace_back(2,0);

  mPosition.resize(5);
}

GBTFrameContainer::GBTFrameContainer(int amount)
  : mEnableAdcClockWarning(true)
  , mEnableSyncPatternWarning(true)
{
  mGBTFrames.reserve(amount);
  for (int i = 0; i < 3; ++i){
    mAdcClock.emplace_back();
  }

  mSyncPattern.emplace_back(0,0);
  mSyncPattern.emplace_back(0,1);
  mSyncPattern.emplace_back(1,0);
  mSyncPattern.emplace_back(1,1);
  mSyncPattern.emplace_back(2,0);

  mPosition.resize(5);
}

GBTFrameContainer::~GBTFrameContainer()
{
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
  if (mAdcClock[0].addSequence(mGBTFrames.back().getAdcClock(0))) { 
    if (mEnableAdcClockWarning) { LOG(WARNING) << "ADC clock error of SAMPA 0 in GBT Frame " << getNentries() << FairLogger::endl;}}
  if (mAdcClock[1].addSequence(mGBTFrames.back().getAdcClock(1))) {
    if (mEnableAdcClockWarning) { LOG(WARNING) << "ADC clock error of SAMPA 1 in GBT Frame " << getNentries() << FairLogger::endl;}}
  if (mAdcClock[2].addSequence(mGBTFrames.back().getAdcClock(2))) {
    if (mEnableAdcClockWarning) { LOG(WARNING) << "ADC clock error of SAMPA 2 in GBT Frame " << getNentries() << FairLogger::endl;}}

  mPosition[0] = mSyncPattern[0].addSequence(
      mGBTFrames.back().getHalfWord(0,0,0),
      mGBTFrames.back().getHalfWord(0,1,0),
      mGBTFrames.back().getHalfWord(0,2,0),
      mGBTFrames.back().getHalfWord(0,3,0));

  mPosition[1] = mSyncPattern[1].addSequence(
      mGBTFrames.back().getHalfWord(0,0,1),
      mGBTFrames.back().getHalfWord(0,1,1),
      mGBTFrames.back().getHalfWord(0,2,1),
      mGBTFrames.back().getHalfWord(0,3,1));

  mPosition[2] = mSyncPattern[2].addSequence(
      mGBTFrames.back().getHalfWord(1,0,0),
      mGBTFrames.back().getHalfWord(1,1,0),
      mGBTFrames.back().getHalfWord(1,2,0),
      mGBTFrames.back().getHalfWord(1,3,0));

  mPosition[3] = mSyncPattern[3].addSequence(
      mGBTFrames.back().getHalfWord(1,0,1),
      mGBTFrames.back().getHalfWord(1,1,1),
      mGBTFrames.back().getHalfWord(1,2,1),
      mGBTFrames.back().getHalfWord(1,3,1));

  mPosition[4] = mSyncPattern[4].addSequence(
      mGBTFrames.back().getHalfWord(2,0),
      mGBTFrames.back().getHalfWord(2,1),
      mGBTFrames.back().getHalfWord(2,2),
      mGBTFrames.back().getHalfWord(2,3));

  if (mPosition[0] != mPosition[1]) {
    if (mEnableSyncPatternWarning) { 
      LOG(WARNING) << "The two half words from SAMPA 0 don't start at the same position, lower bits start at "
        << mPosition[0] << ", higher bits at " << mPosition[1] << FairLogger::endl;
    }
  }
  if (mPosition[2] != mPosition[3]) {
    if (mEnableSyncPatternWarning) {
      LOG(WARNING) << "The two half words from SAMPA 1 don't start at the same position, lower bits start at "
        << mPosition[2] << ", higher bits at " << mPosition[3] << FairLogger::endl;
    }
  }
  if (mPosition[0] != mPosition[2] || mPosition[0] != mPosition[4]) {
    if (mEnableSyncPatternWarning) {
      LOG(WARNING) << "The three SAMPAs don't have the same position, SAMPA0 = " << mPosition[0] 
        << ", SAMPA1 = " << mPosition[2] << ", SAMPA2 = " << mPosition[4] << FairLogger::endl;
    }
  }

}

void GBTFrameContainer::reset() 
{
  LOG(INFO) << "Resetting GBT-Frame container" << FairLogger::endl;
  mGBTFrames.clear();
  for (auto &aAdcClock : mAdcClock) {
    aAdcClock.reset();
  }
  for (auto &aSyncPattern : mSyncPattern) {
    aSyncPattern.reset();
  }
//  for (auto &aGBTFrame : mGBTFrames) {
//    if (aGBTFrame == nullptr) continue;
//    aGBTFrame->reset();
//  }
}

int GBTFrameContainer::getNentries() 
{
  int counter = 0;
  for (auto &aGBTFrame : mGBTFrames) {
    ++counter;
  }
  return counter;
}

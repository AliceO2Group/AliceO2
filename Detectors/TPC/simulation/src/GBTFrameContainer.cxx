/// \file GBTFrameContainer.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/GBTFrameContainer.h"

using namespace AliceO2::TPC;

GBTFrameContainer::GBTFrameContainer()
  : GBTFrameContainer(0,0)
{}

GBTFrameContainer::GBTFrameContainer(int cru, int link)
  : GBTFrameContainer(1024,cru,link)
{}

GBTFrameContainer::GBTFrameContainer(int amount, int cru, int link)
  : mEnableAdcClockWarning(true)
  , mEnableSyncPatternWarning(true)
  , mPositionForHalfSampa(5)
  , mAdcValues(5)
  , mCRU(cru)
  , mLink(link)
  , mTimebin(0)
{
  mGBTFrames.reserve(amount);

  for (int i = 0; i < 3; ++i){
    mAdcClock.emplace_back(i);
  }

  mSyncPattern.emplace_back(0,0);
  mSyncPattern.emplace_back(0,1);
  mSyncPattern.emplace_back(1,0);
  mSyncPattern.emplace_back(1,1);
  mSyncPattern.emplace_back(2,0);
}

GBTFrameContainer::~GBTFrameContainer()
{
}

void GBTFrameContainer::addGBTFrame(GBTFrame& frame) 
{
  mGBTFrames.emplace_back(frame);
  processFrame(mGBTFrames.end());
}

void GBTFrameContainer::addGBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0)
{
  mGBTFrames.emplace_back(word3, word2, word1, word0);
  processFrame(mGBTFrames.end());
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
  processFrame(mGBTFrames.end());
}


void GBTFrameContainer::fillOutputContainer(TClonesArray* output)
{

  TClonesArray &clref = *output;
  for (auto &aGBTFrame : mGBTFrames) {
    new(clref[clref.GetEntriesFast()]) GBTFrame(aGBTFrame);
  }
}

void GBTFrameContainer::processAllFrames()
{
  if (mAdcValues.size() > 0)
    LOG(WARNING) << "There are already some ADC values, maybe the frames were already processed." << FairLogger::endl;
  for (auto it = mGBTFrames.begin(); it != mGBTFrames.end(); ++it) {
    processFrame(it);
  }

}

void GBTFrameContainer::processFrame(std::vector<GBTFrame>::iterator iFrame)
{
  checkAdcClock(&(*iFrame));

  int iOldPosition = searchSyncPattern(&(*iFrame));

  for (auto it = mPositionForHalfSampa.begin(); it != mPositionForHalfSampa.end(); it++) {
    int iPosition = *it;
    if (iPosition != -1) {
      int iHalfSampa = std::distance(mPositionForHalfSampa.begin(),it);
      if (iOldPosition != -1) {
        switch(iPosition) {
          case 0:
            mAdcValues[iHalfSampa].push_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) +
                (int)iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2));
            mAdcValues[iHalfSampa].push_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) + 
                (int)iFrame->getHalfWord(iHalfSampa/2,2,iHalfSampa%2));
            break;
  
          case 1:
            mAdcValues[iHalfSampa].push_back(
                (int)((iFrame-1)->getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) +
                (int)(iFrame-1)->getHalfWord(iHalfSampa/2,1,iHalfSampa%2));
            mAdcValues[iHalfSampa].push_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) +
                (int)(iFrame-1)->getHalfWord(iHalfSampa/2,3,iHalfSampa%2));
            break;
  
          case 2:
            mAdcValues[iHalfSampa].push_back(
                (int)((iFrame-1)->getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) +
                (int)(iFrame-1)->getHalfWord(iHalfSampa/2,2,iHalfSampa%2));
            mAdcValues[iHalfSampa].push_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) + 
                (int)iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2));
            break;
  
          case 3:
            mAdcValues[iHalfSampa].push_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) +
                (int)(iFrame-1)->getHalfWord(iHalfSampa/2,3,iHalfSampa%2));
            mAdcValues[iHalfSampa].push_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) + 
                (int)iFrame->getHalfWord(iHalfSampa/2,1,iHalfSampa%2));
            break;
  
          default:
            LOG(ERROR) << "Position " << iPosition << " not known." << FairLogger::endl;
            break;
        }
//      std::cout << iHalfSampa << " " << std::hex
//         << "0x" << std::setfill('0') << std::setw(3) << *(mAdcValues[iHalfSampa].rbegin()+1) << " "
//         << "0x" << std::setfill('0') << std::setw(3) << *(mAdcValues[iHalfSampa].rbegin()) << std::dec << std::endl;
      }
    }
  }
}

void GBTFrameContainer::checkAdcClock(const GBTFrame* iFrame)
{
  if (mAdcClock[0].addSequence(iFrame->getAdcClock(0))) { 
    if (mEnableAdcClockWarning) { LOG(WARNING) << "ADC clock error of SAMPA 0 in GBT Frame " << iFrame << FairLogger::endl;}}
  if (mAdcClock[1].addSequence(iFrame->getAdcClock(1))) {
    if (mEnableAdcClockWarning) { LOG(WARNING) << "ADC clock error of SAMPA 1 in GBT Frame " << iFrame << FairLogger::endl;}}
  if (mAdcClock[2].addSequence(iFrame->getAdcClock(2))) {
    if (mEnableAdcClockWarning) { LOG(WARNING) << "ADC clock error of SAMPA 2 in GBT Frame " << iFrame << FairLogger::endl;}}
}

int GBTFrameContainer::searchSyncPattern(const GBTFrame* iFrame)
{
  int iOldPosition = mPositionForHalfSampa[0];

  mPositionForHalfSampa[0] = mSyncPattern[0].addSequence(
      iFrame->getHalfWord(0,0,0),
      iFrame->getHalfWord(0,1,0),
      iFrame->getHalfWord(0,2,0),
      iFrame->getHalfWord(0,3,0));

  mPositionForHalfSampa[1] = mSyncPattern[1].addSequence(
      iFrame->getHalfWord(0,0,1),
      iFrame->getHalfWord(0,1,1),
      iFrame->getHalfWord(0,2,1),
      iFrame->getHalfWord(0,3,1));

  mPositionForHalfSampa[2] = mSyncPattern[2].addSequence(
      iFrame->getHalfWord(1,0,0),
      iFrame->getHalfWord(1,1,0),
      iFrame->getHalfWord(1,2,0),
      iFrame->getHalfWord(1,3,0));

  mPositionForHalfSampa[3] = mSyncPattern[3].addSequence(
      iFrame->getHalfWord(1,0,1),
      iFrame->getHalfWord(1,1,1),
      iFrame->getHalfWord(1,2,1),
      iFrame->getHalfWord(1,3,1));

  mPositionForHalfSampa[4] = mSyncPattern[4].addSequence(
      iFrame->getHalfWord(2,0),
      iFrame->getHalfWord(2,1),
      iFrame->getHalfWord(2,2),
      iFrame->getHalfWord(2,3));

  if (mPositionForHalfSampa[0] != mPositionForHalfSampa[1]) {
    if (mEnableSyncPatternWarning) { 
      LOG(WARNING) << "The two half words from SAMPA 0 don't start at the same position, lower bits start at "
        << mPositionForHalfSampa[0] << ", higher bits at " << mPositionForHalfSampa[1] << FairLogger::endl;
    }
  }
  if (mPositionForHalfSampa[2] != mPositionForHalfSampa[3]) {
    if (mEnableSyncPatternWarning) {
      LOG(WARNING) << "The two half words from SAMPA 1 don't start at the same position, lower bits start at "
        << mPositionForHalfSampa[2] << ", higher bits at " << mPositionForHalfSampa[3] << FairLogger::endl;
    }
  }
  if (mPositionForHalfSampa[0] != mPositionForHalfSampa[2] || mPositionForHalfSampa[0] != mPositionForHalfSampa[4]) {
    if (mEnableSyncPatternWarning) {
      LOG(WARNING) << "The three SAMPAs don't have the same position, SAMPA0 = " << mPositionForHalfSampa[0] 
        << ", SAMPA1 = " << mPositionForHalfSampa[2] << ", SAMPA2 = " << mPositionForHalfSampa[4] << FairLogger::endl;
    }
  }

  return iOldPosition;
}

bool GBTFrameContainer::getDigits(std::vector<Digit> *digitContainer)
{

  bool digitsAdded = false;
  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
  {
    if (mAdcValues[iHalfSampa].size() < 16) continue;
    int eventID = -1;
    int trackID = -1;
    int cru = mCRU;
    int timeBin = mTimebin++;
    int row = 0;
    int pad = 0;
    float charge = -1;
    int iSampaChannel = (iHalfSampa == 4) ?     // 5th half SAMPA corresponds to  SAMPA2
        ((mCRU%2) ? 16 : 0) :                   // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
        ((iHalfSampa%2) ? 16 : 0);              // every even half SAMPA containes channel 0-15, the odd ones channel 16-31 
    std::cout << mCRU << " " << iHalfSampa << " " << iHalfSampa%2 << std::endl;
    for (int iChannel = 0; iChannel < 16; ++iChannel)
    {
      std::cout << iSampaChannel << std::endl;
      charge = mAdcValues[iHalfSampa].at(0); mAdcValues[iHalfSampa].pop_front();
      digitContainer->emplace_back(eventID, trackID, cru, charge, row, pad, timeBin);
      digitsAdded = true;
      ++pad;
      ++iSampaChannel;
    }
  }

  return digitsAdded;
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
  for (auto &aAdcValues : mAdcValues) {
    aAdcValues.clear();
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

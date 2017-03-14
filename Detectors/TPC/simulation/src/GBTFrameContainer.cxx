/// \file GBTFrameContainer.cxx
/// \author Sebastian Klewin

#include "TPCSimulation/GBTFrameContainer.h"

using namespace AliceO2::TPC;

GBTFrameContainer::GBTFrameContainer()
  : GBTFrameContainer(0,0)
{}

GBTFrameContainer::GBTFrameContainer(int cru, int link)
  : GBTFrameContainer(0,cru,link)
{}

GBTFrameContainer::GBTFrameContainer(int size, int cru, int link)
  : mtx()
  , mEnableAdcClockWarning(true)
  , mEnableSyncPatternWarning(true)
  , mEnableStoreGBTFrames(true)
  , mPositionForHalfSampa(5,-1)
  , mGBTFrames(size)
  , mAdcValues(5)
  , mCRU(cru)
  , mLink(link)
  , mTimebin(0)
{
//  mGBTFrames.reserve(amount);

  for (int i = 0; i < 3; ++i){
    mAdcClock.emplace_back(i);
  }

  mSyncPattern.emplace_back(0,0);
  mSyncPattern.emplace_back(0,1);
  mSyncPattern.emplace_back(1,0);
  mSyncPattern.emplace_back(1,1);
  mSyncPattern.emplace_back(2,0);

  for (auto &aAdcValues : mAdcValues) {
    aAdcValues = new std::deque<int>;
  }
}

GBTFrameContainer::~GBTFrameContainer()
{
}

void GBTFrameContainer::addGBTFrame(GBTFrame& frame) 
{
  mGBTFrames.emplace_back(frame);
  processFrame(mGBTFrames.end()-1);

  if (!mEnableStoreGBTFrames) {
    if (mGBTFrames.size() > 2) mGBTFrames.pop_front();
  }
}

void GBTFrameContainer::addGBTFrame(unsigned word3, unsigned word2, unsigned word1, unsigned word0)
{
  mGBTFrames.emplace_back(word3, word2, word1, word0);
  processFrame(mGBTFrames.end()-1);
  if (!mEnableStoreGBTFrames) {
    if (mGBTFrames.size() > 2) mGBTFrames.pop_front();
  }
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
  processFrame(mGBTFrames.end()-1);
  if (!mEnableStoreGBTFrames) {
    if (mGBTFrames.size() > 2) mGBTFrames.pop_front();
  }
}

void GBTFrameContainer::addGBTFramesFromFile(std::string fileName)
{
  std::cout << "Reading from file " << fileName << std::endl;
  std::ifstream file(fileName);

  if (!file.is_open()) {
    LOG(ERROR) << "Can't read file " << fileName << FairLogger::endl;
    return;
  }

  /* Expected format in file is
   *
   * decimal-counter : hex-GBTframe
   * e.g.
   *
   * ...
   *  0016 : 0x00000da2d80702e17aaeb0f0052faaa5
   *  0017 : 0x0000078a7805825272261050870502f0
   *  0018 : 0x000007207a0722e1da0bc0520f0582da
   * ...
   * 
   */

  int counter;
  char colon;
  std::string frameString;
  int word0, word1, word2, word3;

  while (file >> counter >> colon >> frameString) {
    sscanf(frameString.substr( 2,8).c_str(), "%x", &word3);
    sscanf(frameString.substr(10,8).c_str(), "%x", &word2);
    sscanf(frameString.substr(18,8).c_str(), "%x", &word1);
    sscanf(frameString.substr(26,8).c_str(), "%x", &word0);

    addGBTFrame(word3,word2,word1,word0);
  }
}


//void GBTFrameContainer::fillOutputContainer(TClonesArray* output)
//{
//
//  TClonesArray &clref = *output;
//  for (auto &aGBTFrame : mGBTFrames) {
//    new(clref[clref.GetEntriesFast()]) GBTFrame(aGBTFrame);
//  }
//}

void GBTFrameContainer::reProcessAllFrames()
{
  resetAdcClock();
  resetSyncPattern();
  resetAdcValues();

  processAllFrames();

}

void GBTFrameContainer::processAllFrames()
{

  mtx.lock();
  for (std::vector<std::deque<int>*>::iterator it = mAdcValues.begin(); it != mAdcValues.end(); ++it) {
    if ((*it)->size() > 0) {
      LOG(WARNING) << "There are already some ADC values for half SAMPA " 
        << std::distance(mAdcValues.begin(),it) 
        << " , maybe the frames were already processed." << FairLogger::endl;
    }
  }
  mtx.unlock();
  for (auto it = mGBTFrames.begin(); it != mGBTFrames.end(); ++it) {
    processFrame(it);
  }

}

void GBTFrameContainer::processFrame(std::deque<GBTFrame>::iterator iFrame)
{
  checkAdcClock(iFrame);

  int iOldPosition = searchSyncPattern(iFrame);

  mtx.lock();
  for (auto it = mPositionForHalfSampa.begin(); it != mPositionForHalfSampa.end(); it++) {
    int iPosition = *it;
    if (iPosition != -1) {
      int iHalfSampa = std::distance(mPositionForHalfSampa.begin(),it);
      if (iOldPosition != -1) {
        switch(iPosition) {
          case 0:
            mAdcValues[iHalfSampa]->emplace_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) +
                (int)iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2));
            mAdcValues[iHalfSampa]->emplace_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) + 
                (int)iFrame->getHalfWord(iHalfSampa/2,2,iHalfSampa%2));
            break;
  
          case 1:
            mAdcValues[iHalfSampa]->emplace_back(
                (int)((iFrame-1)->getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) +
                (int)(iFrame-1)->getHalfWord(iHalfSampa/2,1,iHalfSampa%2));
            mAdcValues[iHalfSampa]->emplace_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) +
                (int)(iFrame-1)->getHalfWord(iHalfSampa/2,3,iHalfSampa%2));
            break;
  
          case 2:
            mAdcValues[iHalfSampa]->emplace_back(
                (int)((iFrame-1)->getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) +
                (int)(iFrame-1)->getHalfWord(iHalfSampa/2,2,iHalfSampa%2));
            mAdcValues[iHalfSampa]->emplace_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) + 
                (int)iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2));
            break;
  
          case 3:
            mAdcValues[iHalfSampa]->emplace_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) +
                (int)(iFrame-1)->getHalfWord(iHalfSampa/2,3,iHalfSampa%2));
            mAdcValues[iHalfSampa]->emplace_back(
                (int)(iFrame->getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) + 
                (int)iFrame->getHalfWord(iHalfSampa/2,1,iHalfSampa%2));
            break;
  
          default:
            LOG(ERROR) << "Position " << iPosition << " not known." << FairLogger::endl;
            return;
        }
//      std::cout << iHalfSampa << " " << std::hex
//         << "0x" << std::setfill('0') << std::setw(3) << *(mAdcValues[iHalfSampa]->rbegin()+1) << " "
//         << "0x" << std::setfill('0') << std::setw(3) << *(mAdcValues[iHalfSampa]->rbegin()) << std::dec << std::endl;
      }
    }
  }
  mtx.unlock();
}

void GBTFrameContainer::checkAdcClock(std::deque<GBTFrame>::iterator iFrame)
{
  if (mAdcClock[0].addSequence(iFrame->getAdcClock(0))) 
  { 
    if (mEnableAdcClockWarning) { LOG(WARNING) << "ADC clock error of SAMPA 0 in GBT Frame " << std::distance(mGBTFrames.begin(),iFrame) << FairLogger::endl;}
  }
  if (mAdcClock[1].addSequence(iFrame->getAdcClock(1))) 
  {
    if (mEnableAdcClockWarning) { LOG(WARNING) << "ADC clock error of SAMPA 1 in GBT Frame " << std::distance(mGBTFrames.begin(),iFrame) << FairLogger::endl;}
  }
  if (mAdcClock[2].addSequence(iFrame->getAdcClock(2))) 
  {
    if (mEnableAdcClockWarning) { LOG(WARNING) << "ADC clock error of SAMPA 2 in GBT Frame " << std::distance(mGBTFrames.begin(),iFrame) << FairLogger::endl;}
  }
}

int GBTFrameContainer::searchSyncPattern(std::deque<GBTFrame>::iterator iFrame)
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

//  std::cout << mPositionForHalfSampa[0] << " " << mPositionForHalfSampa[1] << " " << mPositionForHalfSampa[2] << " " << mPositionForHalfSampa[3] << " " << mPositionForHalfSampa[4] << std::endl;

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

bool GBTFrameContainer::getDigits(std::vector<Digit> *digitContainer, bool removeChannel)
{
  std::vector<std::vector<int>> iData(5);
  mtx.lock();
  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
  {
    if (mAdcValues[iHalfSampa]->size() < 16) continue;
    int position = 0;
    for (int iChannel = 0; iChannel < 16; ++iChannel)
    {
      iData[iHalfSampa].push_back(mAdcValues[iHalfSampa]->at(position));
      if (removeChannel) mAdcValues[iHalfSampa]->pop_front();
      else ++position;
    }
  }
  mtx.unlock();

  const Mapper& mapper = Mapper::instance();
  bool digitsAdded = false;
  int iTimeBin = mTimebin;
  int iSampaChannel;
  int iSampa;
  float iCharge;
  int iRow;
  int iPad;
  std::vector<long> iMcLabel;
  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
  {
    iSampaChannel = (iHalfSampa == 4) ?     // 5th half SAMPA corresponds to  SAMPA2
        ((mCRU%2) ? 16 : 0) :                   // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
        ((iHalfSampa%2) ? 16 : 0);              // every even half SAMPA containes channel 0-15, the odd ones channel 16-31 
    iSampa =  (iHalfSampa == 4) ?
        2 :
        (mCRU%2) ? iHalfSampa/2+3 : iHalfSampa/2;
    int position = 0;
    for (std::vector<int>::iterator it = iData[iHalfSampa].begin(); it != iData[iHalfSampa].end(); ++it)
    {
      const PadPos& padPos = mapper.padPos(
          mCRU/2 /*partition*/, 
          mLink /*FEC in partition*/,
          iSampa,
          iSampaChannel);
      iRow = padPos.getRow();
      iPad = padPos.getPad();
      iCharge = *it; 
//      std::cout << mCRU/2 << " " << mLink << " " << iSampa << " " << iSampaChannel << " " << iRow << " " << iPad << " " << iTimeBin << " " << iCharge << std::endl;

      digitContainer->emplace_back(iMcLabel, mCRU, iCharge, iRow, iPad, iTimeBin);
      digitsAdded = true;
      ++iSampaChannel;
    }
  }

  if (digitsAdded) ++mTimebin;
  return digitsAdded;
}

bool GBTFrameContainer::getData(std::vector<SAMPAData>* container, bool removeChannel)
{
  if (container->size() != 5) {
    LOG(INFO) << "Container had the wrong size, set it to 5" << FairLogger::endl;
    container->resize(5);
  }

  std::vector<std::vector<int>> iData(5);
  mtx.lock();
  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
  {
    if (mAdcValues[iHalfSampa]->size() < 16) continue;
    int position = 0;
    for (int iChannel = 0; iChannel < 16; ++iChannel)
    {
      iData[iHalfSampa].emplace_back(mAdcValues[iHalfSampa]->at(position));
      if (removeChannel) {
        mAdcValues[iHalfSampa]->pop_front();
      }
      else ++position;
    }
  }
  mtx.unlock();

  bool dataAdded = false;

  int iSampaChannel;
  int iSampa;

  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
  {
    iSampaChannel = (iHalfSampa == 4) ?         // 5th half SAMPA corresponds to  SAMPA2
        ((mCRU%2) ? 16 : 0) :                   // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
        ((iHalfSampa%2) ? 16 : 0);              // every even half SAMPA containes channel 0-15, the odd ones channel 16-31 
    iSampa =  (iHalfSampa == 4) ?
        2 :
        (mCRU%2) ? iHalfSampa/2+3 : iHalfSampa/2;

//    std::cout << iHalfSampa << " " << iData[iHalfSampa].size() << " " << mAdcValues[iHalfSampa]->size() << std::endl;
    for (std::vector<int>::iterator it = iData[iHalfSampa].begin(); it != iData[iHalfSampa].end(); ++it)
    {
      container->at(iSampa).setChannel(iSampaChannel,*it);
      ++iSampaChannel;
      dataAdded = true;
    }
  }
  return dataAdded;
}

void GBTFrameContainer::reset() 
{
  LOG(INFO) << "Resetting GBT-Frame container" << FairLogger::endl;
  resetAdcClock();
  resetSyncPattern();
  resetAdcValues();

  mGBTFrames.clear();
}

void GBTFrameContainer::resetAdcClock()
{
  for (auto &aAdcClock : mAdcClock) {
    aAdcClock.reset();
  }
}

void GBTFrameContainer::resetSyncPattern()
{
  for (auto &aSyncPattern : mSyncPattern) {
    aSyncPattern.reset();
  }
  for (auto &aPositionForHalfSampa : mPositionForHalfSampa) {
    aPositionForHalfSampa = -1;
  }
//  mPositionForHalfSampa.clear();
}

void GBTFrameContainer::resetAdcValues()
{
  mtx.lock();
  for (std::vector<std::deque<int>*>::iterator it = mAdcValues.begin(); it != mAdcValues.end(); ++it) {
    (*it)->clear();
  }
  mtx.unlock();
}

int GBTFrameContainer::getNentries() 
{
  int counter = 0;
  mtx.lock();
  for (auto &aAdcValues : mAdcValues) {
    counter += aAdcValues->size();
  }
  mtx.unlock();
  return counter;
}

void GBTFrameContainer::overwriteAdcClock(int sampa, int phase)
{
  unsigned clock = (0xFFFF0000 >> phase);
  unsigned shift = 28;
  for (std::deque<GBTFrame>::iterator it = mGBTFrames.begin(); it != mGBTFrames.end(); ++it) {
    it->setAdcClock(sampa,clock >> shift);
    shift = (shift - 4) % 32;
  }
}

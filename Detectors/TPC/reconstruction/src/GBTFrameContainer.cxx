/// \file GBTFrameContainer.cxx
/// \author Sebastian Klewin

#include "TPCReconstruction/GBTFrameContainer.h"

using namespace o2::TPC;

GBTFrameContainer::GBTFrameContainer()
  : GBTFrameContainer(0,0)
{}

GBTFrameContainer::GBTFrameContainer(int cru, int link)
  : GBTFrameContainer(0,cru,link)
{}

GBTFrameContainer::GBTFrameContainer(int size, int cru, int link)
  : mAdcMutex()
  , mEnableAdcClockWarning(true)
  , mEnableSyncPatternWarning(true)
  , mEnableStoreGBTFrames(true)
  , mEnableCompileAdcValues(true)
  , mAdcClock({
      AdcClockMonitor(0),
      AdcClockMonitor(1),
      AdcClockMonitor(2)})
  , mSyncPattern({
      SyncPatternMonitor(0,0),
      SyncPatternMonitor(0,1),
      SyncPatternMonitor(1,0),
      SyncPatternMonitor(1,1),
      SyncPatternMonitor(2,0)})
  , mPositionForHalfSampa({-1,-1,-1,-1,-1})
  , mGBTFrames()
  , mGBTFramesAnalyzed(0)
  , mCRU(cru)
  , mLink(link)
  , mTimebin(0)
{
  mGBTFrames.reserve(size);

  for (auto &aAdcValues : mAdcValues) {
    aAdcValues = new std::queue<short>;
  }
}

GBTFrameContainer::~GBTFrameContainer()
{
  for (auto &aAdcValues : mAdcValues) {
    delete aAdcValues;
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
  short colon;
  std::string frameString;
  unsigned word0, word1, word2, word3;

  while (file >> counter >> colon >> frameString) {
    sscanf(frameString.substr( 2,8).c_str(), "%x", &word3);
    sscanf(frameString.substr(10,8).c_str(), "%x", &word2);
    sscanf(frameString.substr(18,8).c_str(), "%x", &word1);
    sscanf(frameString.substr(26,8).c_str(), "%x", &word0);

    addGBTFrame(word3,word2,word1,word0);
  }
}

void GBTFrameContainer::addGBTFramesFromBinaryFile(std::string fileName, int frames)
{
  std::cout << "Reading from file " << fileName << std::endl;
  std::ifstream file(fileName);

  if (!file.is_open()) {
    LOG(ERROR) << "Can't read file " << fileName << FairLogger::endl;
    return;
  }

  uint32_t rawData;
  uint32_t rawMarker;
  uint32_t words[3];
  while (!file.eof() && ((frames == -1) || (mGBTFramesAnalyzed < frames))) {
    file.read((char*)&rawData,sizeof(rawData));
    rawMarker = rawData & 0xFFFF0000;
    if ((rawMarker == 0xDEF10000) || (rawMarker == 0xDEF40000)) {
      file.read((char*)&words,3*sizeof(words[0]));
      addGBTFrame(rawData,words[0],words[1],words[2]); 
    }
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
  mGBTFramesAnalyzed = 0;

  processAllFrames();

}

void GBTFrameContainer::processAllFrames()
{

  mAdcMutex.lock();
  for (std::array<std::queue<short>*,5>::iterator it = mAdcValues.begin(); it != mAdcValues.end(); ++it) {
    if ((*it)->size() > 0) {
      LOG(WARNING) << "There are already some ADC values for half SAMPA " 
        << std::distance(mAdcValues.begin(),it) 
        << " , maybe the frames were already processed." << FairLogger::endl;
    }
  }
  mAdcMutex.unlock();

  for (auto it = mGBTFrames.begin(); it != mGBTFrames.end(); ++it) {
    processFrame(it);
  }
}

void GBTFrameContainer::processFrame(std::vector<GBTFrame>::iterator iFrame)
{
  ++mGBTFramesAnalyzed;

  if (mEnableAdcClockWarning) checkAdcClock(iFrame);

  int iLastPosition = searchSyncPattern(iFrame);
  if (iLastPosition == -1) return;

  if (mEnableCompileAdcValues) compileAdcValues(iFrame);
}

void GBTFrameContainer::compileAdcValues(std::vector<GBTFrame>::iterator iFrame)
{
  short value1;
  short value2;
  mAdcMutex.lock();
  for (short iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa) {
    if (mPositionForHalfSampa[iHalfSampa] == -1) continue;
    switch(mPositionForHalfSampa[iHalfSampa]) {
      case 0:
        value1 = 
            (iFrame->getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) |
             iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2);
        value2 = 
            (iFrame->getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) | 
             iFrame->getHalfWord(iHalfSampa/2,2,iHalfSampa%2);
        break;
  
      case 1:
        value1 = 
            ((iFrame-1)->getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) |
             (iFrame-1)->getHalfWord(iHalfSampa/2,1,iHalfSampa%2);
        value2 = 
            (iFrame   ->getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) |
            (iFrame-1)->getHalfWord(iHalfSampa/2,3,iHalfSampa%2);
        break;
  
      case 2:
        value1 = 
            ((iFrame-1)->getHalfWord(iHalfSampa/2,3,iHalfSampa%2) << 5) |
             (iFrame-1)->getHalfWord(iHalfSampa/2,2,iHalfSampa%2);
        value2 = 
            (iFrame->getHalfWord(iHalfSampa/2,1,iHalfSampa%2) << 5) | 
             iFrame->getHalfWord(iHalfSampa/2,0,iHalfSampa%2);
        break;
  
      case 3:
        value1 = 
            (iFrame   ->getHalfWord(iHalfSampa/2,0,iHalfSampa%2) << 5) |
            (iFrame-1)->getHalfWord(iHalfSampa/2,3,iHalfSampa%2);
        value2 = 
            (iFrame->getHalfWord(iHalfSampa/2,2,iHalfSampa%2) << 5) | 
             iFrame->getHalfWord(iHalfSampa/2,1,iHalfSampa%2);
        break;
  
      default:
        LOG(ERROR) << "Position " << mPositionForHalfSampa[iHalfSampa] << " not known." << FairLogger::endl;
        return;
    }

    mAdcValues[iHalfSampa]->emplace(value1 ^ (1 << 9));
    mAdcValues[iHalfSampa]->emplace(value2 ^ (1 << 9));
//  std::cout << iHalfSampa << " " << std::hex
//     << "0x" << std::setfill('0') << std::setw(3) << *(mAdcValues[iHalfSampa]->rbegin()+1) << " "
//     << "0x" << std::setfill('0') << std::setw(3) << *(mAdcValues[iHalfSampa]->rbegin()) << std::dec << std::endl;
  }
  mAdcMutex.unlock();
}

void GBTFrameContainer::checkAdcClock(std::vector<GBTFrame>::iterator iFrame)
{
  if (mAdcClock[0].addSequence(iFrame->getAdcClock(0))) 
    LOG(WARNING) << "ADC clock error of SAMPA 0 in GBT Frame " << std::distance(mGBTFrames.begin(),iFrame) << FairLogger::endl;
  if (mAdcClock[1].addSequence(iFrame->getAdcClock(1))) 
    LOG(WARNING) << "ADC clock error of SAMPA 1 in GBT Frame " << std::distance(mGBTFrames.begin(),iFrame) << FairLogger::endl;
  if (mAdcClock[2].addSequence(iFrame->getAdcClock(2))) 
    LOG(WARNING) << "ADC clock error of SAMPA 2 in GBT Frame " << std::distance(mGBTFrames.begin(),iFrame) << FairLogger::endl;
}

int GBTFrameContainer::searchSyncPattern(std::vector<GBTFrame>::iterator iFrame)
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

  if (mEnableSyncPatternWarning) { 
    if (mPositionForHalfSampa[0] != mPositionForHalfSampa[1]) {
      LOG(WARNING) << "The two half words from SAMPA 0 don't start at the same position, lower bits start at "
        << mPositionForHalfSampa[0] << ", higher bits at " << mPositionForHalfSampa[1] << FairLogger::endl;
    }
    if (mPositionForHalfSampa[2] != mPositionForHalfSampa[3]) {
      LOG(WARNING) << "The two half words from SAMPA 1 don't start at the same position, lower bits start at "
        << mPositionForHalfSampa[2] << ", higher bits at " << mPositionForHalfSampa[3] << FairLogger::endl;
    }
    if (mPositionForHalfSampa[0] != mPositionForHalfSampa[2] || mPositionForHalfSampa[0] != mPositionForHalfSampa[4]) {
      LOG(WARNING) << "The three SAMPAs don't have the same position, SAMPA0 = " << mPositionForHalfSampa[0] 
        << ", SAMPA1 = " << mPositionForHalfSampa[2] << ", SAMPA2 = " << mPositionForHalfSampa[4] << FairLogger::endl;
    }
  }

  return iOldPosition;
}

bool GBTFrameContainer::getData(std::vector<DigitData>& container)
{
//  std::vector<std::vector<int>> iData(5);
//  mAdcMutex.lock();
//  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
//  {
//    if (mAdcValues[iHalfSampa]->size() < 16) continue;
//    for (int iChannel = 0; iChannel < 16; ++iChannel)
//    {
//      iData[iHalfSampa].push_back(mAdcValues[iHalfSampa]->front());
//      mAdcValues[iHalfSampa]->pop();
//    }
//  }
//  mAdcMutex.unlock();
  bool dataAvailable = false;

  mAdcMutex.lock();
  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
  {
    if (mAdcValues[iHalfSampa]->size() < 16) continue;
    dataAvailable = true;
    for (int iChannel = 0; iChannel < 16; ++iChannel)
    {
      mTmpData[iHalfSampa][iChannel] = mAdcValues[iHalfSampa]->front();
      mAdcValues[iHalfSampa]->pop();
    }
  }
  mAdcMutex.unlock();

  if (!dataAvailable) return dataAvailable;

  const Mapper& mapper = Mapper::instance();
  int iTimeBin = mTimebin;
  int iSampaChannel;
  int iSampa;
  float iCharge;
  int iRow;
  int iPad;
  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
  {
    iSampaChannel = (iHalfSampa == 4) ?     // 5th half SAMPA corresponds to  SAMPA2
        ((mCRU%2) ? 16 : 0) :                   // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
        ((iHalfSampa%2) ? 16 : 0);              // every even half SAMPA containes channel 0-15, the odd ones channel 16-31 
    iSampa =  (iHalfSampa == 4) ?
        2 :
        (mCRU%2) ? iHalfSampa/2+3 : iHalfSampa/2;
    for (std::array<short,16>::iterator it = mTmpData[iHalfSampa].begin(); it != mTmpData[iHalfSampa].end(); ++it)
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

      container.emplace_back(mCRU, iCharge, iRow, iPad, iTimeBin);
      ++iSampaChannel;
    }
  }

  if (dataAvailable) ++mTimebin;
  return dataAvailable;
}

bool GBTFrameContainer::getData(std::vector<HalfSAMPAData>& container)
{
  bool dataAvailable = false;

  mAdcMutex.lock();
  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
  {
    if (mAdcValues[iHalfSampa]->size() < 16) continue;
    dataAvailable = true;
    for (int iChannel = 0; iChannel < 16; ++iChannel)
    {
      mTmpData[iHalfSampa][iChannel] = mAdcValues[iHalfSampa]->front();
      mAdcValues[iHalfSampa]->pop();
    }
  }
  mAdcMutex.unlock();

  if (!dataAvailable) return dataAvailable;

//  if (container.size() != 5) {
////    LOG(INFO) << "Container had the wrong size, set it to 5" << FairLogger::endl;
//    container.resize(5);
//  }
//  container.at(0).reset();
//  container.at(1).reset();
//  container.at(2).reset();
//  container.at(3).reset();
//  container.at(4).reset();

  int iSampaChannel;
  int iSampa;

  for (int iHalfSampa = 0; iHalfSampa < 5; ++iHalfSampa)
  {
//    iSampaChannel = 0;
//    iSampaChannel = (iHalfSampa == 4) ?         // 5th half SAMPA corresponds to  SAMPA2
//        ((mCRU%2) ? 16 : 0) :                   // every even CRU receives channel 0-15 from SAMPA 2, the odd ones channel 16-31
//        ((iHalfSampa%2) ? 16 : 0);              // every even half SAMPA containes channel 0-15, the odd ones channel 16-31 
    iSampa =  (iHalfSampa == 4) ?
        2 :
        (mCRU%2) ? iHalfSampa/2+3 : iHalfSampa/2;

    container.emplace_back(iSampa,!((bool)mCRU%2),mTmpData[iHalfSampa]);
//    container.at(iHalfSampa).setID(iSampa);
//    for (std::array<short,16>::iterator it = mTmpData[iHalfSampa].begin(); it != mTmpData[iHalfSampa].end(); ++it)
//    {
//      container.at(iHalfSampa).setChannel(iSampaChannel,*it);
//      ++iSampaChannel;
//    }
  }
  return dataAvailable;
}

void GBTFrameContainer::reset() 
{
  LOG(INFO) << "Resetting GBT-Frame container" << FairLogger::endl;
  resetAdcClock();
  resetSyncPattern();
  resetAdcValues();

  mGBTFrames.clear();
  mGBTFramesAnalyzed = 0;
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
  mAdcMutex.lock();
  for (std::array<std::queue<short>*,5>::iterator it = mAdcValues.begin(); it != mAdcValues.end(); ++it) {
    delete (*it);
    (*it) = new std::queue<short>;
  }
  mAdcMutex.unlock();
}

int GBTFrameContainer::getNentries() 
{
  int counter = 0;
  mAdcMutex.lock();
  for (auto &aAdcValues : mAdcValues) {
    counter += aAdcValues->size();
  }
  mAdcMutex.unlock();
  return counter;
}

void GBTFrameContainer::overwriteAdcClock(int sampa, int phase)
{
  unsigned clock = (0xFFFF0000 >> phase);
  unsigned shift = 28;

  for (std::vector<GBTFrame>::iterator it = mGBTFrames.begin(); it != mGBTFrames.end(); ++it) {
    it->setAdcClock(sampa,clock >> shift);
    shift = (shift - 4) % 32;
  }
}

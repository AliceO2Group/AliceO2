// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file AliTPCUpgradeHwClusterer.cxx
/// \brief Hwclusterer for the TPC

#include "TPCReconstruction/HwClusterer.h"
#include "TPCReconstruction/HwClusterFinder.h"
#include "TPCBase/Digit.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/CRU.h"
#include "TPCBase/PadSecPos.h"
#include "TPCBase/CalArray.h"

#include "FairLogger.h"
#include "TMath.h"

#include <thread>
#include <mutex>

std::mutex g_display_mutex;

using namespace o2::TPC;

//________________________________________________________________________
HwClusterer::HwClusterer(std::vector<o2::TPC::Cluster> *clusterOutput,
    std::unique_ptr<MCLabelContainer> &labelOutput,
    Processing processingType, int cruMin, int cruMax, float minQDiff,
    bool assignChargeUnique, bool enableNoiseSim, bool enablePedestalSubtraction, int padsPerCF, int timebinsPerCF)
  : Clusterer()
  , mClusterArray(clusterOutput)
  , mClusterMcLabelArray(labelOutput)
  , mProcessingType(processingType)
  , mCRUMin(cruMin)
  , mCRUMax(cruMax)
  , mMinQDiff(minQDiff)
  , mAssignChargeUnique(assignChargeUnique)
  , mEnableNoiseSim(enableNoiseSim)
  , mEnablePedestalSubtraction(enablePedestalSubtraction)
  , mIsContinuousReadout(true)
  , mPadsPerCF(padsPerCF)
  , mTimebinsPerCF(timebinsPerCF)
  , mLastTimebin(-1)
  , mNoiseObject(nullptr)
  , mPedestalObject(nullptr)
  , mLastMcDigitTruth()//new MCLabelContainer)
{
  /*
   * initalize all cluster finder
   */
  int iCfPerRow = (int)ceil((double)(mPadsMax+2+2)/(mPadsPerCF-2-2));
  mClusterFinder.resize(mCRUMax+1);
  const Mapper& mapper = Mapper::instance();
  for (int iCRU = mCRUMin; iCRU <= mCRUMax; iCRU++){
    mClusterFinder[iCRU].resize(mapper.getNumberOfRowsPartition(iCRU));
    for (int iRow = 0; iRow < mapper.getNumberOfRowsPartition(iCRU); iRow++){
      mClusterFinder[iCRU][iRow].resize(iCfPerRow);
      for (int iCF = 0; iCF < iCfPerRow; iCF++){
        int padOffset = iCF*(mPadsPerCF-2-2)-2;
        mClusterFinder[iCRU][iRow][iCF] = std::make_unique<HwClusterFinder>(iCRU,iRow,iCF,padOffset,mPadsPerCF,mTimebinsPerCF,mMinQDiff,mMinQMax,mRequirePositiveCharge);
        mClusterFinder[iCRU][iRow][iCF]->setAssignChargeUnique(mAssignChargeUnique);


        /*
         * Connect always two CFs to be able to comunicate found clusters. So
         * the "right" one can tell the one "on the left" which pads were
         * already used for a cluster.
         */
        if (iCF != 0) {
          mClusterFinder[iCRU][iRow][iCF]->setNextCF(mClusterFinder[iCRU][iRow][iCF-1].get());
        }
      }
    }
  }


  /*
   * vector of HwCluster vectors, one vector for each CRU (possible thread)
   * to store the clusters found there
   */
  mClusterStorage.resize(mCRUMax+1);
  mClusterDigitIndexStorage.resize(mCRUMax+1);


  /*
   * vector of digit vectors, one vector for each CRU (possible thread) to
   * store there only those digits which are relevant for this particular
   * CRU (thread)
   */
  mDigitContainer.resize(mCRUMax+1);
  for (int iCRU = mCRUMin; iCRU <= mCRUMax; iCRU++)
    mDigitContainer[iCRU].resize(mapper.getNumberOfRowsPartition(iCRU));

}

//________________________________________________________________________
HwClusterer::~HwClusterer()
{
  LOG(DEBUG) << "Enter Destructor of HwClusterer" << FairLogger::endl;

//  delete mLastMcDigitTruth;
}

//________________________________________________________________________
void HwClusterer::processDigits(
    const std::vector<std::vector<std::tuple<Digit const*, int, int>>>& digits,
    const std::vector<std::vector<std::unique_ptr<HwClusterFinder>>>& clusterFinder,
          std::vector<Cluster>& cluster,
          std::vector<std::vector<std::pair<int,int>>>& label,
    const CfConfig config)
{
  int timeDiff = (config.iMaxTimeBin+1) - config.iMinTimeBin;
  if (timeDiff < 0) return;
//  std::thread::id this_id = std::this_thread::get_id();
//  g_display_mutex.lock();
//  std::cout << "thread " << this_id << " started.\n";
//  g_display_mutex.unlock();

  HwClusterFinder::MiniDigit iAllBins[timeDiff][config.iMaxPads];

  for (int iRow = 0; iRow < config.iMaxRows; iRow++){

    /*
     * prepare local storage
     */
    short t,p;
    float noise;
    if (config.iEnableNoiseSim && config.iNoiseObject != nullptr) {
      for (t=timeDiff; t--;) {
        for (p=config.iMaxPads; p--;) {
          iAllBins[t][p].charge = config.iNoiseObject->getValue(CRU(config.iCRU),iRow,p);
          iAllBins[t][p].index = -1;
          iAllBins[t][p].event = -1;
        }
      }
    } else {
      std::fill(&iAllBins[0][0], &iAllBins[0][0]+timeDiff*config.iMaxPads, HwClusterFinder::MiniDigit());
    }

    /*
     * fill in digits
     */
    for (auto& digit : digits[iRow]){
      const Int_t iTime         = std::get<0>(digit)->getTimeStamp();
      const Int_t iPad          = std::get<0>(digit)->getPad() + 2;  // offset to have 2 empty pads on the "left side"
      const Float_t charge      = std::get<0>(digit)->getChargeFloat();

//      std::cout << iCRU << " " << iRow << " " << iPad << " " << iTime << " (" << iTime-minTime << "," << timeDiff << ") " << charge << std::endl;
      iAllBins[iTime-config.iMinTimeBin][iPad].charge += charge;
      iAllBins[iTime-config.iMinTimeBin][iPad].index = std::get<1>(digit);
      iAllBins[iTime-config.iMinTimeBin][iPad].event = std::get<2>(digit);
      if (config.iEnablePedestalSubtraction && config.iPedestalObject != nullptr) {
        const float pedestal = config.iPedestalObject->getValue(CRU(config.iCRU),iRow,iPad-2);
        //printf("digit: %.2f, pedestal: %.2f\n", iAllBins[iTime-config.iMinTimeBin][iPad], pedestal);
        iAllBins[iTime-config.iMinTimeBin][iPad].charge -= pedestal;
      }
    }

    /*
     * copy data to cluster finders
     */
    const Short_t iPadsPerCF = clusterFinder[iRow][0]->getNpads();
    const Short_t iTimebinsPerCF = clusterFinder[iRow][0]->getNtimebins();
    std::vector<std::vector<std::unique_ptr<HwClusterFinder>>::const_reverse_iterator> cfWithCluster;
    unsigned time,pad;
    for (time = 0; time < timeDiff; ++time){    // ordering important!!
      for (pad = 0; pad < config.iMaxPads; pad = pad + (iPadsPerCF -2 -2 )) {
        const Short_t cf = pad / (iPadsPerCF-2-2);
        clusterFinder[iRow][cf]->AddTimebin(&iAllBins[time][pad],time+config.iMinTimeBin,(config.iMaxPads-pad)>=iPadsPerCF?iPadsPerCF:(config.iMaxPads-pad));
      }

      /*
       * search for clusters and store reference to CF if one was found
       */
      if (clusterFinder[iRow][0]->getTimebinsAfterLastProcessing() == iTimebinsPerCF-2 -2)  {
        /*
         * ordering is important: from right to left, so that the CFs could inform each other if cluster was found
         */
        for (auto rit = clusterFinder[iRow].crbegin(); rit != clusterFinder[iRow].crend(); ++rit) {
          if ((*rit)->findCluster()) {
            cfWithCluster.push_back(rit);
          }
        }
      }
    }

    /*
     * add empty timebins to find last clusters
     */
    if (!config.iIsContinuousReadout) {
      // +2 so that for sure all data is processed
      for (time = 0; time < clusterFinder[iRow][0]->getNtimebins()+2; ++time){
        for (auto rit = clusterFinder[iRow].crbegin(); rit != clusterFinder[iRow].crend(); ++rit) {
          (*rit)->AddZeroTimebin(time+timeDiff+config.iMinTimeBin,iPadsPerCF);
        }

        /*
         * search for clusters and store reference to CF if one was found
         */
        if (clusterFinder[iRow][0]->getTimebinsAfterLastProcessing() == iTimebinsPerCF-2 -2)  {
          /*
           * ordering is important: from right to left, so that the CFs could inform each other if cluster was found
           */
          for (auto rit = clusterFinder[iRow].crbegin(); rit != clusterFinder[iRow].crend(); ++rit) {
            if ((*rit)->findCluster()) {
              cfWithCluster.push_back(rit);
            }
          }
        }
      }
      for (auto rit = clusterFinder[iRow].crbegin(); rit != clusterFinder[iRow].crend(); ++rit) {
        (*rit)->setTimebinsAfterLastProcessing(0);
      }
    }

    /*
     * collect found cluster
     */
    for (auto &cf_rit : cfWithCluster) {
      auto cc = (*cf_rit)->getClusterContainer();
      for (auto& c : *cc) cluster.push_back(c);

      auto ll = (*cf_rit)->getClusterDigitIndices();
      for (auto& l : *ll) {
        label.push_back(l);
      }

      (*cf_rit)->clearClusterContainer();
    }

  }

//  g_display_mutex.lock();
//  std::cout << "thread " << this_id << " finished.\n";
//  g_display_mutex.unlock();
}

//________________________________________________________________________
void HwClusterer::Process(std::vector<o2::TPC::Digit> const &digits, MCLabelContainer const* mcDigitTruth, int eventCount)
{
  mClusterArray->clear();
  mClusterMcLabelArray->clear();


  /*
   * clear old storages
   */
  for (auto& cs : mClusterStorage) cs.clear();
  for (auto& cdis : mClusterDigitIndexStorage) cdis.clear();
  for (auto& dc : mDigitContainer ) {
    for (auto& dcc : dc) dcc.clear();
  }

  int iTimeBin;
  int iTimeBinMin = (mIsContinuousReadout)?mLastTimebin + 1 : 0;
  //int iTimeBinMin = mLastTimebin + 1;
  int iTimeBinMax = mLastTimebin;

  if (mLastMcDigitTruth.find(eventCount) == mLastMcDigitTruth.end())
    mLastMcDigitTruth[eventCount] = std::make_unique<MCLabelContainer>();
  gsl::span<const o2::MCCompLabel> mcArray;
  /*
   * Loop over digits
   */
  int digitIndex = 0;
  for (const auto& digit : digits) {

    /*
     * add current digit to storage
     */

    iTimeBin = digit.getTimeStamp();
    if (digit.getCRU() < mCRUMin || digit.getCRU() > mCRUMax) {
      LOG(DEBUG) << "Digit [" << digitIndex << "] is out of CRU range (" << digit.getCRU() << " < " << mCRUMin << " or > " << mCRUMax << ")" << FairLogger::endl;
      if (mcDigitTruth != nullptr) mLastMcDigitTruth[eventCount]->addElement(digitIndex,o2::MCCompLabel(-1,-1,-1));
      ++digitIndex;
      continue;
    }
    if (iTimeBin < iTimeBinMin) {
      LOG(DEBUG) << "Digit [" << digitIndex << "] time stamp too small (" << iTimeBin << " < " << iTimeBinMin << ")" << FairLogger::endl;
      if (mcDigitTruth != nullptr) mLastMcDigitTruth[eventCount]->addElement(digitIndex,o2::MCCompLabel(-1,-1,-1));
      ++digitIndex;
      continue;
    }

    iTimeBinMax = std::max(iTimeBinMax,iTimeBin);
    if (mcDigitTruth == nullptr)
      mDigitContainer[digit.getCRU()][digit.getRow()].emplace_back(std::make_tuple(&digit,-1,eventCount));
    else {
      mDigitContainer[digit.getCRU()][digit.getRow()].emplace_back(std::make_tuple(&digit,digitIndex,eventCount));
      mcArray = mcDigitTruth->getLabels(digitIndex);
      for (auto &l : mcArray) mLastMcDigitTruth[eventCount]->addElement(digitIndex,l);
    }
    ++digitIndex;
  }

  ProcessTimeBins(iTimeBinMin, iTimeBinMax, mcDigitTruth, eventCount);

  mLastMcDigitTruth.erase(eventCount-mTimebinsPerCF);
}

//________________________________________________________________________
void HwClusterer::Process(std::vector<std::unique_ptr<Digit>>& digits, MCLabelContainer const* mcDigitTruth, int eventCount)
{
  mClusterArray->clear();
  mClusterMcLabelArray->clear();

  /*
   * clear old storages
   */
  for (auto& cs : mClusterStorage) cs.clear();
  for (auto& cdis : mClusterDigitIndexStorage) cdis.clear();
  for (auto& dc : mDigitContainer ) {
    for (auto& dcc : dc) dcc.clear();
  }

  int iTimeBin;
  int iTimeBinMin = (mIsContinuousReadout)?mLastTimebin + 1 : 0;
  int iTimeBinMax = mLastTimebin;

  if (mLastMcDigitTruth.find(eventCount) == mLastMcDigitTruth.end())
    mLastMcDigitTruth[eventCount] = std::make_unique<MCLabelContainer>();
  gsl::span<const o2::MCCompLabel> mcArray;

  /*
   * Loop over digits
   */
  int digitIndex = 0;
  for (auto& digit_ptr : digits) {
    Digit* digit = digit_ptr.get();

    /*
     * add current digit to storage
     */
    iTimeBin = digit->getTimeStamp();
    if (digit->getCRU() < mCRUMin || digit->getCRU() > mCRUMax) {
      LOG(DEBUG) << "Digit [" << digitIndex << "] is out of CRU range (" << digit->getCRU() << " < " << mCRUMin << " or > " << mCRUMax << ")" << FairLogger::endl;
      if (mcDigitTruth != nullptr) mLastMcDigitTruth[eventCount]->addElement(digitIndex,o2::MCCompLabel(-1,-1,-1));
      ++digitIndex;
      continue;
    }
    if (iTimeBin < iTimeBinMin) {
      LOG(DEBUG) << "Digit [" << digitIndex << "] time stamp too small (" << iTimeBin << " < " << iTimeBinMin << ")" << FairLogger::endl;
      if (mcDigitTruth != nullptr) mLastMcDigitTruth[eventCount]->addElement(digitIndex,o2::MCCompLabel(-1,-1,-1));
      ++digitIndex;
      continue;
    }

    iTimeBinMax = std::max(iTimeBinMax,iTimeBin);
    if (mcDigitTruth == nullptr)
      mDigitContainer[digit->getCRU()][digit->getRow()].emplace_back(std::make_tuple(digit,-1,eventCount));
    else {
      mDigitContainer[digit->getCRU()][digit->getRow()].emplace_back(std::make_tuple(digit,digitIndex,eventCount));
      mcArray = mcDigitTruth->getLabels(digitIndex);
      for (auto &l : mcArray) mLastMcDigitTruth[eventCount]->addElement(digitIndex,l);
    }
    ++digitIndex;
  }

  ProcessTimeBins(iTimeBinMin, iTimeBinMax, mcDigitTruth, eventCount);

  mLastMcDigitTruth.erase(eventCount-mTimebinsPerCF);

}

void HwClusterer::ProcessTimeBins(int iTimeBinMin, int iTimeBinMax, MCLabelContainer const* mcDigitTruth, int eventCount)
{

   /*
   * vector to store all threads for parallel processing
   * one thread per CRU (360 in total)
   */
  std::vector<std::thread> thread_vector;
  if (mProcessingType == Processing::Parallel)
    LOG(DEBUG) << std::thread::hardware_concurrency() << " concurrent threads are supported." << FairLogger::endl;

  /*
   * if CRU number of current digit changes, start processing (either
   * sequential or parallel) of all CRUs in between the last processed
   * one and the current one.
   */

  const Mapper& mapper = Mapper::instance();
  for (int iCRU = mCRUMin; iCRU <= mCRUMax; ++iCRU) {
    struct CfConfig cfConfig = {
      iCRU,
      mapper.getNumberOfRowsPartition(iCRU),
      mPadsMax+2+2,
      iTimeBinMin,
      iTimeBinMax,
      mEnableNoiseSim,
      mEnablePedestalSubtraction,
      mIsContinuousReadout,
      mNoiseObject,
      mPedestalObject
    };
//    LOG(DEBUG) << "Start processing CRU " << iCRU << FairLogger::endl;
    if (mProcessingType == Processing::Parallel)
      thread_vector.emplace_back(
            processDigits,                      // function name
            std::ref(mDigitContainer[iCRU]),    // digit container for individual CRUs
            std::ref(mClusterFinder[iCRU]),     // cluster finder for individual CRUs
            std::ref(mClusterStorage[iCRU]),    // container to store found clusters
            std::ref(mClusterDigitIndexStorage[iCRU]),    // container to store found cluster MC Labels
            cfConfig
        );
    else {
      processDigits(
          std::ref(mDigitContainer[iCRU]),
          std::ref(mClusterFinder[iCRU]),
          std::ref(mClusterStorage[iCRU]),
          std::ref(mClusterDigitIndexStorage[iCRU]),
          cfConfig);
    }
  }


  /*
   * wait for threads to join
   */
  for (std::thread& t: thread_vector) {
    t.join();
  }

  /*
   * collect clusters from individual cluster finder
   */
  for (int cru = 0; cru < mClusterStorage.size(); ++cru) {
    std::vector<Cluster>* clustersFromCRU = &mClusterStorage[cru];
    std::vector<std::vector<std::pair<int,int>>>* labelsFromCRU = &mClusterDigitIndexStorage[cru];

    for(int c = 0; c < clustersFromCRU->size(); ++c) {
      const auto clusterPos = mClusterArray->size();
      mClusterArray->emplace_back(clustersFromCRU->at(c));
      for (auto &digitIndex : labelsFromCRU->at(c)) {
        if (digitIndex.first < 0) continue;
        for (auto &l : mLastMcDigitTruth[digitIndex.second]->getLabels(digitIndex.first))
          mClusterMcLabelArray->addElement(clusterPos,l);
      }
    }
  }

  mLastTimebin = iTimeBinMax;
}


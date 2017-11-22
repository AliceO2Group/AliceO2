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
    MCLabelContainer *labelOutput, int cruMin, int cruMax,
    float minQDiff, bool assignChargeUnique, bool enableNoiseSim,
    bool enablePedestalSubtraction, int padsPerCF, int timebinsPerCF,
    int rowsMax, int padsMax, int timeBinsMax, int minQMax,
    bool requirePositiveCharge, bool requireNeighbouringPad)
  : Clusterer(rowsMax,padsMax,timeBinsMax,minQMax,requirePositiveCharge,requireNeighbouringPad)
  , mAssignChargeUnique(assignChargeUnique)
  , mEnableNoiseSim(enableNoiseSim)
  , mEnablePedestalSubtraction(enablePedestalSubtraction)
  , mIsContinuousReadout(true)
  , mLastTimebin(-1)
  , mCRUMin(cruMin)
  , mCRUMax(cruMax)
  , mPadsPerCF(padsPerCF)
  , mTimebinsPerCF(timebinsPerCF)
  , mNumThreads(std::thread::hardware_concurrency())
  , mMinQDiff(minQDiff)
  , mClusterFinder()
  , mDigitContainer()
  , mClusterStorage()
  , mClusterDigitIndexStorage()
  , mClusterArray(clusterOutput)
  , mClusterMcLabelArray(labelOutput)
  , mNoiseObject(nullptr)
  , mPedestalObject(nullptr)
  , mLastMcDigitTruth()
{
  /*
   * initialize all cluster finder
   */
  unsigned iCfPerRow = (unsigned)ceil((double)(mPadsMax+2+2)/(static_cast<int>(mPadsPerCF)-2-2));
  mClusterFinder.resize(mCRUMax+1);
  const Mapper& mapper = Mapper::instance();
  for (unsigned iCRU = mCRUMin; iCRU <= mCRUMax; ++iCRU){
    mClusterFinder[iCRU].resize(mapper.getNumberOfRowsPartition(iCRU));
    for (int iRow = 0; iRow < mapper.getNumberOfRowsPartition(iCRU); ++iRow){
      mClusterFinder[iCRU][iRow].resize(iCfPerRow);
      for (unsigned iCF = 0; iCF < iCfPerRow; ++iCF){
        int padOffset = iCF*(static_cast<int>(mPadsPerCF)-2-2)-2;
        mClusterFinder[iCRU][iRow][iCF] = std::make_shared<HwClusterFinder>(iCRU,iRow,padOffset,mPadsPerCF,mTimebinsPerCF,mMinQDiff,mMinQMax,mRequirePositiveCharge);
        mClusterFinder[iCRU][iRow][iCF]->setAssignChargeUnique(mAssignChargeUnique);


        /*
         * Connect always two CFs to be able to communicate found clusters. So
         * the "right" one can tell the one "on the left" which pads were
         * already used for a cluster.
         */
        if (iCF != 0) {
          mClusterFinder[iCRU][iRow][iCF]->setNextCF(mClusterFinder[iCRU][iRow][iCF-1]);
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
  for (unsigned iCRU = mCRUMin; iCRU <= mCRUMax; ++iCRU)
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
    const std::vector<std::vector<std::vector<std::tuple<Digit const*, int, int>>>>& digits,
    const std::vector<std::vector<std::vector<std::shared_ptr<HwClusterFinder>>>>& clusterFinder,
          std::vector<std::vector<Cluster>>& cluster,
          std::vector<std::vector<std::vector<std::pair<int,int>>>>& label,
    const CfConfig config)
{
//  std::thread::id this_id = std::this_thread::get_id();
//  g_display_mutex.lock();
//  std::cout << "thread " << this_id << " started.\n";
//  g_display_mutex.unlock();

  int timeDiff = (config.iMaxTimeBin+1) - config.iMinTimeBin;
  if (timeDiff < 0) return;
  const Mapper& mapper = Mapper::instance();
  std::vector<std::vector<HwClusterFinder::MiniDigit>> iAllBins(timeDiff,std::vector<HwClusterFinder::MiniDigit>(config.iMaxPads));

  for (unsigned iCRU = config.iCRUMin; iCRU <= config.iCRUMax; ++iCRU) {
    if (iCRU % config.iThreadMax != config.iThreadID) continue;

    for (int iRow = 0; iRow < mapper.getNumberOfRowsPartition(iCRU); iRow++){

      /*
       * prepare local storage
       */
      short t,p;
      if (config.iEnableNoiseSim && config.iNoiseObject != nullptr) {
        for (t=timeDiff; t--;) {
          for (p=config.iMaxPads; p--;) {
            iAllBins[t][p].charge = config.iNoiseObject->getValue(CRU(iCRU),iRow,p);
            iAllBins[t][p].index = -1;
            iAllBins[t][p].event = -1;
          }
        }
      } else {
        for (auto &bins : iAllBins) std::fill(bins.begin(),bins.end(),HwClusterFinder::MiniDigit());
//        std::fill(&iAllBins[0][0], &iAllBins[0][0]+timeDiff*config.iMaxPads, HwClusterFinder::MiniDigit());
      }

      /*
       * fill in digits
       */
      for (auto& digit : digits[iCRU][iRow]){
        const Int_t iTime         = std::get<0>(digit)->getTimeStamp();
        const Int_t iPad          = std::get<0>(digit)->getPad() + 2;  // offset to have 2 empty pads on the "left side"
        const Float_t charge      = std::get<0>(digit)->getChargeFloat();

        //      std::cout << iCRU << " " << iRow << " " << iPad << " " << iTime << " (" << iTime-minTime << "," << timeDiff << ") " << charge << std::endl;
        iAllBins[iTime-config.iMinTimeBin][iPad].charge += charge;
        iAllBins[iTime-config.iMinTimeBin][iPad].index = std::get<1>(digit);
        iAllBins[iTime-config.iMinTimeBin][iPad].event = std::get<2>(digit);
        if (config.iEnablePedestalSubtraction && config.iPedestalObject != nullptr) {
          const float pedestal = config.iPedestalObject->getValue(CRU(iCRU),iRow,iPad-2);
          //printf("digit: %.2f, pedestal: %.2f\n", iAllBins[iTime-config.iMinTimeBin][iPad], pedestal);
          iAllBins[iTime-config.iMinTimeBin][iPad].charge -= pedestal;
        }
      }

      /*
       * copy data to cluster finders
       */
      const unsigned iPadsPerCF = static_cast<const unsigned>(clusterFinder[iCRU][iRow][0]->getNpads());
      const unsigned iTimebinsPerCF = static_cast<const unsigned>(clusterFinder[iCRU][iRow][0]->getNtimebins());
      std::vector<std::vector<std::shared_ptr<HwClusterFinder>>::const_reverse_iterator> cfWithCluster;
      int time;
      unsigned pad;
      for (time = 0; time < timeDiff; ++time){    // ordering important!!
        for (pad = 0; pad < config.iMaxPads; pad = pad + (iPadsPerCF -2 -2 )) {
          const Short_t cf = pad / (iPadsPerCF-2-2);
          clusterFinder[iCRU][iRow][cf]->addTimebin(iAllBins[time].begin()+pad,time+config.iMinTimeBin,(config.iMaxPads-pad)>=iPadsPerCF?iPadsPerCF:(config.iMaxPads-pad));
        }

        /*
         * search for clusters and store reference to CF if one was found
         */
        if (clusterFinder[iCRU][iRow][0]->getTimebinsAfterLastProcessing() == iTimebinsPerCF-2 -2)  {
          /*
           * ordering is important: from right to left, so that the CFs could inform each other if cluster was found
           */
          for (auto rit = clusterFinder[iCRU][iRow].crbegin(); rit != clusterFinder[iCRU][iRow].crend(); ++rit) {
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
        for (time = 0; time < clusterFinder[iCRU][iRow][0]->getNtimebins()+2; ++time){
          for (auto rit = clusterFinder[iCRU][iRow].crbegin(); rit != clusterFinder[iCRU][iRow].crend(); ++rit) {
            (*rit)->addZeroTimebin(time+timeDiff+config.iMinTimeBin,iPadsPerCF);
          }

          /*
           * search for clusters and store reference to CF if one was found
           */
          if (clusterFinder[iCRU][iRow][0]->getTimebinsAfterLastProcessing() == iTimebinsPerCF-2 -2)  {
            /*
             * ordering is important: from right to left, so that the CFs could inform each other if cluster was found
             */
            for (auto rit = clusterFinder[iCRU][iRow].crbegin(); rit != clusterFinder[iCRU][iRow].crend(); ++rit) {
              if ((*rit)->findCluster()) {
                cfWithCluster.push_back(rit);
              }
            }
          }
        }
        for (auto rit = clusterFinder[iCRU][iRow].crbegin(); rit != clusterFinder[iCRU][iRow].crend(); ++rit) {
          (*rit)->setTimebinsAfterLastProcessing(0);
        }
      }

      /*
       * collect found cluster
       */
      for (auto &cf_rit : cfWithCluster) {
        auto cc = (*cf_rit)->getClusterContainer();
        for (auto& c : *cc) cluster[iCRU].push_back(c);

        auto ll = (*cf_rit)->getClusterDigitIndices();
        for (auto& l : *ll) {
          label[iCRU].push_back(l);
        }

        (*cf_rit)->clearClusterContainer();
      }

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
  if(mClusterMcLabelArray) mClusterMcLabelArray->clear();


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

  /*
   * Loop over digits
   */
  int digitIndex = 0;
  for (const auto& digit : digits) {

    /*
     * add current digit to storage
     */

    iTimeBin = digit.getTimeStamp();
    if (digit.getCRU() < static_cast<int>(mCRUMin) || digit.getCRU() > static_cast<int>(mCRUMax)) {
      LOG(DEBUG) << "Digit [" << digitIndex << "] is out of CRU range (" << digit.getCRU() << " < " << mCRUMin << " or > " << mCRUMax << ")" << FairLogger::endl;
      // Necessary because MCTruthContainer requires continuous indexing
      ++digitIndex;
      continue;
    }
    if (iTimeBin < iTimeBinMin) {
      LOG(DEBUG) << "Digit [" << digitIndex << "] time stamp too small (" << iTimeBin << " < " << iTimeBinMin << ")" << FairLogger::endl;
      // Necessary because MCTruthContainer requires continuous indexing
      ++digitIndex;
      continue;
    }

    iTimeBinMax = std::max(iTimeBinMax,iTimeBin);
    if (mcDigitTruth == nullptr)
      mDigitContainer[digit.getCRU()][digit.getRow()].emplace_back(std::make_tuple(&digit,-1,eventCount));
    else {
      mDigitContainer[digit.getCRU()][digit.getRow()].emplace_back(std::make_tuple(&digit,digitIndex,eventCount));
    }
    ++digitIndex;
  }

  if (mcDigitTruth != nullptr && mClusterMcLabelArray != nullptr )
    mLastMcDigitTruth[eventCount] = std::make_unique<MCLabelContainer>(*mcDigitTruth);

  ProcessTimeBins(iTimeBinMin, iTimeBinMax);

  mLastMcDigitTruth.erase(eventCount-mTimebinsPerCF);

  LOG(DEBUG) << "Event ranged from time bin " << iTimeBinMin << " to " << iTimeBinMax << "." << FairLogger::endl;
}

//________________________________________________________________________
void HwClusterer::Process(std::vector<std::unique_ptr<Digit>>& digits, MCLabelContainer const* mcDigitTruth, int eventCount)
{
  mClusterArray->clear();
  if(mClusterMcLabelArray) mClusterMcLabelArray->clear();

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
    if (digit->getCRU() < static_cast<int>(mCRUMin) || digit->getCRU() > static_cast<int>(mCRUMax)) {
      LOG(DEBUG) << "Digit [" << digitIndex << "] is out of CRU range (" << digit->getCRU() << " < " << mCRUMin << " or > " << mCRUMax << ")" << FairLogger::endl;
      // Necessary because MCTruthContainer requires continuous indexing
      ++digitIndex;
      continue;
    }
    if (iTimeBin < iTimeBinMin) {
      LOG(DEBUG) << "Digit [" << digitIndex << "] time stamp too small (" << iTimeBin << " < " << iTimeBinMin << ")" << FairLogger::endl;
      // Necessary because MCTruthContainer requires continuous indexing
      ++digitIndex;
      continue;
    }

    iTimeBinMax = std::max(iTimeBinMax,iTimeBin);
    if (mcDigitTruth == nullptr)
      mDigitContainer[digit->getCRU()][digit->getRow()].emplace_back(std::make_tuple(digit,-1,eventCount));
    else {
      mDigitContainer[digit->getCRU()][digit->getRow()].emplace_back(std::make_tuple(digit,digitIndex,eventCount));
    }
    ++digitIndex;
  }

  if (mcDigitTruth != nullptr && mClusterMcLabelArray != nullptr )
    mLastMcDigitTruth[eventCount] = std::make_unique<MCLabelContainer>(*mcDigitTruth);

  ProcessTimeBins(iTimeBinMin, iTimeBinMax);

  mLastMcDigitTruth.erase(eventCount-mTimebinsPerCF);

  LOG(DEBUG) << "Event ranged from time bin " << iTimeBinMin << " to " << iTimeBinMax << "." << FairLogger::endl;
}

void HwClusterer::ProcessTimeBins(int iTimeBinMin, int iTimeBinMax)
{

   /*
   * vector to store threads for parallel processing
   */
  std::vector<std::thread> thread_vector;

  LOG(DEBUG) << "Starting " << mNumThreads << " threads, hardware supports " << std::thread::hardware_concurrency() << " parallel threads." << FairLogger::endl;

  for (unsigned threadId = 0; threadId < std::min(mNumThreads,mCRUMax); ++threadId) {
    struct CfConfig cfConfig = {
      threadId,
      mNumThreads,
      mCRUMin,
      mCRUMax,
      static_cast<unsigned>(mPadsMax)+2+2,
      iTimeBinMin,
      iTimeBinMax,
      mEnableNoiseSim,
      mEnablePedestalSubtraction,
      mIsContinuousReadout,
      mNoiseObject,
      mPedestalObject
    };
    thread_vector.emplace_back(
        processDigits,                          // function name
        std::ref(mDigitContainer),              // digit container for individual CRUs
        std::ref(mClusterFinder),               // cluster finder for individual CRUs
        std::ref(mClusterStorage),              // container to store found clusters
        std::ref(mClusterDigitIndexStorage),    // container to store found cluster MC Labels
        cfConfig                                // configuration
        );
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

  // map to count unique MC labels
  std::map<MCCompLabel,int> labelCount;

  // multiset to sort labels according to occurrence
  auto mcComp = [](const std::pair<MCCompLabel, int>& a, const std::pair<MCCompLabel, int>& b) { return a.second > b.second;};
  std::multiset<std::pair<MCCompLabel,int>,decltype(mcComp)> labelSort(mcComp);

  // for each CRU
  for (unsigned cru = 0; cru < mClusterStorage.size(); ++cru) {
    std::vector<Cluster>* clustersFromCRU = &mClusterStorage[cru];
    std::vector<std::vector<std::pair<int,int>>>* labelsFromCRU = &mClusterDigitIndexStorage[cru];

    // for each found cluster
    for(unsigned c = 0; c < clustersFromCRU->size(); ++c) {
      const auto clusterPos = mClusterArray->size();
      mClusterArray->emplace_back(clustersFromCRU->at(c));
      if (mClusterMcLabelArray == nullptr) continue;
      labelCount.clear();
      labelSort.clear();

      // for each used digit
      for (auto &digitIndex : labelsFromCRU->at(c)) {
        if (digitIndex.first < 0) continue;
        for (auto &l : mLastMcDigitTruth[digitIndex.second]->getLabels(digitIndex.first)) {
          labelCount[l]++;
        }
      }
      for (auto &l : labelCount) labelSort.insert(l);
      for (auto &l : labelSort) mClusterMcLabelArray->addElement(clusterPos,l.first);
    }
  }

  mLastTimebin = iTimeBinMax;
}

void HwClusterer::setNumThreads(unsigned threads) {
  mNumThreads = (threads == 0) ? std::thread::hardware_concurrency() : threads;
}


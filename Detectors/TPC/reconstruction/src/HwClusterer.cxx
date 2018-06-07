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
#include "TPCBase/Digit.h"
#include "TPCBase/PadPos.h"
#include "TPCBase/CRU.h"
#include "TPCBase/PadSecPos.h"
#include "TPCBase/CalArray.h"
#include "TPCBase/Mapper.h"

#include "FairLogger.h"
#include "TMath.h"

#include <set>
#include <map>
#include <thread>
#include <limits>

using namespace o2::TPC;

//________________________________________________________________________
HwClusterer::HwClusterer(
    std::shared_ptr<std::vector<ClusterHardwareContainer8kb>> clusterOutput,
    std::shared_ptr<MCLabelContainer> labelOutput, int sectorid)
  : mClusterSector(sectorid),
    mNumRows(0),
    mLastTimebin(-1),
    mLastHB(0),
    mPeakChargeThreshold(2),
    mContributionChargeThreshold(0),
    mRequireNeighbouringTimebin(false),
    mRequireNeighbouringPad(false),
    mIsContinuousReadout(true),
    mPadsPerRow(),
    mGlobalRowToRegion(),
    mGlobalRowToLocalRow(),
    mDataBuffer(),
    mIndexBuffer(),
    mMCtruth(),
    mTmpClusterArray(),
    mClusterArray(clusterOutput),
    mClusterMcLabelArray(labelOutput)
//  : Clusterer(rowsMax,padsMax,timeBinsMax,minQMax,requirePositiveCharge,requireNeighbouringPad)
//  , mAssignChargeUnique(assignChargeUnique)
//  , mEnableNoiseSim(enableNoiseSim)
//  , mEnablePedestalSubtraction(enablePedestalSubtraction)
//  , mLastTimebin(-1)
//  , mCRUMin(cruMin)
//  , mCRUMax(cruMax)
//  , mPadsPerCF(padsPerCF)
//  , mTimebinsPerCF(timebinsPerCF)
//  , mNumThreads(std::thread::hardware_concurrency())
//  , mMinQDiff(minQDiff)
//  , mClusterFinder()
//  , mDigitContainer()
//  , mClusterStorage()
//  , mClusterDigitIndexStorage()
//  , mNoiseObject(nullptr)
//  , mPedestalObject(nullptr)
//  , mLastMcDigitTruth()
{
  LOG(DEBUG) << "Enter Initializer of HwClusterer" << FairLogger::endl;
  /*
   * Prepare temporary storage for digits
   */
  Mapper &mapper = Mapper::instance();

  mNumRows = mapper.getNumberOfRows();
  mDataBuffer.resize(mNumRows);
  mIndexBuffer.resize(mNumRows);
  mPadsPerRow.resize(mNumRows);

  for (unsigned short row = 0; row < mNumRows; ++row) {
    // add two empty pads on the left and on the right
    mPadsPerRow[row] = mapper.getNumberOfPadsInRowSector(row) +2 +2;

    // prepare for 5 timebins
    mDataBuffer[row].resize(mPadsPerRow[row]*5,0);
    mIndexBuffer[row].resize(mPadsPerRow[row]*5,-1);
  }

  mTmpClusterArray.resize(10);
  for (unsigned short region = 0; region < 10; ++region) {
    mTmpClusterArray[region] = std::make_unique<std::vector<std::pair<std::shared_ptr<ClusterHardware>,std::shared_ptr<std::vector<std::pair<MCCompLabel,unsigned>>>>>>();
  }

  mGlobalRowToRegion.resize(mNumRows);
  mGlobalRowToLocalRow.resize(mNumRows);
  unsigned short row = 0;
  for (unsigned short region = 0; region < 10; ++region) {
    for (unsigned short localRow = 0; localRow < mapper.getNumberOfRowsRegion(region); ++localRow) {
      mGlobalRowToRegion[row] = region;
      mGlobalRowToLocalRow[row] = localRow;
      ++row;
    }
  }
  mMCtruth.resize(5,nullptr);
//  /*
//   * initialize all cluster finder
//   */
//  unsigned iCfPerRow = (unsigned)ceil(static_cast<double>(mPadsMax+2+2)/(static_cast<int>(mPadsPerCF)-2));
//  mClusterFinder.resize(mCRUMax+1);
//  const Mapper& mapper = Mapper::instance();
//  for (unsigned iCRU = mCRUMin; iCRU <= mCRUMax; ++iCRU){
//    mClusterFinder[iCRU].resize(mapper.getNumberOfRowsPartition(iCRU));
//    for (int iRow = 0; iRow < mapper.getNumberOfRowsPartition(iCRU); ++iRow){
//      mClusterFinder[iCRU][iRow].resize(iCfPerRow);
//      for (unsigned iCF = 0; iCF < iCfPerRow; ++iCF){
//        int padOffset = iCF*(static_cast<int>(mPadsPerCF)-2-2)-2;
//        mClusterFinder[iCRU][iRow][iCF] = std::make_shared<HwClusterFinder>(iCRU,iRow,padOffset,mPadsPerCF,mTimebinsPerCF,mMinQDiff,mMinQMax,mRequirePositiveCharge);
//        mClusterFinder[iCRU][iRow][iCF]->setAssignChargeUnique(mAssignChargeUnique);
//
//
//        /*
//         * Connect always two CFs to be able to communicate found clusters. So
//         * the "right" one can tell the one "on the left" which pads were
//         * already used for a cluster.
//         */
//        if (iCF != 0) {
//          mClusterFinder[iCRU][iRow][iCF]->setNextCF(mClusterFinder[iCRU][iRow][iCF-1]);
//        }
//      }
//    }
//  }
//
//
//  /*
//   * vector of HwCluster vectors, one vector for each CRU (possible thread)
//   * to store the clusters found there
//   */
//  mClusterStorage.resize(mCRUMax+1);
//  mClusterDigitIndexStorage.resize(mCRUMax+1);
//
//
//  /*
//   * vector of digit vectors, one vector for each CRU (possible thread) to
//   * store there only those digits which are relevant for this particular
//   * CRU (thread)
//   */
//  mDigitContainer.resize(mCRUMax+1);
//  for (unsigned iCRU = mCRUMin; iCRU <= mCRUMax; ++iCRU)
//    mDigitContainer[iCRU].resize(mapper.getNumberOfRowsPartition(iCRU));
//
}

////________________________________________________________________________
//void HwClusterer::processDigits(
//    const std::vector<std::vector<std::vector<std::tuple<Digit const*, int, int>>>>& digits,
//    const std::vector<std::vector<std::vector<std::shared_ptr<HwClusterFinder>>>>& clusterFinder,
//          std::vector<std::vector<Cluster>>& cluster,
//          std::vector<std::vector<std::vector<std::pair<int,int>>>>& label,
//    const CfConfig config)
//{
////  std::thread::id this_id = std::this_thread::get_id();
////  g_display_mutex.lock();
////  std::cout << "thread " << this_id << " started.\n";
////  g_display_mutex.unlock();
//
//  int timeDiff = (config.iMaxTimeBin+1) - config.iMinTimeBin;
//  if (timeDiff < 0) return;
//  const Mapper& mapper = Mapper::instance();
//  std::vector<std::vector<HwClusterFinder::MiniDigit>> iAllBins(timeDiff,std::vector<HwClusterFinder::MiniDigit>(config.iMaxPads));
//
//  for (unsigned iCRU = config.iCRUMin; iCRU <= config.iCRUMax; ++iCRU) {
//    if (iCRU % config.iThreadMax != config.iThreadID) continue;
//
//    for (int iRow = 0; iRow < mapper.getNumberOfRowsPartition(iCRU); iRow++){
//
//      /*
//       * prepare local storage
//       */
//      short t,p;
//      if (config.iEnableNoiseSim && config.iNoiseObject != nullptr) {
//        for (t=timeDiff; t--;) {
//          for (p=config.iMaxPads; p--;) {
//            iAllBins[t][p].charge = config.iNoiseObject->getValue(CRU(iCRU),iRow,p);
//            iAllBins[t][p].index = -1;
//            iAllBins[t][p].event = -1;
//          }
//        }
//      } else {
//        for (auto &bins : iAllBins) std::fill(bins.begin(),bins.end(),HwClusterFinder::MiniDigit());
////        std::fill(&iAllBins[0][0], &iAllBins[0][0]+timeDiff*config.iMaxPads, HwClusterFinder::MiniDigit());
//      }
//
//      /*
//       * fill in digits
//       */
//      for (auto& digit : digits[iCRU][iRow]){
//        const Int_t iTime         = std::get<0>(digit)->getTimeStamp();
//        const Int_t iPad          = std::get<0>(digit)->getPad() + 2;  // offset to have 2 empty pads on the "left side"
//        const Float_t charge      = std::get<0>(digit)->getChargeFloat();
//
//        //      std::cout << iCRU << " " << iRow << " " << iPad << " " << iTime << " (" << iTime-minTime << "," << timeDiff << ") " << charge << std::endl;
//        iAllBins[iTime-config.iMinTimeBin][iPad].charge += charge;
//        iAllBins[iTime-config.iMinTimeBin][iPad].index = std::get<1>(digit);
//        iAllBins[iTime-config.iMinTimeBin][iPad].event = std::get<2>(digit);
//        if (config.iEnablePedestalSubtraction && config.iPedestalObject != nullptr) {
//          const float pedestal = config.iPedestalObject->getValue(CRU(iCRU),iRow,iPad-2);
//          //printf("digit: %.2f, pedestal: %.2f\n", iAllBins[iTime-config.iMinTimeBin][iPad], pedestal);
//          iAllBins[iTime-config.iMinTimeBin][iPad].charge -= pedestal;
//        }
//      }
//
//      /*
//       * copy data to cluster finders
//       */
//      const unsigned iPadsPerCF = static_cast<const unsigned>(clusterFinder[iCRU][iRow][0]->getNpads());
//      const unsigned iTimebinsPerCF = static_cast<const unsigned>(clusterFinder[iCRU][iRow][0]->getNtimebins());
//      std::vector<std::vector<std::shared_ptr<HwClusterFinder>>::const_reverse_iterator> cfWithCluster;
//      int time;
//      unsigned pad;
//      for (time = 0; time < timeDiff; ++time){    // ordering important!!
//        for (short cf = 0; cf < clusterFinder[iCRU][iRow].size(); ++cf) {
//          pad = cf*(iPadsPerCF - 2);
//          clusterFinder[iCRU][iRow][cf]->addTimebin(iAllBins[time].begin()+pad,time+config.iMinTimeBin,(config.iMaxPads-pad)>=iPadsPerCF?iPadsPerCF:(config.iMaxPads-pad));
//        }
//
//        /*
//         * search for clusters and store reference to CF if one was found
//         */
//        if (clusterFinder[iCRU][iRow][0]->getTimebinsAfterLastProcessing() == iTimebinsPerCF-2 -2)  {
//          /*
//           * ordering is important: from right to left, so that the CFs could inform each other if cluster was found
//           */
//          for (auto rit = clusterFinder[iCRU][iRow].crbegin(); rit != clusterFinder[iCRU][iRow].crend(); ++rit) {
//            if ((*rit)->findCluster()) {
//              cfWithCluster.push_back(rit);
//            }
//          }
//        }
//      }
//
//      /*
//       * add empty timebins to find last clusters
//       */
//      if (!config.iIsContinuousReadout) {
//        // +2 so that for sure all data is processed
//        for (time = 0; time < clusterFinder[iCRU][iRow][0]->getNtimebins()+2; ++time){
//          for (auto rit = clusterFinder[iCRU][iRow].crbegin(); rit != clusterFinder[iCRU][iRow].crend(); ++rit) {
//            (*rit)->addZeroTimebin(time+timeDiff+config.iMinTimeBin,iPadsPerCF);
//          }
//
//          /*
//           * search for clusters and store reference to CF if one was found
//           */
//          if (clusterFinder[iCRU][iRow][0]->getTimebinsAfterLastProcessing() == iTimebinsPerCF-2 -2)  {
//            /*
//             * ordering is important: from right to left, so that the CFs could inform each other if cluster was found
//             */
//            for (auto rit = clusterFinder[iCRU][iRow].crbegin(); rit != clusterFinder[iCRU][iRow].crend(); ++rit) {
//              if ((*rit)->findCluster()) {
//                cfWithCluster.push_back(rit);
//              }
//            }
//          }
//        }
//        for (auto rit = clusterFinder[iCRU][iRow].crbegin(); rit != clusterFinder[iCRU][iRow].crend(); ++rit) {
//          (*rit)->setTimebinsAfterLastProcessing(0);
//        }
//      }
//
//      /*
//       * collect found cluster
//       */
//      for (auto &cf_rit : cfWithCluster) {
//        auto cc = (*cf_rit)->getClusterContainer();
//        for (auto& c : *cc) cluster[iCRU].push_back(c);
//
//        auto ll = (*cf_rit)->getClusterDigitIndices();
//        for (auto& l : *ll) {
//          label[iCRU].push_back(l);
//        }
//
//        (*cf_rit)->clearClusterContainer();
//      }
//
//    }
//  }
//
////  g_display_mutex.lock();
////  std::cout << "thread " << this_id << " finished.\n";
////  g_display_mutex.unlock();
//}

//________________________________________________________________________
void HwClusterer::Process(std::shared_ptr<const std::vector<o2::TPC::Digit>> digits, std::shared_ptr<const MCLabelContainer> mcDigitTruth, int eventCount)
{
  mClusterArray->clear();
  mClusterMcLabelArray->clear();

//  if(mClusterMcLabelArray) mClusterMcLabelArray->clear();
//
//
//  /*
//   * clear old storages
//   */
//  for (auto& cs : mClusterStorage) cs.clear();
//  for (auto& cdis : mClusterDigitIndexStorage) cdis.clear();
//  for (auto& dc : mDigitContainer ) {
//    for (auto& dcc : dc) dcc.clear();
//  }
//
//  int iTimeBin;
//  int iTimeBinMin = (mIsContinuousReadout)?mLastTimebin + 1 : 0;
//  //int iTimeBinMin = mLastTimebin + 1;
//  int iTimeBinMax = mLastTimebin;
//
//  /*
//   * Loop over digits
//   */
  int digitIndex = 0;
  int index;
  int HB;

  /*
   * Loop over all (time ordered) digits
   */
  for (const auto& digit : *digits) {
    /*
     * This loop does the following:
     *  - add digits to the tmp storage
     *  - look for clusters
     *  - fill cluster output
     *
     * Before adding current digit to the storage, the new timebin has to be
     * prepared first, by setting it completely 0, or to pedestal or ...
     *
     * This needs to be done only of the timestamps changes, otherwise it was
     * already done.
     */

    if (digit.getTimeStamp() != mLastTimebin) {
      mMCtruth[digit.getTimeStamp()%5] = mcDigitTruth;

      /*
       * If the timebin changes, it could change by more then just 1 (not every
       * timebin has digits). Since the tmp storage covers 5 timebins, at most
       * 5 new timebins need to be prepared and checked for clusters.
       */
      for (int i = mLastTimebin+1; (i <= digit.getTimeStamp()) && (i-(mLastTimebin+1) < 5); ++i) {

        /*
         * If the HB of the cluster which will be found in a few lines, NOT the
         * current timebin is a new one, we have to fill the  output container
         * with the so far found clusters. Because cluster center and timebin
         * have an offset of two with respect to each other (see next comment),
         * the HB is calculated with (i-2). By the way, it is not possible, that
         * a cluster is found with a negative HB, because at least 2 timebins
         * have to be filled to be able to find a cluster.
         */
        HB = (i-2)/447; // integer division on purpose
        if (HB != mLastHB) writeOutputForTimeOffset(i-2);

        /*
         * For each row, we first check for cluster peaks in the timebin i-2,
         * or because (i-2) mod 5 = (i+3) mod 5, in i+3, to ensure positive times.
         *
         * If timebin 4 is the oldest one (and has to be replaced by the new
         * arriving one, the cluster which still could be ranging from timebin
         * 0 to 4 has to found, which must have its center at timebin 2.
         *       ---------
         *    0 |
         *    1 |
         *    2 | XXXXXX
         *    3 |
         * -> 4 |
         *       ---------
         */
        findClusterForTime(i+3);

        clearBuffer(i);

        mLastHB = HB;
      }
    }

    /*
     * add current digit to storage
     */
    index = (digit.getTimeStamp()%5)*mPadsPerRow[digit.getRow()]+digit.getPad();
    mDataBuffer[digit.getRow()][index] += static_cast<unsigned>(digit.getChargeFloat()*(1<<4));
    mIndexBuffer[digit.getRow()][index] = digitIndex++;

//    if (mcDigitTruth != nullptr) {
//      for (auto &label : mcDigitTruth->getLabels(digitIndex)) {
//        std::cout << digitIndex << " " << label << std::endl;
//      }
//    }

    mLastTimebin = digit.getTimeStamp();
//    iTimeBin = digit.getTimeStamp();
//    if (digit.getCRU() < static_cast<int>(mCRUMin) || digit.getCRU() > static_cast<int>(mCRUMax)) {
//      LOG(DEBUG) << "Digit [" << digitIndex << "] is out of CRU range (" << digit.getCRU() << " < " << mCRUMin << " or > " << mCRUMax << ")" << FairLogger::endl;
//      // Necessary because MCTruthContainer requires continuous indexing
//      ++digitIndex;
//      continue;
//    }
//    if (iTimeBin < iTimeBinMin) {
//      LOG(DEBUG) << "Digit [" << digitIndex << "] time stamp too small (" << iTimeBin << " < " << iTimeBinMin << ")" << FairLogger::endl;
//      // Necessary because MCTruthContainer requires continuous indexing
//      ++digitIndex;
//      continue;
//    }
//
//    iTimeBinMax = std::max(iTimeBinMax,iTimeBin);
//    if (mcDigitTruth == nullptr)
//      mDigitContainer[digit.getCRU()][digit.getRow()].emplace_back(std::make_tuple(&digit,-1,eventCount));
//    else {
//      mDigitContainer[digit.getCRU()][digit.getRow()].emplace_back(std::make_tuple(&digit,digitIndex,eventCount));
//    }
  }

  if (!mIsContinuousReadout)
    finishFrame(true);

//
//  if (mcDigitTruth != nullptr && mClusterMcLabelArray != nullptr )
//    mLastMcDigitTruth[eventCount] = std::make_unique<MCLabelContainer>(*mcDigitTruth);
//
//  ProcessTimeBins(iTimeBinMin, iTimeBinMax);
//
//  mLastMcDigitTruth.erase(eventCount-mTimebinsPerCF);
//
  if (digits->size() != 0)
    LOG(DEBUG) << "Event ranged from time bin " << digits->front().getTimeStamp() << " to " << digits->back().getTimeStamp() << "." << FairLogger::endl;
}

//________________________________________________________________________
void HwClusterer::FinishProcess(std::shared_ptr<const std::vector<o2::TPC::Digit>> digits, std::shared_ptr<const MCLabelContainer> mcDigitTruth, int eventCount)
{
  // Process the last digits (if there are any)
  Process(digits, mcDigitTruth, eventCount);

  // Search in last remaining timebins
  finishFrame();
}
//________________________________________________________________________
//void HwClusterer::Process(std::vector<std::unique_ptr<Digit>>* digits, MCLabelContainer const* mcDigitTruth, int eventCount)
//{
//  mClusterArray->clear();
//  if(mClusterMcLabelArray) mClusterMcLabelArray->clear();
//
//  /*
//   * clear old storages
//   */
//  for (auto& cs : mClusterStorage) cs.clear();
//  for (auto& cdis : mClusterDigitIndexStorage) cdis.clear();
//  for (auto& dc : mDigitContainer ) {
//    for (auto& dcc : dc) dcc.clear();
//  }
//
//  int iTimeBin;
//  int iTimeBinMin = (mIsContinuousReadout)?mLastTimebin + 1 : 0;
//  int iTimeBinMax = mLastTimebin;
//
//  /*
//   * Loop over digits
//   */
//  int digitIndex = 0;
//  for (auto& digit_ptr : *digits) {
//    Digit* digit = digit_ptr.get();
//
//    /*
//     * add current digit to storage
//     */
//    iTimeBin = digit->getTimeStamp();
//    if (digit->getCRU() < static_cast<int>(mCRUMin) || digit->getCRU() > static_cast<int>(mCRUMax)) {
//      LOG(DEBUG) << "Digit [" << digitIndex << "] is out of CRU range (" << digit->getCRU() << " < " << mCRUMin << " or > " << mCRUMax << ")" << FairLogger::endl;
//      // Necessary because MCTruthContainer requires continuous indexing
//      ++digitIndex;
//      continue;
//    }
//    if (iTimeBin < iTimeBinMin) {
//      LOG(DEBUG) << "Digit [" << digitIndex << "] time stamp too small (" << iTimeBin << " < " << iTimeBinMin << ")" << FairLogger::endl;
//      // Necessary because MCTruthContainer requires continuous indexing
//      ++digitIndex;
//      continue;
//    }
//
//    iTimeBinMax = std::max(iTimeBinMax,iTimeBin);
//    if (mcDigitTruth == nullptr)
//      mDigitContainer[digit->getCRU()][digit->getRow()].emplace_back(std::make_tuple(digit,-1,eventCount));
//    else {
//      mDigitContainer[digit->getCRU()][digit->getRow()].emplace_back(std::make_tuple(digit,digitIndex,eventCount));
//    }
//    ++digitIndex;
//  }
//
//  if (mcDigitTruth != nullptr && mClusterMcLabelArray != nullptr )
//    mLastMcDigitTruth[eventCount] = std::make_unique<MCLabelContainer>(*mcDigitTruth);
//
//  ProcessTimeBins(iTimeBinMin, iTimeBinMax);
//
//  mLastMcDigitTruth.erase(eventCount-mTimebinsPerCF);
//
//  LOG(DEBUG) << "Event ranged from time bin " << iTimeBinMin << " to " << iTimeBinMax << "." << FairLogger::endl;
//}

//void HwClusterer::ProcessTimeBins(int iTimeBinMin, int iTimeBinMax)
//{
//
//   /*
//   * vector to store threads for parallel processing
//   */
//  std::vector<std::thread> thread_vector;
//
//  LOG(DEBUG) << "Starting " << mNumThreads << " threads, hardware supports " << std::thread::hardware_concurrency() << " parallel threads." << FairLogger::endl;
//
//  for (unsigned threadId = 0; threadId < std::min(mNumThreads, mCRUMax + 1); ++threadId) {
//    struct CfConfig cfConfig = {
//      threadId,
//      mNumThreads,
//      mCRUMin,
//      mCRUMax,
//      static_cast<unsigned>(mPadsMax)+2+2,
//      iTimeBinMin,
//      iTimeBinMax,
//      mEnableNoiseSim,
//      mEnablePedestalSubtraction,
//      mIsContinuousReadout,
//      mNoiseObject,
//      mPedestalObject
//    };
//    thread_vector.emplace_back(
//        processDigits,                          // function name
//        std::ref(mDigitContainer),              // digit container for individual CRUs
//        std::ref(mClusterFinder),               // cluster finder for individual CRUs
//        std::ref(mClusterStorage),              // container to store found clusters
//        std::ref(mClusterDigitIndexStorage),    // container to store found cluster MC Labels
//        cfConfig                                // configuration
//        );
//  }
//
//
//  /*
//   * wait for threads to join
//   */
//  for (std::thread& t: thread_vector) {
//    t.join();
//  }
//
//  /*
//   * collect clusters from individual cluster finder
//   */
//
//  // map to count unique MC labels
//  std::map<MCCompLabel,int> labelCount;
//
//  // multiset to sort labels according to occurrence
//  auto mcComp = [](const std::pair<MCCompLabel, int>& a, const std::pair<MCCompLabel, int>& b) { return a.second > b.second;};
//  std::multiset<std::pair<MCCompLabel,int>,decltype(mcComp)> labelSort(mcComp);
//
//  // for each CRU
//  for (unsigned cru = 0; cru < mClusterStorage.size(); ++cru) {
//    std::vector<Cluster>* clustersFromCRU = &mClusterStorage[cru];
//    std::vector<std::vector<std::pair<int,int>>>* labelsFromCRU = &mClusterDigitIndexStorage[cru];
//
//    // for each found cluster
//    for(unsigned c = 0; c < clustersFromCRU->size(); ++c) {
//      const auto clusterPos = mClusterArray->size();
//      mClusterArray->emplace_back(clustersFromCRU->at(c));
//      if (mClusterMcLabelArray == nullptr) continue;
//      labelCount.clear();
//      labelSort.clear();
//
//      // for each used digit
//      for (auto &digitIndex : labelsFromCRU->at(c)) {
//        if (digitIndex.first < 0) continue;
//        for (auto &l : mLastMcDigitTruth[digitIndex.second]->getLabels(digitIndex.first)) {
//          labelCount[l]++;
//        }
//      }
//      for (auto &l : labelCount) labelSort.insert(l);
//      for (auto &l : labelSort) mClusterMcLabelArray->addElement(clusterPos,l.first);
//    }
//  }
//
//  mLastTimebin = iTimeBinMax;
//}

bool HwClusterer::hwClusterFinder(unsigned short center_pad, unsigned center_time, unsigned short row,
    std::shared_ptr<ClusterHardware> cluster, std::shared_ptr<std::vector<std::pair<MCCompLabel,unsigned>>> sortedMcLabels) {

//  unsigned qMax = mDataBuffer[row][(center_time%5)*mPadsPerRow[row]+center_pad];
  unsigned qMaxIndex = (center_time%5)*mPadsPerRow[row]+center_pad;

  // check if center peak is above peak charge threshold
  if (mDataBuffer[row][qMaxIndex] <= (mPeakChargeThreshold<<4)) return false;

  // Require at least one neighboring time bin with signal
  if (mRequireNeighbouringTimebin &&
      mDataBuffer[row][(center_time+1)%5*mPadsPerRow[row]+center_pad] <= 0 &&
      mDataBuffer[row][(center_time+4)%5*mPadsPerRow[row]+center_pad] <= 0) return false;
      //                         (x+4)%5 = (x-1)%5

  // Require at least one neighboring pad with signal
  if (mRequireNeighbouringPad &&
      mDataBuffer[row][center_time%5*mPadsPerRow[row]+center_pad+1] <= 0 &&
      mDataBuffer[row][center_time%5*mPadsPerRow[row]+center_pad-1] <= 0) return false;

  // check for local maximum
  if (!(mDataBuffer[row][qMaxIndex] >= mDataBuffer[row][(center_time+4)%5*mPadsPerRow[row]+center_pad]))   return false;
  if (!(mDataBuffer[row][qMaxIndex] >  mDataBuffer[row][(center_time+1)%5*mPadsPerRow[row]+center_pad]))   return false;
  if (!(mDataBuffer[row][qMaxIndex] >= mDataBuffer[row][center_time    %5*mPadsPerRow[row]+center_pad-1])) return false;
  if (!(mDataBuffer[row][qMaxIndex] >  mDataBuffer[row][center_time    %5*mPadsPerRow[row]+center_pad+1])) return false;
  if (!(mDataBuffer[row][qMaxIndex] >= mDataBuffer[row][(center_time+4)%5*mPadsPerRow[row]+center_pad-1])) return false;
  if (!(mDataBuffer[row][qMaxIndex] >  mDataBuffer[row][(center_time+1)%5*mPadsPerRow[row]+center_pad+1])) return false;
  if (!(mDataBuffer[row][qMaxIndex] >  mDataBuffer[row][(center_time+1)%5*mPadsPerRow[row]+center_pad-1])) return false;
  if (!(mDataBuffer[row][qMaxIndex] >= mDataBuffer[row][(center_time+4)%5*mPadsPerRow[row]+center_pad+1])) return false;

//  if (row == 14 && center_time%447 == 177/* && (qMax>>4) == 36*/ && center_time/447 == 3) {
//    std::cout << std::endl << std::endl << center_pad-2 << ", " << center_time%447 << ", " << static_cast<int>((qMax>>3)/2.0+0.5) << std::endl;
//    for (int dt = -2; dt <= 2; ++dt) {
//      for (int dp = -2; dp <= 2; ++dp) {
//        std::cout << (*data)[(center_time+dt)*nPads+center_pad+dp] << "\t";
//      }
//      std::cout << std::endl;
//    }
//    std::cout << std::endl << std::endl;
//  }

  unsigned qTot = 0;
  int pad = 0;
  int time = 0;
  int sigmaPad2 = 0;
  int sigmaTime2 = 0;
  int flags = 0;

  // Cluster:
  //
  // o  o   o   o   o
  // o  i   i   i   o
  // o  i   C   i   o
  // o  i   i   i   o
  // o  o   o   o   o

  //
  //    i   i   i
  //    i   C   i
  //    i   i   i
  //
  for (short dt = -1; dt <= 1; ++dt) {
    for (short dp = -1; dp <= 1; ++dp) {
      updateCluster(row, center_pad, center_time, dp, dt, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    }
  }

  //        o
  //        i
  // o  i   C   i   o
  //        i
  //        o
  if (mDataBuffer[row][center_time%5*mPadsPerRow[row]+center_pad-1] > (mContributionChargeThreshold<<4)) {
    updateCluster(row, center_pad, center_time, -2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][center_time%5*mPadsPerRow[row]+center_pad+1] > (mContributionChargeThreshold<<4)) {
    updateCluster(row, center_pad, center_time, +2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5+((center_time-1)%5))%5*mPadsPerRow[row]+center_pad] > (mContributionChargeThreshold<<4)) {
    updateCluster(row, center_pad, center_time, 0, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5+((center_time+1)%5))%5*mPadsPerRow[row]+center_pad] > (mContributionChargeThreshold<<4)) {
    updateCluster(row, center_pad, center_time, 0, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }

  // o  o       o   o
  // o  i       i   o
  //        C
  // o  i       i   o
  // o  o       o   o
  if (mDataBuffer[row][(5+((center_time-1)%5))%5*mPadsPerRow[row]+center_pad-1] > (mContributionChargeThreshold<<4)) {
    updateCluster(row, center_pad, center_time, -2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, -2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, -1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5+((center_time+1)%5))%5*mPadsPerRow[row]+center_pad-1] > (mContributionChargeThreshold<<4)) {
    updateCluster(row, center_pad, center_time, -2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, -2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, -1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5+((center_time-1)%5))%5*mPadsPerRow[row]+center_pad+1] > (mContributionChargeThreshold<<4)) {
    updateCluster(row, center_pad, center_time, +2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, +2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, +1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5+((center_time+1)%5))%5*mPadsPerRow[row]+center_pad+1] > (mContributionChargeThreshold<<4)) {
    updateCluster(row, center_pad, center_time, +2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, +2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, +1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }

  cluster->setCluster(
      center_pad-2,     // we have to artificial empty pads "on the left" which needs to be subtracted
      center_time%447,  // the time within a HB
      pad,time,
      sigmaPad2,sigmaTime2,
      (mDataBuffer[row][qMaxIndex]>>3),        // keep only 1 fixed point precision bit
      qTot,
      mGlobalRowToLocalRow[row], // the hardware knows only about the local row
      flags);

  using vecType = std::pair<MCCompLabel,unsigned>;
  std::sort(sortedMcLabels->begin(), sortedMcLabels->end(), [](const vecType& a, const vecType& b) { return a.second > b.second; });

  return true;
}

void HwClusterer::writeOutputForTimeOffset(unsigned timeOffset) {
  unsigned clusterCounter = 0;
  // Check in which regions cluster were found
  for (unsigned short region = 0; region < 10; ++region) {
    if (mTmpClusterArray[region]->size() == 0) continue;

    // Create new container
    mClusterArray->emplace_back();
    auto clusterContainer = mClusterArray->back().getContainer();

    // Set meta data
    clusterContainer->CRU = mClusterSector*10 + region;
    clusterContainer->numberOfClusters = 0;
    clusterContainer->timeBinOffset = timeOffset;

    for (auto &c : *mTmpClusterArray[region]) {
      // if the container is full, create a new one
      if (clusterContainer->numberOfClusters == mClusterArray->back().getMaxNumberOfClusters()) {
        mClusterArray->emplace_back();
        clusterContainer = mClusterArray->back().getContainer();
        clusterContainer->CRU = mClusterSector*10 + region;
        clusterContainer->numberOfClusters = 0;
        clusterContainer->timeBinOffset = timeOffset;
      }
      // Copy cluster and increment cluster counter
      clusterContainer->clusters[clusterContainer->numberOfClusters++] = *(c.first);
      for (auto &mcLabel : *(c.second)) {
        mClusterMcLabelArray->addElement(clusterCounter,mcLabel.first);
      }
      ++clusterCounter;
    }
    // Clear copied temporary storage
    mTmpClusterArray[region]->clear();
  }
}

void HwClusterer::findClusterForTime(unsigned timebin) {
  auto cluster = std::make_shared<ClusterHardware>();
  auto sortedMcLabels = std::make_shared<std::vector<std::pair<MCCompLabel,unsigned>>>();

  for (unsigned short row = mNumRows; row--;) {
    // two empty pads on the left and right without a cluster peak
    for (unsigned short pad = 2; pad < mPadsPerRow[row]-2; ++pad) {
      if (hwClusterFinder(pad,timebin,row,cluster,sortedMcLabels)) {
        mTmpClusterArray[mGlobalRowToRegion[row]]->emplace_back(cluster,sortedMcLabels);

        // create new empty cluster
        cluster = std::make_shared<ClusterHardware>();
        sortedMcLabels = std::make_shared<std::vector<std::pair<MCCompLabel,unsigned>>>();
      }
    }
  }
}

void HwClusterer::finishFrame(bool clear) {
  int HB;
  // Search in last remaining timebins for clusters
  for (int i = mLastTimebin+1; i-mLastTimebin < 3; ++i) {
    HB = (i-2)/447; // integer division on purpose
    if (HB != mLastHB) writeOutputForTimeOffset(i-2);

    findClusterForTime(i+3);
    clearBuffer(i);
    mLastHB = HB;
  }
  writeOutputForTimeOffset(mLastTimebin+3);

  if (clear) {
    for (auto i : {0,1,2,3,4}) clearBuffer(i);
  }
}

void HwClusterer::clearBuffer(unsigned timebin) {
  for (unsigned short row = 0; row < mNumRows; ++row) {
    // reset timebin which is not needed anymore
    // TODO: fill with pedestal/noise instead of 0
    std::fill(mDataBuffer[row].begin()+(timebin%5)*mPadsPerRow[row],
              mDataBuffer[row].begin()+(timebin%5)*mPadsPerRow[row]+mPadsPerRow[row]-1,0);
    std::fill(mIndexBuffer[row].begin()+(timebin%5)*mPadsPerRow[row],
              mIndexBuffer[row].begin()+(timebin%5)*mPadsPerRow[row]+mPadsPerRow[row]-1,-1);
  }

}

void HwClusterer::updateCluster(
    int row, unsigned short center_pad, unsigned center_time, short dp, short dt,
    unsigned& qTot, int& pad, int& time, int& sigmaPad2, int&sigmaTime2,
    std::shared_ptr<std::vector<std::pair<MCCompLabel,unsigned>>> mcLabels)
{

  // to avoid negative numbers after modulo:
  // (b + (a % b)) % b in [0,b-1] even if a < 0
  int index = (5+((center_time+dt)%5))%5*mPadsPerRow[row]+center_pad+dp;
  unsigned charge = mDataBuffer[row][index];

  qTot          += charge;
  pad           += charge * dp;
  time          += charge * dt;
  sigmaPad2     += charge * dp * dp;
  sigmaTime2    += charge * dt * dt;

  if (mMCtruth[(center_time+dt+5)%5] != nullptr) {
    for (auto &label : mMCtruth[(center_time+dt+5)%5]->getLabels(mIndexBuffer[row][index])) {
      bool isKnown = false;
      for (auto &vecLabel : *mcLabels) {
        if (label == vecLabel.first) {
          ++vecLabel.second;
          isKnown = true;
        }
      }
      if (!isKnown) {
        mcLabels->emplace_back(label,1);
      }
    }
  }
}



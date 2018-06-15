// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HwClusterer.cxx
/// \brief Hwclusterer for the TPC

#include "TPCReconstruction/HwClusterer.h"
#include "TPCBase/Digit.h"
#include "TPCBase/CRU.h"
#include "TPCBase/Mapper.h"
#include "DataFormatsTPC/Cluster.h"
#include "DataFormatsTPC/ClusterHardware.h"

#include "FairLogger.h"

#include <limits>

using namespace o2::TPC;

//______________________________________________________________________________
HwClusterer::HwClusterer(
  std::shared_ptr<std::vector<ClusterHardwareContainer8kb>> clusterOutputContainer,
  std::shared_ptr<std::vector<Cluster>> clusterOutputSimple,
  std::shared_ptr<MCLabelContainer> labelOutput, int sectorid)
  : Clusterer(),
    mClusterSector(sectorid),
    mNumRows(0),
    mLastTimebin(-1),
    mLastHB(0),
    mPeakChargeThreshold(2),
    mContributionChargeThreshold(0),
    mClusterCounter(0),
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
    mClusterMcLabelArray(labelOutput),
    mClusterArray(clusterOutputContainer),
    mPlainClusterArray(clusterOutputSimple)
{
  LOG(DEBUG) << "Enter Initializer of HwClusterer" << FairLogger::endl;
  /*
   * Prepare temporary storage for digits
   */
  Mapper& mapper = Mapper::instance();

  mNumRows = mapper.getNumberOfRows();
  mDataBuffer.resize(mNumRows);
  mIndexBuffer.resize(mNumRows);
  mPadsPerRow.resize(mNumRows);

  for (unsigned short row = 0; row < mNumRows; ++row) {
    // add two empty pads on the left and on the right
    mPadsPerRow[row] = mapper.getNumberOfPadsInRowSector(row) + 2 + 2;

    // prepare for 5 timebins
    mDataBuffer[row].resize(mPadsPerRow[row] * 5, 0);
    mIndexBuffer[row].resize(mPadsPerRow[row] * 5, -1);
  }

  mTmpClusterArray.resize(10);
  for (unsigned short region = 0; region < 10; ++region) {
    mTmpClusterArray[region] = std::make_unique<std::vector<std::pair<std::unique_ptr<ClusterHardware>, std::unique_ptr<std::vector<std::pair<MCCompLabel, unsigned>>>>>>();
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
  mMCtruth.resize(5);
}

//______________________________________________________________________________
HwClusterer::HwClusterer(
  std::shared_ptr<std::vector<Cluster>> clusterOutput,
  std::shared_ptr<MCLabelContainer> labelOutput, int sectorid)
  : HwClusterer(nullptr, clusterOutput, labelOutput, sectorid)
{
}

//______________________________________________________________________________
HwClusterer::HwClusterer(
  std::shared_ptr<std::vector<ClusterHardwareContainer8kb>> clusterOutput,
  std::shared_ptr<MCLabelContainer> labelOutput, int sectorid)
  : HwClusterer(clusterOutput, nullptr, labelOutput, sectorid)
{
}

//______________________________________________________________________________
void HwClusterer::Process(std::vector<o2::TPC::Digit> const& digits, MCLabelContainer const* mcDigitTruth, int eventCount)
{
  if (mClusterArray)
    mClusterArray->clear();
  if (mPlainClusterArray)
    mPlainClusterArray->clear();

  if (mClusterMcLabelArray)
    mClusterMcLabelArray->clear();
  mClusterCounter = 0;

  int digitIndex = 0;
  int index;
  int HB;

  /*
   * Loop over all (time ordered) digits
   */
  for (const auto& digit : digits) {
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
      // we have to copy the MC truth container because we need the information
      // maybe only in the next events (we store permanently 5 timebins), where
      // the original pointer could already point to the next container.
      if (mcDigitTruth)
        mMCtruth[digit.getTimeStamp() % 5] = std::make_unique<MCLabelContainer const>(*mcDigitTruth);
      else
        mMCtruth[digit.getTimeStamp() % 5] = std::unique_ptr<MCLabelContainer const>(nullptr);

      /*
       * If the timebin changes, it could change by more then just 1 (not every
       * timebin has digits). Since the tmp storage covers 5 timebins, at most
       * 5 new timebins need to be prepared and checked for clusters.
       */
      for (int i = mLastTimebin + 1; (i <= digit.getTimeStamp()) && (i - (mLastTimebin + 1) < 5); ++i) {

        /*
         * If the HB of the cluster which will be found in a few lines, NOT the
         * current timebin is a new one, we have to fill the  output container
         * with the so far found clusters. Because cluster center and timebin
         * have an offset of two with respect to each other (see next comment),
         * the HB is calculated with (i-2). By the way, it is not possible, that
         * a cluster is found with a negative HB, because at least 2 timebins
         * have to be filled to be able to find a cluster.
         */
        HB = (i - 2) / 447; // integer division on purpose
        if (HB != mLastHB) {
          writeOutputForTimeOffset(mLastHB * 447);
        }

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
        findClusterForTime(i + 3);

        clearBuffer(i);

        mLastHB = HB;
      }
    }

    /*
     * add current digit to storage
     */
    index = (digit.getTimeStamp() % 5) * mPadsPerRow[digit.getRow()] + (digit.getPad() + 2);
    // offset of digit pad because of 2 empty pads on both sides

    if (mPedestalObject) {
      /*
       * If a pedestal object was registered, check if charge of pad is greater
       * than pedestal value. If so, assign difference of charge and pedestal
       * to buffer, if not, set buffer to 0.
       */
      if (digit.getChargeFloat() < mPedestalObject->getValue(CRU(digit.getCRU()), digit.getRow(), digit.getPad()))
        mDataBuffer[digit.getRow()][index] = 0;
      else
        mDataBuffer[digit.getRow()][index] += static_cast<unsigned>(
          (digit.getChargeFloat() - mPedestalObject->getValue(CRU(digit.getCRU()), digit.getRow(), digit.getPad())) * (1 << 4));

    } else {
      mDataBuffer[digit.getRow()][index] += static_cast<unsigned>(digit.getChargeFloat() * (1 << 4));
    }
    mIndexBuffer[digit.getRow()][index] = digitIndex++;

    mLastTimebin = digit.getTimeStamp();
  }

  if (!mIsContinuousReadout)
    finishFrame(true);

  if (digits.size() != 0)
    LOG(DEBUG) << "Event ranged from time bin " << digits.front().getTimeStamp() << " to " << digits.back().getTimeStamp() << "." << FairLogger::endl;
}

//______________________________________________________________________________
void HwClusterer::FinishProcess(std::vector<o2::TPC::Digit> const& digits, MCLabelContainer const* mcDigitTruth, int eventCount)
{
  // Process the last digits (if there are any)
  Process(digits, mcDigitTruth, eventCount);

  // Search in last remaining timebins
  finishFrame(false);
}

//______________________________________________________________________________
bool HwClusterer::hwClusterFinder(unsigned short center_pad, unsigned center_time, unsigned short row,
                                  ClusterHardware& cluster, std::vector<std::pair<MCCompLabel, unsigned>>& sortedMcLabels)
{

  unsigned qMaxIndex = (center_time % 5) * mPadsPerRow[row] + center_pad;

  // check if center peak is above peak charge threshold
  if (mDataBuffer[row][qMaxIndex] <= (mPeakChargeThreshold << 4))
    return false;

  // Require at least one neighboring time bin with signal
  if (mRequireNeighbouringTimebin &&
      mDataBuffer[row][(center_time + 1) % 5 * mPadsPerRow[row] + center_pad] <= 0 &&
      mDataBuffer[row][(center_time + 4) % 5 * mPadsPerRow[row] + center_pad] <= 0)
    return false;
  //                         (x+4)%5 = (x-1)%5

  // Require at least one neighboring pad with signal
  if (mRequireNeighbouringPad &&
      mDataBuffer[row][center_time % 5 * mPadsPerRow[row] + center_pad + 1] <= 0 &&
      mDataBuffer[row][center_time % 5 * mPadsPerRow[row] + center_pad - 1] <= 0)
    return false;

  // check for local maximum
  if (!(mDataBuffer[row][qMaxIndex] >= mDataBuffer[row][(center_time + 4) % 5 * mPadsPerRow[row] + center_pad]))
    return false;
  if (!(mDataBuffer[row][qMaxIndex] > mDataBuffer[row][(center_time + 1) % 5 * mPadsPerRow[row] + center_pad]))
    return false;
  if (!(mDataBuffer[row][qMaxIndex] >= mDataBuffer[row][center_time % 5 * mPadsPerRow[row] + center_pad - 1]))
    return false;
  if (!(mDataBuffer[row][qMaxIndex] > mDataBuffer[row][center_time % 5 * mPadsPerRow[row] + center_pad + 1]))
    return false;
  if (!(mDataBuffer[row][qMaxIndex] >= mDataBuffer[row][(center_time + 4) % 5 * mPadsPerRow[row] + center_pad - 1]))
    return false;
  if (!(mDataBuffer[row][qMaxIndex] > mDataBuffer[row][(center_time + 1) % 5 * mPadsPerRow[row] + center_pad + 1]))
    return false;
  if (!(mDataBuffer[row][qMaxIndex] > mDataBuffer[row][(center_time + 1) % 5 * mPadsPerRow[row] + center_pad - 1]))
    return false;
  if (!(mDataBuffer[row][qMaxIndex] >= mDataBuffer[row][(center_time + 4) % 5 * mPadsPerRow[row] + center_pad + 1]))
    return false;

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
  if (mDataBuffer[row][center_time % 5 * mPadsPerRow[row] + center_pad - 1] > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, -2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][center_time % 5 * mPadsPerRow[row] + center_pad + 1] > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, +2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5 + ((center_time - 1) % 5)) % 5 * mPadsPerRow[row] + center_pad] > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, 0, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5 + ((center_time + 1) % 5)) % 5 * mPadsPerRow[row] + center_pad] > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, 0, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }

  // o  o       o   o
  // o  i       i   o
  //        C
  // o  i       i   o
  // o  o       o   o
  if (mDataBuffer[row][(5 + ((center_time - 1) % 5)) % 5 * mPadsPerRow[row] + center_pad - 1] > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, -2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, -2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, -1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5 + ((center_time + 1) % 5)) % 5 * mPadsPerRow[row] + center_pad - 1] > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, -2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, -2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, -1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5 + ((center_time - 1) % 5)) % 5 * mPadsPerRow[row] + center_pad + 1] > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, +2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, +2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, +1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }
  if (mDataBuffer[row][(5 + ((center_time + 1) % 5)) % 5 * mPadsPerRow[row] + center_pad + 1] > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, +2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, +2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
    updateCluster(row, center_pad, center_time, +1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, sortedMcLabels);
  }

  cluster.setCluster(
    center_pad - 2,    // we have two artificial empty pads "on the left" which needs to be subtracted
    center_time % 447, // the time within a HB
    pad, time,
    sigmaPad2, sigmaTime2,
    (mDataBuffer[row][qMaxIndex] >> 3), // keep only 1 fixed point precision bit
    qTot,
    mGlobalRowToLocalRow[row], // the hardware knows only about the local row
    flags);

  using vecType = std::pair<MCCompLabel, unsigned>;
  std::sort(sortedMcLabels.begin(), sortedMcLabels.end(), [](const vecType& a, const vecType& b) { return a.second > b.second; });

  return true;
}

//______________________________________________________________________________
void HwClusterer::writeOutputForTimeOffset(unsigned timeOffset)
{
  // Check in which regions cluster were found
  for (unsigned short region = 0; region < 10; ++region) {
    if (mTmpClusterArray[region]->size() == 0)
      continue;

    if (mClusterArray) {
      // Create new container
      mClusterArray->emplace_back();
      auto clusterContainer = mClusterArray->back().getContainer();

      // Set meta data
      clusterContainer->CRU = mClusterSector * 10 + region;
      clusterContainer->numberOfClusters = 0;
      clusterContainer->timeBinOffset = timeOffset;

      for (auto& c : *mTmpClusterArray[region]) {
        // if the container is full, create a new one
        if (clusterContainer->numberOfClusters == mClusterArray->back().getMaxNumberOfClusters()) {
          mClusterArray->emplace_back();
          clusterContainer = mClusterArray->back().getContainer();
          clusterContainer->CRU = mClusterSector * 10 + region;
          clusterContainer->numberOfClusters = 0;
          clusterContainer->timeBinOffset = timeOffset;
        }
        // Copy cluster and increment cluster counter
        clusterContainer->clusters[clusterContainer->numberOfClusters++] = *(c.first);
        for (auto& mcLabel : *(c.second)) {
          mClusterMcLabelArray->addElement(mClusterCounter, mcLabel.first);
        }
        ++mClusterCounter;
      }
    } else if (mPlainClusterArray) {
      short cru = mClusterSector * 10 + region;
      for (auto& c : *mTmpClusterArray[region]) {
        auto& cluster = *(c.first);
        mPlainClusterArray->emplace_back(
          cru,
          cluster.getRow(),
          cluster.getQTotFloat(),
          cluster.getQMax(),
          cluster.getPad(),
          std::sqrt(cluster.getSigmaPad2()),
          cluster.getTimeLocal() + timeOffset,
          std::sqrt(cluster.getSigmaTime2()));

        for (auto& mcLabel : *(c.second)) {
          mClusterMcLabelArray->addElement(mClusterCounter, mcLabel.first);
        }
        ++mClusterCounter;
      }
    }

    // Clear copied temporary storage
    mTmpClusterArray[region]->clear();
  }
}

//______________________________________________________________________________
void HwClusterer::findClusterForTime(unsigned timebin)
{
  auto cluster = std::make_unique<ClusterHardware>();
  auto sortedMcLabels = std::make_unique<std::vector<std::pair<MCCompLabel, unsigned>>>();

  for (unsigned short row = mNumRows; row--;) {
    // two empty pads on the left and right without a cluster peak
    for (unsigned short pad = 2; pad < mPadsPerRow[row] - 2; ++pad) {
      if (hwClusterFinder(pad, timebin - 5, row, *cluster.get(), *sortedMcLabels.get())) {
        mTmpClusterArray[mGlobalRowToRegion[row]]->emplace_back(std::move(cluster), std::move(sortedMcLabels));

        // create new empty cluster
        cluster = std::make_unique<ClusterHardware>();
        sortedMcLabels = std::make_unique<std::vector<std::pair<MCCompLabel, unsigned>>>();
      }
    }
  }
}

//______________________________________________________________________________
void HwClusterer::finishFrame(bool clear)
{
  int HB;
  // Search in last remaining timebins for clusters
  for (int i = mLastTimebin + 1; i - mLastTimebin < 3; ++i) {
    HB = (i - 2) / 447; // integer division on purpose
    if (HB != mLastHB) {
      writeOutputForTimeOffset(mLastHB * 447);
    }

    findClusterForTime(i + 3);
    clearBuffer(i);
    mLastHB = HB;
  }
  writeOutputForTimeOffset(mLastHB * 447);

  if (clear) {
    for (auto i : { 0, 1, 2, 3, 4 })
      clearBuffer(i);
  }
}

//______________________________________________________________________________
void HwClusterer::clearBuffer(unsigned timebin)
{
  for (unsigned short row = 0; row < mNumRows; ++row) {
    // reset timebin which is not needed anymore
    // TODO: for simulation fill with pedestal/noise instead of 0
    std::fill(mDataBuffer[row].begin() + (timebin % 5) * mPadsPerRow[row],
              mDataBuffer[row].begin() + (timebin % 5) * mPadsPerRow[row] + mPadsPerRow[row] - 1, 0);
    std::fill(mIndexBuffer[row].begin() + (timebin % 5) * mPadsPerRow[row],
              mIndexBuffer[row].begin() + (timebin % 5) * mPadsPerRow[row] + mPadsPerRow[row] - 1, -1);
  }
}

//______________________________________________________________________________
void HwClusterer::updateCluster(
  int row, short center_pad, int center_time, short dp, short dt,
  unsigned& qTot, int& pad, int& time, int& sigmaPad2, int& sigmaTime2,
  std::vector<std::pair<MCCompLabel, unsigned>>& mcLabels)
{

  // to avoid negative numbers after modulo:
  // (b + (a % b)) % b in [0,b-1] even if a < 0
  int index = (5 + ((center_time + dt) % 5)) % 5 * mPadsPerRow[row] + center_pad + dp;
  unsigned charge = mDataBuffer[row][index];

  qTot += charge;
  pad += charge * dp;
  time += charge * dt;
  sigmaPad2 += charge * dp * dp;
  sigmaTime2 += charge * dt * dt;

  if (mMCtruth[(center_time + dt + 5) % 5] != nullptr) {
    for (auto& label : mMCtruth[(center_time + dt + 5) % 5]->getLabels(mIndexBuffer[row][index])) {
      bool isKnown = false;
      for (auto& vecLabel : mcLabels) {
        if (label == vecLabel.first) {
          ++vecLabel.second;
          isKnown = true;
        }
      }
      if (!isKnown) {
        mcLabels.emplace_back(label, 1);
      }
    }
  }
}

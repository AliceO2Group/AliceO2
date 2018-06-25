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

#include <cassert>
#include <limits>

using namespace o2::TPC;

//______________________________________________________________________________
HwClusterer::HwClusterer(
  std::vector<ClusterHardwareContainer8kb>* clusterOutputContainer,
  std::vector<Cluster>* clusterOutputSimple,
  int sectorid, MCLabelContainer* labelOutput)
  : Clusterer(),
    mNumRows(0),
    mCurrentMcContainerInBuffer(0),
    mClusterSector(sectorid),
    mLastTimebin(-1),
    mLastHB(0),
    mPeakChargeThreshold(2),
    mContributionChargeThreshold(0),
    mClusterCounter(0),
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

  // Given sector ID must be within 0 and 35 for a proper CRU ID calculation
  assert(sectorid >= 0 && sectorid < 36);

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
    mDataBuffer[row].resize(mPadsPerRow[row] * mTimebinsInBuffer, 0);
    mIndexBuffer[row].resize(mPadsPerRow[row] * mTimebinsInBuffer, -1);
  }

  mTmpClusterArray.resize(10);
  for (unsigned short region = 0; region < 10; ++region) {
    mTmpClusterArray[region] = std::make_unique<std::vector<std::pair<ClusterHardware, std::vector<std::pair<MCCompLabel, unsigned>>>>>();
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
  mMCtruth.resize(mTimebinsInBuffer, nullptr);
}

//______________________________________________________________________________
HwClusterer::HwClusterer(
  std::vector<Cluster>* clusterOutput,
  int sectorid, MCLabelContainer* labelOutput)
  : HwClusterer(nullptr, clusterOutput, sectorid, labelOutput)
{
}

//______________________________________________________________________________
HwClusterer::HwClusterer(
  std::vector<ClusterHardwareContainer8kb>* clusterOutput,
  int sectorid, MCLabelContainer* labelOutput)
  : HwClusterer(clusterOutput, nullptr, sectorid, labelOutput)
{
}

//______________________________________________________________________________
void HwClusterer::process(std::vector<o2::TPC::Digit> const& digits, MCLabelContainer const* mcDigitTruth)
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
  unsigned HB;
  mCurrentMcContainerInBuffer = 0;

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
      /*
       * If the timebin changes, it could change by more then just 1 (not every
       * timebin has digits). Since the tmp storage covers mTimebinsInBuffer,
       * at most mTimebinsInBuffer new timebins need to be prepared and checked
       * for clusters.
       */
      for (int i = mLastTimebin; (i < digit.getTimeStamp()) && (i - mLastTimebin < mTimebinsInBuffer); ++i) {

        /*
         * If the HB of the cluster which will be found in a few lines, NOT the
         * current timebin is a new one, we have to fill the output container
         * with the so far found clusters. Because cluster center and timebin
         * have an offset of two with respect to each other (see next comment),
         * the HB is calculated with (i-2). By the way, it is not possible, that
         * a cluster is found with a negative HB, because at least 2 timebins
         * have to be filled to be able to find a cluster.
         */
        HB = i < 2 ? 0 : (i - 2) / 447; // integer division on purpose
        if (HB != mLastHB) {
          writeOutputWithTimeOffset(mLastHB * 447);
        }

        /*
         * For each row, we first check for cluster peaks in the timebin i-2.
         *
         * If mTimebinsInBuffer would be 5 and i 5 is the new digit timebin (4
         * would be mLastTimebin), then 0 is the oldest one timebin and will to
         * be replaced by the new arriving one. The cluster which could be
         * found, would then range from timebin 0 to 4 and has its center at
         * timebin 2. Threrefore we are looking in (i - 2) for clusters and
         * clearing (i - 4), or (i + 1) afterwards.
         *       ---------
         * -> 0 |
         *    1 |
         *    2 | XXXXXX
         *    3 |
         *    4 |
         *       ---------
         */
        findPeaksForTime(i);
        computeClusterForTime(i - 2);

        clearBuffer(i + 1);

        mLastHB = HB;
      }

      // we have to copy the MC truth container because we need the information
      // maybe only in the next events (we store permanently 5 timebins), where
      // the original pointer could already point to the next container.
      if (mcDigitTruth) {
        if (mCurrentMcContainerInBuffer == 0)
          mMCtruth[mapTimeInRange(digit.getTimeStamp())] = std::make_shared<MCLabelContainer const>(*mcDigitTruth);
        else
          mMCtruth[mapTimeInRange(digit.getTimeStamp())] = std::shared_ptr<MCLabelContainer const>(mMCtruth[getFirstSetBitOfField()]);

        mCurrentMcContainerInBuffer |= (0x1 << (mapTimeInRange(digit.getTimeStamp())));
      }
    }

    /*
     * add current digit to storage
     */
    index = mapTimeInRange(digit.getTimeStamp()) * mPadsPerRow[digit.getRow()] + (digit.getPad() + 2);
    // offset of digit pad because of 2 empty pads on both sides

    if (mPedestalObject) {
      /*
       * If a pedestal object was registered, check if charge of pad is greater
       * than pedestal value. If so, assign difference of charge and pedestal
       * to buffer, if not, set buffer to 0.
       */
      if (digit.getChargeFloat() < mPedestalObject->getValue(CRU(digit.getCRU()), digit.getRow(), digit.getPad())) {
        mDataBuffer[digit.getRow()][index] = 0;
      } else {
        mDataBuffer[digit.getRow()][index] += static_cast<unsigned>(
          (digit.getChargeFloat() - mPedestalObject->getValue(CRU(digit.getCRU()), digit.getRow(), digit.getPad())) * (1 << 4));
      }
    } else {
      mDataBuffer[digit.getRow()][index] += static_cast<unsigned>(digit.getChargeFloat() * (1 << 4));
    }
    if (mDataBuffer[digit.getRow()][index] > 0x3FFF)
      mDataBuffer[digit.getRow()][index] = 0x3FFF; // set only 14 LSBs

    mIndexBuffer[digit.getRow()][index] = digitIndex++;

    mLastTimebin = digit.getTimeStamp();
  }

  if (!mIsContinuousReadout)
    finishFrame(true);

  if (digits.size() != 0)
    LOG(DEBUG) << "Event ranged from time bin " << digits.front().getTimeStamp() << " to " << digits.back().getTimeStamp() << "." << FairLogger::endl;
}

//______________________________________________________________________________
void HwClusterer::finishProcess(std::vector<o2::TPC::Digit> const& digits, MCLabelContainer const* mcDigitTruth)
{
  // Process the last digits (if there are any)
  process(digits, mcDigitTruth);

  // Search in last remaining timebins
  finishFrame(false);
}

//______________________________________________________________________________
void HwClusterer::hwClusterProcessor(unsigned qMaxIndex, short center_pad, int center_time, unsigned short row,
                                     ClusterHardware& cluster, std::vector<std::pair<MCCompLabel, unsigned>>& mcLabels)
{
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

  // Use always inner 3x3 matrix
  //    i   i   i
  //    i   C   i
  //    i   i   i
  //
  for (short dt = -1; dt <= 1; ++dt) {
    for (short dp = -1; dp <= 1; ++dp) {
      updateCluster(row, center_pad, center_time, dp, dt, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    }
  }

  // Use outer pads (o) only, if corresponding inner pad (i) is above contribution threshold
  //        o
  //        i
  // o  i   C   i   o
  //        i
  //        o
  if (getFpOfADC(mDataBuffer[row][mapTimeInRange(center_time) * mPadsPerRow[row] + center_pad - 1]) > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, -2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
  }
  if (getFpOfADC(mDataBuffer[row][mapTimeInRange(center_time) * mPadsPerRow[row] + center_pad + 1]) > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, +2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
  }
  if (getFpOfADC(mDataBuffer[row][mapTimeInRange(center_time - 1) * mPadsPerRow[row] + center_pad]) > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, 0, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
  }
  if (getFpOfADC(mDataBuffer[row][mapTimeInRange(center_time + 1) * mPadsPerRow[row] + center_pad]) > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, 0, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
  }

  // o  o       o   o
  // o  i       i   o
  //        C
  // o  i       i   o
  // o  o       o   o
  if (getFpOfADC(mDataBuffer[row][mapTimeInRange(center_time - 1) * mPadsPerRow[row] + center_pad - 1]) > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, -2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(row, center_pad, center_time, -2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(row, center_pad, center_time, -1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
  }
  if (getFpOfADC(mDataBuffer[row][mapTimeInRange(center_time + 1) * mPadsPerRow[row] + center_pad - 1]) > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, -2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(row, center_pad, center_time, -2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(row, center_pad, center_time, -1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
  }
  if (getFpOfADC(mDataBuffer[row][mapTimeInRange(center_time - 1) * mPadsPerRow[row] + center_pad + 1]) > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, +2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(row, center_pad, center_time, +2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(row, center_pad, center_time, +1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
  }
  if (getFpOfADC(mDataBuffer[row][mapTimeInRange(center_time + 1) * mPadsPerRow[row] + center_pad + 1]) > (mContributionChargeThreshold << 4)) {
    updateCluster(row, center_pad, center_time, +2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(row, center_pad, center_time, +2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(row, center_pad, center_time, +1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
  }

  cluster.setCluster(
    center_pad - 2,    // we have two artificial empty pads "on the left" which needs to be subtracted
    center_time % 447, // the time within a HB
    pad, time,
    sigmaPad2, sigmaTime2,
    (getFpOfADC(mDataBuffer[row][qMaxIndex]) >> 3), // keep only 1 fixed point precision bit
    qTot,
    mGlobalRowToLocalRow[row], // the hardware knows only about the local row
    flags);

  using vecType = std::pair<MCCompLabel, unsigned>;
  std::sort(mcLabels.begin(), mcLabels.end(), [](const vecType& a, const vecType& b) { return a.second > b.second; });
}
//______________________________________________________________________________
void HwClusterer::hwPeakFinder(unsigned qMaxIndex, short center_pad, int center_time, unsigned short row)
{

  // Always the center pad is compared with a fixed other pad. The first
  // comparison is in pad direction. That bin (p+2,t+1) is a peak in pad
  // direction, it has to be >= then (p+1,t+1) and in the next iteration, where
  // the next pad is the center, the comparison of (the new center with the new
  // left one) must be false, so that (p+3,t+1) > (p+2,t+1). Afterwards, this
  // is also done in time direction and both diagonal directions. With those 4
  // comparisons, all 8 neighboring pads are checked iteratively.
  //
  // Example in pad direction:
  //     p+0 p+1 p+2 p+3 p+4                  p+1 p+2 p+3 p+4 p+5
  // t+0  o   o   o   o   o                    o   o   o   o   o
  // t+1  o   i   i   i   o   next iteration   o   i   i   i   o
  // t+2  o   I<->C   i   o  --------------->  O   I<->c   i   o
  // t+3  o   i   i   i   o                    o   i   i   i   o
  // t+4  o   o   o   o   o                    o   o   o   o   o
  //
  //
  // Meaning of the set bit:
  //
  // bit | meaning if set
  //  31 | peak in pad direction
  //  30 | peak in time direction
  //  29 | peak in 1. diagonal direction, top left to bottom right
  //  28 | peak in 2. diagonal direction, top right to bottom left
  //  27 | ADC above peak threshold
  //  26 | minimum in pad direction
  //  25 | minimum in time direction
  //  24 | minimum in 1. diagonal direction
  //  23 | minimum in 1. diagonal direction

  //////////////////////////////////////
  // Comparison in pad direction
  compareForPeak(qMaxIndex, mapTimeInRange(center_time) * mPadsPerRow[row] + center_pad - 1, 31, 26, row);

  //////////////////////////////////////
  // Comparison in time direction
  compareForPeak(qMaxIndex, mapTimeInRange(center_time - 1) * mPadsPerRow[row] + center_pad, 30, 25, row);

  //////////////////////////////////////
  // Comparison in 1. diagonal direction
  compareForPeak(qMaxIndex, mapTimeInRange(center_time - 1) * mPadsPerRow[row] + center_pad - 1, 29, 24, row);

  //////////////////////////////////////
  // Comparison in 2. diagonal direction
  compareForPeak(qMaxIndex, mapTimeInRange(center_time - 1) * mPadsPerRow[row] + center_pad + 1, 28, 23, row);

  //////////////////////////////////////
  // Comparison peak threshold
  if (getFpOfADC(mDataBuffer[row][qMaxIndex]) > (mPeakChargeThreshold << 4))
    mDataBuffer[row][qMaxIndex] |= (0x1 << 27);
}

//______________________________________________________________________________
void HwClusterer::writeOutputWithTimeOffset(int timeOffset)
{
  // Check in which regions cluster were found
  for (unsigned int region = 0; region < 10; ++region) {
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
        clusterContainer->clusters[clusterContainer->numberOfClusters++] = c.first;
        if (mClusterMcLabelArray) {
          for (auto& mcLabel : c.second) {
            mClusterMcLabelArray->addElement(mClusterCounter, mcLabel.first);
          }
        }
        ++mClusterCounter;
      }
    } else if (mPlainClusterArray) {
      const int cru = mClusterSector * 10 + region;
      for (auto& c : *mTmpClusterArray[region]) {
        auto& cluster = c.first;
        mPlainClusterArray->emplace_back(
          cru,
          cluster.getRow(),
          cluster.getQTotFloat(),
          cluster.getQMax(),
          cluster.getPad(),
          std::sqrt(cluster.getSigmaPad2()),
          cluster.getTimeLocal() + timeOffset,
          std::sqrt(cluster.getSigmaTime2()));

        if (mClusterMcLabelArray) {
          for (auto& mcLabel : c.second) {
            mClusterMcLabelArray->addElement(mClusterCounter, mcLabel.first);
          }
        }
        ++mClusterCounter;
      }
    }

    // Clear copied temporary storage
    mTmpClusterArray[region]->clear();
  }
}

//______________________________________________________________________________
void HwClusterer::findPeaksForTime(int timebin)
{
  if (timebin < 0)
    return;

  const unsigned timeBinWrapped = mapTimeInRange(timebin);
  for (unsigned short row = 0; row < mNumRows; ++row) {
    const unsigned padOffset = timeBinWrapped * mPadsPerRow[row];
    // two empty pads on the left and right without a cluster peak, check one
    // beyond rightmost pad for remaining relations
    for (short pad = 2; pad < mPadsPerRow[row] - 1; ++pad) {
      const unsigned qMaxIndex = padOffset + pad;
      hwPeakFinder(qMaxIndex, pad, timebin, row);
    }
  }
}

//______________________________________________________________________________
void HwClusterer::computeClusterForTime(int timebin)
{
  if (timebin < 0)
    return;

  ClusterHardware cluster;
  std::vector<std::pair<MCCompLabel, unsigned>> mcLabels;

  const unsigned timeBinWrapped = mapTimeInRange(timebin);
  for (unsigned short row = 0; row < mNumRows; ++row) {
    const unsigned padOffset = timeBinWrapped * mPadsPerRow[row];
    // two empty pads on the left and right without a cluster peak
    for (short pad = 2; pad < mPadsPerRow[row] - 2; ++pad) {
      const unsigned qMaxIndex = padOffset + pad;
      //      if (mDataBuffer[row][qMaxIndex] & (0x1 << 21)) { // if pad is peak

      if (mDataBuffer[row][qMaxIndex] >> 27 == 0x1F) { // all peak bits are set and ADC above threshold
        hwClusterProcessor(qMaxIndex, pad, timebin, row, cluster, mcLabels);
        mTmpClusterArray[mGlobalRowToRegion[row]]->emplace_back(cluster, std::move(mcLabels));
        mcLabels.clear();
      }
    }
  }
}

//______________________________________________________________________________
void HwClusterer::finishFrame(bool clear)
{
  unsigned HB;
  // Search in last remaining timebins for clusters
  for (int i = mLastTimebin; i - mLastTimebin < mTimebinsInBuffer; ++i) {
    HB = i < 2 ? 0 : (i - 2) / 447; // integer division on purpose
    if (HB != mLastHB) {
      writeOutputWithTimeOffset(mLastHB * 447);
    }

    findPeaksForTime(i);
    computeClusterForTime(i - 2);
    clearBuffer(i + 1);
    mLastHB = HB;
  }
  writeOutputWithTimeOffset(mLastHB * 447);

  if (clear) {
    for (int i = 0; i < mTimebinsInBuffer; ++i)
      clearBuffer(i);
  }
}

//______________________________________________________________________________
void HwClusterer::clearBuffer(int timebin)
{
  const int wrappedTime = mapTimeInRange(timebin);
  mMCtruth[wrappedTime].reset();
  mCurrentMcContainerInBuffer &= ~(0x1 << wrappedTime); // clear bit
  for (unsigned short row = 0; row < mNumRows; ++row) {
    // reset timebin which is not needed anymore
    // TODO: for simulation fill with pedestal/noise instead of 0
    // TODO: try to improve speed
    std::fill(mDataBuffer[row].begin() + wrappedTime * mPadsPerRow[row],
              mDataBuffer[row].begin() + wrappedTime * mPadsPerRow[row] + mPadsPerRow[row] - 1, 0);
    std::fill(mIndexBuffer[row].begin() + wrappedTime * mPadsPerRow[row],
              mIndexBuffer[row].begin() + wrappedTime * mPadsPerRow[row] + mPadsPerRow[row] - 1, -1);
  }
}

//______________________________________________________________________________
void HwClusterer::updateCluster(
  int row, short center_pad, int center_time, short dp, short dt,
  unsigned& qTot, int& pad, int& time, int& sigmaPad2, int& sigmaTime2,
  std::vector<std::pair<MCCompLabel, unsigned>>& mcLabels)
{

  int index = mapTimeInRange(center_time + dt) * mPadsPerRow[row] + center_pad + dp;
  unsigned charge = getFpOfADC(mDataBuffer[row][index]);

  qTot += charge;
  pad += charge * dp;
  time += charge * dt;
  sigmaPad2 += charge * dp * dp;
  sigmaTime2 += charge * dt * dt;

  if (mMCtruth[mapTimeInRange(center_time + dt)] != nullptr) {
    for (auto& label : mMCtruth[mapTimeInRange(center_time + dt)]->getLabels(mIndexBuffer[row][index])) {
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

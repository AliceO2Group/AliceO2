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
#include "DataFormatsTPC/ClusterHardware.h"

#include "FairLogger.h"

#include <cassert>
#include <limits>

using namespace o2::tpc;

//______________________________________________________________________________
HwClusterer::HwClusterer(
  std::vector<ClusterHardwareContainer8kb>* clusterOutputContainer,
  int sectorid, MCLabelContainer* labelOutput)
  : Clusterer(),
    mNumRows(0),
    mNumRowSets(0),
    mCurrentMcContainerInBuffer(0),
    mSplittingMode(0),
    mClusterSector(sectorid),
    mLastTimebin(-1),
    mLastHB(0),
    mPeakChargeThreshold(2),
    mContributionChargeThreshold(0),
    mClusterCounter(0),
    mIsContinuousReadout(true),
    mRejectSinglePadClusters(false),
    mRejectSingleTimeClusters(false),
    mRejectLaterTimebin(false),
    mPadsPerRowSet(),
    mGlobalRowToRegion(),
    mGlobalRowToLocalRow(),
    mGlobalRowToVcIndex(),
    mGlobalRowToRowSet(),
    mDataBuffer(),
    mIndexBuffer(),
    mMCtruth(),
    mTmpClusterArray(),
    mTmpLabelArray(),
    mClusterMcLabelArray(labelOutput),
    mClusterArray(clusterOutputContainer)
{
  LOG(DEBUG) << "Enter Initializer of HwClusterer";

  // Given sector ID must be within 0 and 35 for a proper CRU ID calculation
  assert(sectorid >= 0 && sectorid < 36);

  /*
   * Prepare temporary storage for digits
   */
  Mapper& mapper = Mapper::instance();

  mNumRows = mapper.getNumberOfRows();
  mNumRowSets = std::ceil(mapper.getNumberOfRows() / Vc::uint_v::Size);
  mDataBuffer.resize(mNumRowSets);
  mIndexBuffer.resize(mNumRowSets);
  mPadsPerRowSet.resize(mNumRowSets);

  mGlobalRowToVcIndex.resize(mNumRows);
  mGlobalRowToRowSet.resize(mNumRows);
  for (unsigned short row = 0; row < mNumRowSets; ++row) {
    // add two empty pads on the left and on the right
    int max = 0;
    for (int subrow = 0; subrow < Vc::uint_v::Size; ++subrow) {
      max = std::max(max, mapper.getNumberOfPadsInRowSector(row * Vc::uint_v::Size + subrow) + 2 + 2);
      mGlobalRowToRowSet[row * Vc::uint_v::Size + subrow] = row;
      mGlobalRowToVcIndex[row * Vc::uint_v::Size + subrow] = subrow;
    }
    mPadsPerRowSet[row] = max;

    // prepare for number of timebins
    mDataBuffer[row].resize(mPadsPerRowSet[row] * mTimebinsInBuffer, 0);
    mIndexBuffer[row].resize(mPadsPerRowSet[row] * mTimebinsInBuffer, -1);
  }

  mTmpClusterArray.resize(10);
  mTmpLabelArray.resize(10);
  for (unsigned short region = 0; region < 10; ++region) {
    mTmpClusterArray[region] = std::make_unique<std::vector<ClusterHardware>>();
    mTmpLabelArray[region] = std::make_unique<std::vector<std::vector<std::pair<MCCompLabel, unsigned>>>>();
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
void HwClusterer::process(std::vector<o2::tpc::Digit> const& digits, MCLabelContainer const* mcDigitTruth, bool clearContainerFirst)
{
  if (clearContainerFirst) {
    if (mClusterArray)
      mClusterArray->clear();

    if (mClusterMcLabelArray)
      mClusterMcLabelArray->clear();
    mClusterCounter = 0;
  }

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
        HB = i < 3 ? 0 : (i - 3) / 447; // integer division on purpose
        if (HB != mLastHB) {
          writeOutputWithTimeOffset(mLastHB * 447);
        }

        /*
         * For each row(set), we first compute all the pad relations for the
         * latest timebin (i), afterwards the clusters for timebin i-2 are
         * collected and computed. We need the -2 because a cluster spreads
         * over 5 timbins, and the relations are always computed with respect
         * to the older timebin. Also a (i-1) and (i-2) would be possible, but
         * doens't matter.
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
        computeClusterForTime(i - 3);

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
    index = mapTimeInRange(digit.getTimeStamp()) * mPadsPerRowSet[mGlobalRowToRowSet[digit.getRow()]] + (digit.getPad() + 2);
    // offset of digit pad because of 2 empty pads on both sides

    float charge = digit.getChargeFloat();

    // TODO: fill noise here as well if necessary
    if (mPedestalObject) {
      charge -= mPedestalObject->getValue(CRU(digit.getCRU()), digit.getRow(), digit.getPad());
    }
    /*
     * charge could be smaller than 0 due to pedestal subtraction, if so set it to zero
     * noise thresholds for zero suppression could also be done here ...
     */
    if (charge < 0) {
      charge = 0;
    }

    mDataBuffer[mGlobalRowToRowSet[digit.getRow()]][index][mGlobalRowToVcIndex[digit.getRow()]] = static_cast<unsigned>(charge * (1 << 4));
    if (mDataBuffer[mGlobalRowToRowSet[digit.getRow()]][index][mGlobalRowToVcIndex[digit.getRow()]] > 0x3FFF) {
      mDataBuffer[mGlobalRowToRowSet[digit.getRow()]][index][mGlobalRowToVcIndex[digit.getRow()]] = 0x3FFF; // set only 14 LSBs
    }

    mIndexBuffer[mGlobalRowToRowSet[digit.getRow()]][index][mGlobalRowToVcIndex[digit.getRow()]] = digitIndex++;

    mLastTimebin = digit.getTimeStamp();
  }

  if (!mIsContinuousReadout)
    finishFrame(true);

  if (digits.size() != 0)
    LOG(DEBUG) << "Event ranged from time bin " << digits.front().getTimeStamp() << " to " << digits.back().getTimeStamp() << ".";
}

//______________________________________________________________________________
void HwClusterer::finishProcess(std::vector<o2::tpc::Digit> const& digits, MCLabelContainer const* mcDigitTruth, bool clearContainerFirst)
{
  // Process the last digits (if there are any)
  process(digits, mcDigitTruth, clearContainerFirst);

  // Search in last remaining timebins
  finishFrame(false);
}

//______________________________________________________________________________
void HwClusterer::hwClusterProcessor(const Vc::uint_m peakMask, unsigned qMaxIndex, short centerPad, int centerTime, unsigned short row)
{
  Vc::uint_v qTot = 0;
  Vc::int_v pad = 0;
  Vc::int_v time = 0;
  Vc::int_v sigmaPad2 = 0;
  Vc::int_v sigmaTime2 = 0;
  Vc::int_v flags = 0;

  using labelPair = std::pair<MCCompLabel, unsigned>;
  std::vector<std::unique_ptr<std::vector<labelPair>>> mcLabels(Vc::uint_v::Size);
  for (int i = 0; i < Vc::uint_v::Size; ++i) {
    mcLabels[i] = std::make_unique<std::vector<labelPair>>();
  }

  const unsigned llttIndex = mapTimeInRange(centerTime - 2) * mPadsPerRowSet[row] + centerPad - 2;
  const unsigned lltIndex = mapTimeInRange(centerTime - 1) * mPadsPerRowSet[row] + centerPad - 2;
  const unsigned llIndex = mapTimeInRange(centerTime + 0) * mPadsPerRowSet[row] + centerPad - 2;
  const unsigned llbIndex = mapTimeInRange(centerTime + 1) * mPadsPerRowSet[row] + centerPad - 2;
  const unsigned llbbIndex = mapTimeInRange(centerTime + 2) * mPadsPerRowSet[row] + centerPad - 2;

  const unsigned lttIndex = llttIndex + 1;
  const unsigned ltIndex = lltIndex + 1;
  const unsigned lIndex = llIndex + 1;
  const unsigned lbIndex = llbIndex + 1;
  const unsigned lbbIndex = llbbIndex + 1;

  const unsigned ttIndex = lttIndex + 1;
  const unsigned tIndex = ltIndex + 1;
  const unsigned bIndex = lbIndex + 1;
  const unsigned bbIndex = lbbIndex + 1;

  const unsigned rttIndex = ttIndex + 1;
  const unsigned rtIndex = tIndex + 1;
  const unsigned rIndex = lIndex + 2;
  const unsigned rbIndex = bIndex + 1;
  const unsigned rbbIndex = bbIndex + 1;

  const unsigned rrttIndex = rttIndex + 1;
  const unsigned rrtIndex = rtIndex + 1;
  const unsigned rrIndex = rIndex + 1;
  const unsigned rrbIndex = rbIndex + 1;
  const unsigned rrbbIndex = rbbIndex + 1;

  Vc::uint_m selectionMask;
  if (mSplittingMode == 0) {
    // Charge is not splitted between nearby peaks, every peak uses all charges

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
    for (short dt = -1; dt <= 1; ++dt) {
      for (short dp = -1; dp <= 1; ++dp) {
        updateCluster(peakMask, row, centerPad, centerTime, dp, dt, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
      }
    }

    // Use outer pads (o) only, if corresponding inner pad (i) is above contribution threshold
    //        o
    //        i
    // o  i   C   i   o
    //        i
    //        o
    selectionMask = peakMask & (getFpOfADC(mDataBuffer[row][lIndex]) > (mContributionChargeThreshold << 4));
    updateCluster(selectionMask, row, centerPad, centerTime, -2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);

    selectionMask = peakMask & (getFpOfADC(mDataBuffer[row][rIndex]) > (mContributionChargeThreshold << 4));
    updateCluster(selectionMask, row, centerPad, centerTime, +2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);

    selectionMask = peakMask & (getFpOfADC(mDataBuffer[row][tIndex]) > (mContributionChargeThreshold << 4));
    updateCluster(selectionMask, row, centerPad, centerTime, 0, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);

    selectionMask = peakMask & (getFpOfADC(mDataBuffer[row][bIndex]) > (mContributionChargeThreshold << 4));
    updateCluster(selectionMask, row, centerPad, centerTime, 0, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);

    // o  o       o   o
    // o  i       i   o
    //        C
    // o  i       i   o
    // o  o       o   o
    selectionMask = peakMask & (getFpOfADC(mDataBuffer[row][ltIndex]) > (mContributionChargeThreshold << 4));
    updateCluster(selectionMask, row, centerPad, centerTime, -2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(selectionMask, row, centerPad, centerTime, -2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(selectionMask, row, centerPad, centerTime, -1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);

    selectionMask = peakMask & (getFpOfADC(mDataBuffer[row][lbIndex]) > (mContributionChargeThreshold << 4));
    updateCluster(selectionMask, row, centerPad, centerTime, -2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(selectionMask, row, centerPad, centerTime, -2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(selectionMask, row, centerPad, centerTime, -1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);

    selectionMask = peakMask & (getFpOfADC(mDataBuffer[row][rtIndex]) > (mContributionChargeThreshold << 4));
    updateCluster(selectionMask, row, centerPad, centerTime, +2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(selectionMask, row, centerPad, centerTime, +2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(selectionMask, row, centerPad, centerTime, +1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);

    selectionMask = peakMask & (getFpOfADC(mDataBuffer[row][rbIndex]) > (mContributionChargeThreshold << 4));
    updateCluster(selectionMask, row, centerPad, centerTime, +2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(selectionMask, row, centerPad, centerTime, +2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);
    updateCluster(selectionMask, row, centerPad, centerTime, +1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);

  } else if (mSplittingMode == 1) {
    // Charge is split at minimum, the minimum itself contributes half too all
    // adjacent peaks (even if there are 3).

    // Use always inner 3x3 matrix
    //    i   i   i
    //    i   C   i
    //    i   i   i
    // but only half the charge if (i) is minimum

    // center
    updateCluster(peakMask, row, centerPad, centerTime, 0, 0, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels);

    // horizontal, look for minimum in pad direction
    auto splitMask = (((mDataBuffer[row][lIndex] >> 26) & 0x1) == 0x1);
    updateCluster(peakMask, row, centerPad, centerTime, -1, 0, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    splitMask = (((mDataBuffer[row][rIndex] >> 26) & 0x1) == 0x1);
    updateCluster(peakMask, row, centerPad, centerTime, +1, 0, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    // vertical look for minimum in time direction
    splitMask = (((mDataBuffer[row][tIndex] >> 25) & 0x1) == 0x1);
    updateCluster(peakMask, row, centerPad, centerTime, 0, -1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    splitMask = (((mDataBuffer[row][bIndex] >> 25) & 0x1) == 0x1);
    updateCluster(peakMask, row, centerPad, centerTime, 0, +1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    // diagonal tl/br, look for minimum in 1. diagonal + vertical + horizontal direction
    splitMask = (((mDataBuffer[row][ltIndex] >> 24) & 0x1) == 0x1) |
                (((mDataBuffer[row][ltIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][ltIndex] >> 26) & 0x1) == 0x1);
    updateCluster(peakMask, row, centerPad, centerTime, -1, -1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    splitMask = (((mDataBuffer[row][rbIndex] >> 24) & 0x1) == 0x1) |
                (((mDataBuffer[row][rbIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][rbIndex] >> 26) & 0x1) == 0x1);
    updateCluster(peakMask, row, centerPad, centerTime, +1, +1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    // diagonal tr/bl, look for minimum in 2. diagonal + vertical + horizontal direction
    splitMask = (((mDataBuffer[row][rtIndex] >> 23) & 0x1) == 0x1) |
                (((mDataBuffer[row][rtIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][rtIndex] >> 26) & 0x1) == 0x1);
    updateCluster(peakMask, row, centerPad, centerTime, +1, -1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    splitMask = (((mDataBuffer[row][lbIndex] >> 23) & 0x1) == 0x1) |
                (((mDataBuffer[row][lbIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][lbIndex] >> 26) & 0x1) == 0x1);
    updateCluster(peakMask, row, centerPad, centerTime, -1, +1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    // Use outer pads (o) only, if corresponding inner pad (i) is above contribution threshold
    //        o
    //        i
    // o  i   C   i   o
    //        i
    //        o
    // and only if inner pad (i) is not a minimum (in corresponding direction).
    // Split charge, if (o) is a minimum.
    selectionMask = peakMask &                                                                     // using (o) (if we have a peak in the center)
                    (getFpOfADC(mDataBuffer[row][lIndex]) > (mContributionChargeThreshold << 4)) & // AND (i) is above threshold
                    (((mDataBuffer[row][lIndex] >> 26) & 0x1) != 0x1);                             // AND (i) is not minimum in corresponding direction
    splitMask = (((mDataBuffer[row][llIndex] >> 26) & 0x1) == 0x1);                                // split (o) if it is miminum in corresponding direction
    updateCluster(selectionMask, row, centerPad, centerTime, -2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][rIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][rIndex] >> 26) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][rrIndex] >> 26) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, +2, 0, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][tIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][tIndex] >> 25) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][ttIndex] >> 26) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, 0, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][bIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][bIndex] >> 25) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][bbIndex] >> 26) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, 0, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    // o  o       o   o
    // o  i       i   o
    //        C
    // o  i       i   o
    // o  o       o   o
    selectionMask = peakMask &                                                                      // using (o) (if we have a peak in the center)
                    (getFpOfADC(mDataBuffer[row][ltIndex]) > (mContributionChargeThreshold << 4)) & // AND (i) is above threshold
                    (((mDataBuffer[row][ltIndex] >> 26) & 0x1) != 0x1);                             // AND (i) is not minimum in corresponding direction
    splitMask = (((mDataBuffer[row][lltIndex] >> 26) & 0x1) == 0x1) |                               // split (o) if it is miminum in corresponding directions
                (((mDataBuffer[row][lltIndex] >> 24) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, -2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &                                                                      // using (o) (if we have a peak in the center)
                    (getFpOfADC(mDataBuffer[row][ltIndex]) > (mContributionChargeThreshold << 4)) & // AND (i) is above threshold
                    (((mDataBuffer[row][ltIndex] >> 24) & 0x1) != 0x1) &                            // AND (i) is not minimum in corresponding directions
                    (((mDataBuffer[row][ltIndex] >> 25) & 0x1) != 0x1) &
                    (((mDataBuffer[row][ltIndex] >> 26) & 0x1) != 0x1) &
                    (((mDataBuffer[row][lltIndex] >> 25) & 0x1) != 0x1) & // AND other (o) is not minimum in corresponding direction
                    (((mDataBuffer[row][lttIndex] >> 26) & 0x1) != 0x1);  // AND other (o) is not minimum in corresponding direction
    splitMask = (((mDataBuffer[row][llttIndex] >> 24) & 0x1) == 0x1) |    // split (o) if it is miminum in corresponding direction
                (((mDataBuffer[row][llttIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][llttIndex] >> 26) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, -2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][ltIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][ltIndex] >> 25) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][lttIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][lttIndex] >> 24) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, -1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][lbIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][lbIndex] >> 26) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][llbIndex] >> 26) & 0x1) == 0x1) |
                (((mDataBuffer[row][llbIndex] >> 23) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, -2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][lbIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][lbIndex] >> 23) & 0x1) != 0x1) &
                    (((mDataBuffer[row][lbIndex] >> 25) & 0x1) != 0x1) &
                    (((mDataBuffer[row][lbIndex] >> 26) & 0x1) != 0x1) &
                    (((mDataBuffer[row][llbIndex] >> 25) & 0x1) != 0x1) &
                    (((mDataBuffer[row][lbbIndex] >> 26) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][llbbIndex] >> 23) & 0x1) == 0x1) |
                (((mDataBuffer[row][llbbIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][llbbIndex] >> 26) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, -2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][lbIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][lbIndex] >> 25) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][lbbIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][lbbIndex] >> 23) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, -1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][rtIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][rtIndex] >> 26) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][rrtIndex] >> 26) & 0x1) == 0x1) |
                (((mDataBuffer[row][rrtIndex] >> 23) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, +2, -1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][rtIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][rtIndex] >> 23) & 0x1) != 0x1) &
                    (((mDataBuffer[row][rtIndex] >> 25) & 0x1) != 0x1) &
                    (((mDataBuffer[row][rtIndex] >> 26) & 0x1) != 0x1) &
                    (((mDataBuffer[row][rrtIndex] >> 25) & 0x1) != 0x1) &
                    (((mDataBuffer[row][rttIndex] >> 26) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][rrttIndex] >> 23) & 0x1) == 0x1) |
                (((mDataBuffer[row][rrttIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][rrttIndex] >> 26) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, +2, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][rtIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][rtIndex] >> 25) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][rttIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][rttIndex] >> 23) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, +1, -2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][rbIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][rbIndex] >> 26) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][rrbIndex] >> 26) & 0x1) == 0x1) |
                (((mDataBuffer[row][rrbIndex] >> 24) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, +2, +1, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][rbIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][rbIndex] >> 24) & 0x1) != 0x1) &
                    (((mDataBuffer[row][rbIndex] >> 25) & 0x1) != 0x1) &
                    (((mDataBuffer[row][rbIndex] >> 26) & 0x1) != 0x1) &
                    (((mDataBuffer[row][rrbIndex] >> 25) & 0x1) != 0x1) &
                    (((mDataBuffer[row][rbbIndex] >> 26) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][rrbbIndex] >> 24) & 0x1) == 0x1) |
                (((mDataBuffer[row][rrbbIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][rrbbIndex] >> 26) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, +2, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);

    selectionMask = peakMask &
                    (getFpOfADC(mDataBuffer[row][rbIndex]) > (mContributionChargeThreshold << 4)) &
                    (((mDataBuffer[row][rbIndex] >> 25) & 0x1) != 0x1);
    splitMask = (((mDataBuffer[row][rbbIndex] >> 25) & 0x1) == 0x1) |
                (((mDataBuffer[row][rbbIndex] >> 24) & 0x1) == 0x1);
    updateCluster(selectionMask, row, centerPad, centerTime, +1, +2, qTot, pad, time, sigmaPad2, sigmaTime2, mcLabels, splitMask);
  }

  selectionMask = peakMask;
  if (mRejectSinglePadClusters)
    selectionMask &= !(sigmaPad2 == 0);
  if (mRejectSingleTimeClusters)
    selectionMask &= !(sigmaTime2 == 0);

  ClusterHardware tmpCluster;
  for (int i = 0; i < Vc::uint_v::Size; ++i) {
    if (selectionMask[i]) {

      mTmpClusterArray[mGlobalRowToRegion[row * Vc::uint_v::Size + i]]->emplace_back();
      mTmpClusterArray[mGlobalRowToRegion[row * Vc::uint_v::Size + i]]->back().setCluster(
        centerPad - 2,    // we have two artificial empty pads "on the left" which needs to be subtracted
        centerTime % 447, // the time within a HB
        pad[i], time[i],
        sigmaPad2[i], sigmaTime2[i],
        (getFpOfADC(mDataBuffer[row][qMaxIndex])[i] >> 3), // keep only 1 fixed point precision bit
        qTot[i],
        mGlobalRowToLocalRow[row * Vc::uint_v::Size + i], // the hardware knows only about the local row
        flags[i]);

      std::sort(mcLabels[i]->begin(), mcLabels[i]->end(), [](const labelPair& a, const labelPair& b) { return a.second > b.second; });
      mTmpLabelArray[mGlobalRowToRegion[row * Vc::uint_v::Size + i]]->push_back(std::move(*mcLabels[i]));
    }
  }
}

//______________________________________________________________________________
void HwClusterer::hwPeakFinder(unsigned qMaxIndex, short centerPad, int mappedCenterTime, unsigned short row)
{

  // Always the center pad is compared with a fixed other pad. The first
  // comparison is in pad direction. That bin (p,t) is a peak in pad direction,
  // it has to be >= then (p-1,t) and in the next iteration, where the next pad
  // is the center, the comparison of (the new center with the new left one)
  // must be false, so that (p+1,t) > (p,t). Afterwards, this is also done in
  // time direction and both diagonal directions. With those 4 comparisons, all
  // 8 neighboring pads are checked iteratively.
  //
  // Example in pad direction:
  //     p-2 p-1  p  p+1 p+2                  p-1  p  p+1 p+2 p+3
  // t-2  o   o   o   o   o                    o   o   o   o   o
  // t-1  o   i   i   i   o   next iteration   o   i   i   i   o
  // t    o   I<->C   i   o  --------------->  O   I<->c   i   o
  // t+1  o   i   i   i   o                    o   i   i   i   o
  // t+2  o   o   o   o   o                    o   o   o   o   o
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
  //  23 | minimum in 2. diagonal direction

  const int compareIndex0 = mappedCenterTime * mPadsPerRowSet[row] + centerPad - 1;
  const int compareIndex1 = mapTimeInRange(mappedCenterTime - 1) * mPadsPerRowSet[row] + centerPad;
  const int compareIndex2 = compareIndex1 - 1;
  const int compareIndex3 = compareIndex1 + 1;
  const auto adcMask = (getFpOfADC(mDataBuffer[row][qMaxIndex]) == 0) &
                       (getFpOfADC(mDataBuffer[row][compareIndex0]) == 0) &
                       (getFpOfADC(mDataBuffer[row][compareIndex1]) == 0) &
                       (getFpOfADC(mDataBuffer[row][compareIndex2]) == 0) &
                       (getFpOfADC(mDataBuffer[row][compareIndex3]) == 0);
  if (adcMask.isFull()) {
    // if all 0, we can skip the following part and directly set the bits where
    // the comparisons were true, TODO: still possible if difference is applied?
    mDataBuffer[row][qMaxIndex] |= (0x1 << 31);
    mDataBuffer[row][compareIndex0] &= ~(0x1 << 31);
    mDataBuffer[row][qMaxIndex] |= (0x1 << 30);
    mDataBuffer[row][compareIndex1] &= ~(0x1 << 30);
    mDataBuffer[row][qMaxIndex] |= (0x1 << 29);
    mDataBuffer[row][compareIndex2] &= ~(0x1 << 29);
    mDataBuffer[row][qMaxIndex] |= (0x1 << 28);
    mDataBuffer[row][compareIndex3] &= ~(0x1 << 28);
    return;
  }

  //////////////////////////////////////
  // Comparison in pad direction
  // TODO: define needed difference
  auto tmpMask = getFpOfADC(mDataBuffer[row][qMaxIndex]) >= getFpOfADC(mDataBuffer[row][compareIndex0]);
  // if true:
  //  - current center could be peak
  //  - cleare possible maxBit of other
  //  - other is minimum if minBit was already set before
  Vc::where(tmpMask) | mDataBuffer[row][qMaxIndex] |= (0x1 << 31);
  Vc::where(tmpMask) | mDataBuffer[row][compareIndex0] &= ~(0x1 << 31);

  // if false:
  //  - current center could be minimum
  //  - cleare possible minBit of other
  //  - other is peak if maxBit was already set before
  Vc::where(!tmpMask) | mDataBuffer[row][qMaxIndex] |= (0x1 << 26);
  Vc::where(!tmpMask) | mDataBuffer[row][compareIndex0] &= ~(0x1 << 26);

  //////////////////////////////////////
  // Comparison in time direction
  tmpMask = getFpOfADC(mDataBuffer[row][qMaxIndex]) >= getFpOfADC(mDataBuffer[row][compareIndex1]);
  Vc::where(tmpMask) | mDataBuffer[row][qMaxIndex] |= (0x1 << 30);
  Vc::where(tmpMask) | mDataBuffer[row][compareIndex1] &= ~(0x1 << 30);
  Vc::where(!tmpMask) | mDataBuffer[row][qMaxIndex] |= (0x1 << 25);
  Vc::where(!tmpMask) | mDataBuffer[row][compareIndex1] &= ~(0x1 << 25);

  //////////////////////////////////////
  // Comparison in 1. diagonal direction
  tmpMask = getFpOfADC(mDataBuffer[row][qMaxIndex]) >= getFpOfADC(mDataBuffer[row][compareIndex2]);
  Vc::where(tmpMask) | mDataBuffer[row][qMaxIndex] |= (0x1 << 29);
  Vc::where(tmpMask) | mDataBuffer[row][compareIndex2] &= ~(0x1 << 29);
  Vc::where(!tmpMask) | mDataBuffer[row][qMaxIndex] |= (0x1 << 24);
  Vc::where(!tmpMask) | mDataBuffer[row][compareIndex2] &= ~(0x1 << 24);

  //////////////////////////////////////
  // Comparison in 2. diagonal direction
  tmpMask = getFpOfADC(mDataBuffer[row][qMaxIndex]) >= getFpOfADC(mDataBuffer[row][compareIndex3]);
  Vc::where(tmpMask) | mDataBuffer[row][qMaxIndex] |= (0x1 << 28);
  Vc::where(tmpMask) | mDataBuffer[row][compareIndex3] &= ~(0x1 << 28);
  Vc::where(!tmpMask) | mDataBuffer[row][qMaxIndex] |= (0x1 << 23);
  Vc::where(!tmpMask) | mDataBuffer[row][compareIndex3] &= ~(0x1 << 23);

  //////////////////////////////////////
  // Comparison peak threshold
  Vc::where(getFpOfADC(mDataBuffer[row][qMaxIndex]) > (mPeakChargeThreshold << 4)) | mDataBuffer[row][qMaxIndex] |= (0x1 << 27);
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

      for (int c = 0; c < mTmpClusterArray[region]->size(); ++c) {
        // if the container is full, create a new one
        if (clusterContainer->numberOfClusters == mClusterArray->back().getMaxNumberOfClusters()) {
          mClusterArray->emplace_back();
          clusterContainer = mClusterArray->back().getContainer();
          clusterContainer->CRU = mClusterSector * 10 + region;
          clusterContainer->numberOfClusters = 0;
          clusterContainer->timeBinOffset = timeOffset;
        }
        // Copy cluster and increment cluster counter
        clusterContainer->clusters[clusterContainer->numberOfClusters++] = std::move((*mTmpClusterArray[region])[c]);
        if (mClusterMcLabelArray) {
          for (auto& mcLabel : (*mTmpLabelArray[region])[c]) {
            mClusterMcLabelArray->addElement(mClusterCounter, mcLabel.first);
          }
        }
        ++mClusterCounter;
      }
    }

    // Clear copied temporary storage
    mTmpClusterArray[region]->clear();
    mTmpLabelArray[region]->clear();
  }
}

//______________________________________________________________________________
void HwClusterer::findPeaksForTime(int timebin)
{
  if (timebin < 0)
    return;

  const unsigned timeBinWrapped = mapTimeInRange(timebin);
  for (unsigned short row = 0; row < mNumRowSets; ++row) {
    const unsigned padOffset = timeBinWrapped * mPadsPerRowSet[row];
    // two empty pads on the left and right without a cluster peak, check one
    // beyond rightmost pad for remaining relations
    for (short pad = 1; pad < mPadsPerRowSet[row] - 1; ++pad) {
      const unsigned qMaxIndex = padOffset + pad;
      hwPeakFinder(qMaxIndex, pad, timeBinWrapped, row);
    }
  }
}

//______________________________________________________________________________
void HwClusterer::computeClusterForTime(int timebin)
{
  if (timebin < 0)
    return;

  const unsigned timeBinWrapped = mapTimeInRange(timebin);
  if (mRejectLaterTimebin) {
    const unsigned previousTimeBinWrapped = mapTimeInRange(timebin - 2);
    for (unsigned short row = 0; row < mNumRowSets; ++row) {
      const unsigned padOffset = timeBinWrapped * mPadsPerRowSet[row];
      const unsigned previousPadOffset = previousTimeBinWrapped * mPadsPerRowSet[row];
      // two empty pads on the left and right without a cluster peak
      for (short pad = 2; pad < mPadsPerRowSet[row] - 2; ++pad) {
        const unsigned qMaxIndex = padOffset + pad;
        const unsigned qMaxPreviousIndex = previousPadOffset + pad;

        // TODO: define needed difference
        const auto peakMask = ((mDataBuffer[row][qMaxIndex] >> 27) == 0x1F) &                                              //  True if current pad is peak AND
                              (getFpOfADC(mDataBuffer[row][qMaxIndex]) > getFpOfADC(mDataBuffer[row][qMaxPreviousIndex]) | // previous has smaller charge
                               !((mDataBuffer[row][qMaxPreviousIndex] >> 27) == 0x1F));                                    //  or previous one was not a peak
        if (peakMask.isEmpty())
          continue;

        hwClusterProcessor(peakMask, qMaxIndex, pad, timebin, row);
      }
    }
  } else {
    for (unsigned short row = 0; row < mNumRowSets; ++row) {
      const unsigned padOffset = timeBinWrapped * mPadsPerRowSet[row];
      // two empty pads on the left and right without a cluster peak
      for (short pad = 2; pad < mPadsPerRowSet[row] - 2; ++pad) {
        const unsigned qMaxIndex = padOffset + pad;

        const auto peakMask = ((mDataBuffer[row][qMaxIndex] >> 27) == 0x1F);
        if (peakMask.isEmpty())
          continue;

        hwClusterProcessor(peakMask, qMaxIndex, pad, timebin, row);
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
    HB = i < 3 ? 0 : (i - 3) / 447; // integer division on purpose
    if (HB != mLastHB) {
      writeOutputWithTimeOffset(mLastHB * 447);
    }

    findPeaksForTime(i);
    computeClusterForTime(i - 3);
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
  for (unsigned short row = 0; row < mNumRowSets; ++row) {
    // reset timebin which is not needed anymore
    std::fill(mDataBuffer[row].begin() + wrappedTime * mPadsPerRowSet[row],
              mDataBuffer[row].begin() + wrappedTime * mPadsPerRowSet[row] + mPadsPerRowSet[row], 0);
    std::fill(mIndexBuffer[row].begin() + wrappedTime * mPadsPerRowSet[row],
              mIndexBuffer[row].begin() + wrappedTime * mPadsPerRowSet[row] + mPadsPerRowSet[row], -1);
  }
}

//______________________________________________________________________________
void HwClusterer::updateCluster(
  const Vc::uint_m selectionMask, int row, short centerPad, int centerTime, short dp, short dt,
  Vc::uint_v& qTot, Vc::int_v& pad, Vc::int_v& time, Vc::int_v& sigmaPad2, Vc::int_v& sigmaTime2,
  std::vector<std::unique_ptr<std::vector<std::pair<MCCompLabel, unsigned>>>>& mcLabels, const Vc::uint_m splitMask)
{
  if (selectionMask.isEmpty())
    return;

  const int mappedTime = mapTimeInRange(centerTime + dt);
  const int index = mappedTime * mPadsPerRowSet[row] + centerPad + dp;

  // If the charge should be split, only half of the charge is used
  Vc::where(selectionMask & splitMask) | qTot += (getFpOfADC(mDataBuffer[row][index]) >> 1);
  Vc::where(selectionMask & splitMask) | pad += (getFpOfADC(mDataBuffer[row][index]) >> 1) * dp;
  Vc::where(selectionMask & splitMask) | time += (getFpOfADC(mDataBuffer[row][index]) >> 1) * dt;
  Vc::where(selectionMask & splitMask) | sigmaPad2 += (getFpOfADC(mDataBuffer[row][index]) >> 1) * dp * dp;
  Vc::where(selectionMask & splitMask) | sigmaTime2 += (getFpOfADC(mDataBuffer[row][index]) >> 1) * dt * dt;

  // Otherwise the full charge
  Vc::where(selectionMask & (!splitMask)) | qTot += getFpOfADC(mDataBuffer[row][index]);
  Vc::where(selectionMask & (!splitMask)) | pad += getFpOfADC(mDataBuffer[row][index]) * dp;
  Vc::where(selectionMask & (!splitMask)) | time += getFpOfADC(mDataBuffer[row][index]) * dt;
  Vc::where(selectionMask & (!splitMask)) | sigmaPad2 += getFpOfADC(mDataBuffer[row][index]) * dp * dp;
  Vc::where(selectionMask & (!splitMask)) | sigmaTime2 += getFpOfADC(mDataBuffer[row][index]) * dt * dt;

  for (int i = 0; i < Vc::uint_v::Size; ++i) {
    if (selectionMask[i] && mMCtruth[mappedTime] != nullptr) {
      for (auto& label : mMCtruth[mappedTime]->getLabels(mIndexBuffer[row][index][i])) {
        bool isKnown = false;
        for (auto& vecLabel : *mcLabels[i]) {
          if (label == vecLabel.first) {
            ++vecLabel.second;
            isKnown = true;
          }
        }
        if (!isKnown) {
          mcLabels[i]->emplace_back(label, 1);
        }
      }
    }
  }
}

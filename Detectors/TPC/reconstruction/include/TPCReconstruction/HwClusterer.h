// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HwClusterer.h
/// \brief Class for TPC HW cluster finding
/// \author Sebastian Klewin <sebastian.klewin@cern.ch>

#ifndef ALICEO2_TPC_HWClusterer_H_
#define ALICEO2_TPC_HWClusterer_H_

#include <Vc/Vc>

#include "TPCReconstruction/Clusterer.h"
#include "DataFormatsTPC/Helpers.h"

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <vector>
#include <utility>
#include <memory>

namespace o2{
namespace TPC {

class Digit;
class Cluster;
class ClusterHardware;

/// \class HwClusterer
/// \brief Class for TPC HW cluster finding
class HwClusterer : public Clusterer
{

 private:
  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

  /// Main Constructor
  HwClusterer(
    std::vector<ClusterHardwareContainer8kb>* clusterOutputContainer,
    std::vector<Cluster>* clusterOutputSimple, int sectorid,
    MCLabelContainer* labelOutput);

 public:
  /// Constructor
  /// \param clusterOutput is pointer to vector to be filled with ClusterHardwareContainers
  /// \param labelOutput is pointer to storage to be filled with MC labels
  /// \param sectorid is sector number to be processed
  HwClusterer(
    std::vector<ClusterHardwareContainer8kb>* clusterOutput,
    int sectorid,
    MCLabelContainer* labelOutput = nullptr);

  /// Constructor
  /// \param clusterOutput is pointer to vector to be filled with clusters
  /// \param labelOutput is pointer to storage to be filled with MC labels
  /// \param sectorid is sector number to be processed
  HwClusterer(
    std::vector<Cluster>* clusterOutput,
    int sectorid,
    MCLabelContainer* labelOutput = nullptr);

  /// Destructor
  ~HwClusterer() override = default;

  /// Copy Constructor
  HwClusterer(HwClusterer const& other) = default;

  /// Process digits
  /// \param digits Container with TPC digits
  /// \param mcDigitTruth MC Digit Truth container
  void process(std::vector<o2::TPC::Digit> const& digits, MCLabelContainer const* mcDigitTruth) override;

  /// Finish processing digits
  /// \param digits Container with TPC digits
  /// \param mcDigitTruth MC Digit Truth container
  void finishProcess(std::vector<o2::TPC::Digit> const& digits, MCLabelContainer const* mcDigitTruth) override;

  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  void setContinuousReadout(bool isContinuous);

  /// Sets the charge threshold for the peak
  /// \param charge Threshold which will be used
  void setPeakChargeThreshold(unsigned charge);

  /// Sets the charge threshold for the contributing pads
  /// \param charge Threshold which will be used
  void setContributionChargeThreshold(unsigned charge);

 private:
  /*
   * Helper functions
   */

  /// HW Cluster Processor
  /// \param peakMask       VC-mask with only peaks enabled
  /// \param qMaxIndex      Buffer index of center pad
  /// \param center_pad     Pad number to be checked for cluster
  /// \param center_time    Time to be checked for cluster
  /// \param row            Row number for cluster properties
  void hwClusterProcessor(Vc::uint_m peakMask, unsigned qMaxIndex, short center_pad, int center_time, unsigned short row);

  /// HW Peak Finder
  /// \param qMaxIndex        Buffer index of central pad
  /// \param center_pad       Pad number to be checked for cluster
  /// \param center_time      Time to be checked for cluster
  /// \param row              Row number for cluster properties
  void hwPeakFinder(unsigned qMaxIndex, short center_pad, int center_time, unsigned short row);

  /// Helper function to update cluster properties and MC labels
  /// \param selectionMask  VC-mask with slected pads enabled
  /// \param row            Current row
  /// \param center_pad     Pad of peak
  /// \param center_time    Timebin of peak
  /// \param dp             delta pad
  /// \param dt             delta time
  /// \param qTot           Total charge
  /// \param pad            Weighted pad parameter
  /// \param time           Weighted time parameter
  /// \param sigmaPad2      Weighted sigma pad ^2 parameter
  /// \param sigmaTime2     Weighted sigma time ^2 parameter
  /// \param mcLabel        Vector with MClabel-counter-pairs
  void updateCluster(const Vc::uint_m selectionMask, int row, short center_pad, int center_time, short dp, short dt, Vc::uint_v& qTot, Vc::int_v& pad, Vc::int_v& time, Vc::int_v& sigmaPad2, Vc::int_v& sigmaTime2, std::vector<std::unique_ptr<std::vector<std::pair<MCCompLabel, unsigned>>>>& mcLabels);

  /// Writes clusters from temporary storage to cluster output
  /// \param timeOffset   Time offset of cluster container
  void writeOutputWithTimeOffset(int timeOffset);

  /// Processes and collects the peaks after they were found
  /// \param timebin  Timebin to cluster peaks
  void computeClusterForTime(int timebin);

  /// Does the Peak Finding in all rows for given timebin
  /// \param timebin  Timebin to cluster peaks
  void findPeaksForTime(int timebin);

  /// Searches for last remaining cluster and writes them out
  /// \param clear    Clears data buffer afterwards (for not continuous readout)
  void finishFrame(bool clear = false);

  /// Clears the buffer at given timebin TODO: and fills timebin with noise + pedestal
  /// \param timebin  Timebin to be cleared
  void clearBuffer(int timebin);

  /// Returns least significant set bit of mCurrentMcContainerInBuffer, only the mTimebinsInBuffer LSBs are checked
  /// \return LSB index which is set
  short getFirstSetBitOfField();

  /// Maps the given time into the available range of the stored buffer
  /// \param time   time to be maped
  /// \return (mTimebinsInBuffer + (time % mTimebinsInBuffer)) % mTimebinsInBuffer which is always in range [0, mTimebinsInBuffer-1] even if time < 0
  int mapTimeInRange(int time);

  /// Returns the 14 LSB FP part of the value
  /// \param value  some value
  Vc::uint_v getFpOfADC(const Vc::uint_v value);

  /// Does the comparison between two pads and sets the bits accordingly
  /// \param qMaxIndex      Buffer index of central pad
  /// \param compareIndex   Buffer index of pad to compare with
  /// \param bitMax         Bit to be set for peak finding
  /// \param bitMin         Bit to be set for minimum finding
  /// \param row            Row number
  void compareForPeak(const unsigned qMaxIndex, const unsigned compareIndex, const unsigned bitMax, const unsigned bitMin, const unsigned short row);

  /*
   * class members
   */
  static const int mTimebinsInBuffer = 5;

  unsigned short mNumRows;               ///< Number of rows in this sector
  unsigned short mNumRowSets;            ///< Number of row sets (Number of rows / Vc::Size) in this sector
  short mCurrentMcContainerInBuffer;     ///< Bit field, where to find the current MC container in buffer
  int mClusterSector;                    ///< Sector to be processed
  int mLastTimebin;                      ///< Last time bin of previous event
  unsigned mLastHB;                      ///< Last HB bin of previous event
  unsigned mPeakChargeThreshold;         ///< Charge threshold for the central peak in ADC counts
  unsigned mContributionChargeThreshold; ///< Charge threshold for the contributing pads in ADC counts
  unsigned mClusterCounter;              ///< Cluster counter in output container for MC truth matching
  bool mIsContinuousReadout;             ///< Switch for continuous readout

  std::vector<unsigned short> mPadsPerRow;                       ///< Number of pads for given row (offset of 2 pads on both sides is already added)
  std::vector<unsigned short> mPadsPerRowSet;                    ///< Number of pads for given row set (offset of 2 pads on both sides is already added), a row set combines rows for parallel SIMD processing
  std::vector<unsigned short> mGlobalRowToRegion;                ///< Mapping global row number to region
  std::vector<unsigned short> mGlobalRowToLocalRow;              ///< Converting global row number to local row number within region
  std::vector<unsigned short> mGlobalRowToVcIndex;               ///< Converting global row number to VC index
  std::vector<unsigned short> mGlobalRowToRowSet;                ///< Converting global row number to row set number
  std::vector<std::vector<Vc::uint_v>> mDataBuffer;              ///< Buffer with digits (+noise +CM +...)
  std::vector<std::vector<Vc::int_v>> mIndexBuffer;              ///< Buffer with digits indices for MC labels
  std::vector<std::shared_ptr<MCLabelContainer const>> mMCtruth; ///< MC truth information of timebins in buffer

  std::vector<std::unique_ptr<std::vector<ClusterHardware>>> mTmpClusterArray;                             ///< Temporary cluster storage for each region to accumulate cluster before filling output container
  std::vector<std::unique_ptr<std::vector<std::vector<std::pair<MCCompLabel, unsigned>>>>> mTmpLabelArray; ///< Temporary cluster storage for each region to accumulate cluster before filling output container

  std::vector<ClusterHardwareContainer8kb>* mClusterArray; ///< Pointer to output cluster container
  std::vector<Cluster>* mPlainClusterArray;                ///< Pointer to output cluster container
  MCLabelContainer* mClusterMcLabelArray;                  ///< Pointer to MC Label container
};

inline void HwClusterer::setContinuousReadout(bool isContinuous)
{
  mIsContinuousReadout = isContinuous;
}

inline void HwClusterer::setPeakChargeThreshold(unsigned charge)
{
  mPeakChargeThreshold = charge;
}

inline void HwClusterer::setContributionChargeThreshold(unsigned charge)
{
  mContributionChargeThreshold = charge;
}

inline int HwClusterer::mapTimeInRange(int time)
{
  return (mTimebinsInBuffer + (time % mTimebinsInBuffer)) % mTimebinsInBuffer;
}

inline Vc::uint_v HwClusterer::getFpOfADC(const Vc::uint_v value)
{
  return value & 0x3FFF;
}

inline short HwClusterer::getFirstSetBitOfField()
{
  for (short i = 0; i < mTimebinsInBuffer; ++i) {
    if ((mCurrentMcContainerInBuffer >> i) & 0x1)
      return i;
  }
  return -1;
}

inline void HwClusterer::compareForPeak(const unsigned qMaxIndex, const unsigned compareIndex, const unsigned bitMax, const unsigned bitMin, const unsigned short row)
{

  const auto tmpMask = getFpOfADC(mDataBuffer[row][qMaxIndex]) >= getFpOfADC(mDataBuffer[row][compareIndex]);

  // current center could be peak in one direction
  where(tmpMask) | mDataBuffer[row][qMaxIndex] |= (0x1 << bitMax);

  // other is smaller than center
  // and if other one was not a peak candidate before (bit is 0), it is a minimum in one direction,
  // so bitMin has to be set to inverse of bitMax of pad to compare
  where(tmpMask) | mDataBuffer[row][compareIndex] |= (~mDataBuffer[row][compareIndex] & (0x1 << bitMax)) >> (bitMax - bitMin);

  // other is not peak
  where(tmpMask) | mDataBuffer[row][compareIndex] &= ~(0x1 << bitMax);

  // other is peak if bit was already set
  where(!tmpMask) | mDataBuffer[row][compareIndex] |= (mDataBuffer[row][compareIndex] & (0x1 << bitMax));
}

inline void HwClusterer::updateCluster(
  const Vc::uint_m selectionMask, int row, short center_pad, int center_time, short dp, short dt,
  Vc::uint_v& qTot, Vc::int_v& pad, Vc::int_v& time, Vc::int_v& sigmaPad2, Vc::int_v& sigmaTime2,
  std::vector<std::unique_ptr<std::vector<std::pair<MCCompLabel, unsigned>>>>& mcLabels)
{
  const int mappedTime = mapTimeInRange(center_time + dt);
  const int index = mappedTime * mPadsPerRowSet[row] + center_pad + dp;

  where(selectionMask) | qTot += getFpOfADC(mDataBuffer[row][index]);
  where(selectionMask) | pad += getFpOfADC(mDataBuffer[row][index]) * dp;
  where(selectionMask) | time += getFpOfADC(mDataBuffer[row][index]) * dt;
  where(selectionMask) | sigmaPad2 += getFpOfADC(mDataBuffer[row][index]) * dp * dp;
  where(selectionMask) | sigmaTime2 += getFpOfADC(mDataBuffer[row][index]) * dt * dt;

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
}
}


#endif

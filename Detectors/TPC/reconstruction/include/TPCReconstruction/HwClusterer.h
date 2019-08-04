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

namespace o2
{
namespace tpc
{

class Digit;
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

  /// Destructor
  ~HwClusterer() override = default;

  /// Copy Constructor
  HwClusterer(HwClusterer const& other) = default;

  /// Process digits
  /// \param digits Container with TPC digits
  /// \param mcDigitTruth MC Digit Truth container
  /// \param clearContainerFirst Clears the outpcontainer for clusters and MC labels first, before processing
  void process(std::vector<o2::tpc::Digit> const& digits, MCLabelContainer const* mcDigitTruth) override;
  void process(std::vector<o2::tpc::Digit> const& digits, MCLabelContainer const* mcDigitTruth, bool clearContainerFirst);

  /// Finish processing digits
  /// \param digits Container with TPC digits
  /// \param mcDigitTruth MC Digit Truth container
  /// \param clearContainerFirst Clears the outpcontainer for clusters and MC labels first, before processing
  void finishProcess(std::vector<o2::tpc::Digit> const& digits, MCLabelContainer const* mcDigitTruth) override;
  void finishProcess(std::vector<o2::tpc::Digit> const& digits, MCLabelContainer const* mcDigitTruth, bool clearContainerFirst);

  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  void setContinuousReadout(bool isContinuous);

  /// Sets the charge threshold for the peak
  /// \param charge Threshold which will be used
  void setPeakChargeThreshold(unsigned charge);

  /// Sets the charge threshold for the contributing pads
  /// \param charge Threshold which will be used
  void setContributionChargeThreshold(unsigned charge);

  /// Switch to reject single pad clusters
  /// \param doReject - true to reject clusters with sigmaPad2Pre == 0
  void setRejectSinglePadClusters(bool doReject);

  /// Switch to reject single time clusters
  /// \param doReject - true to reject clusters with sigmaTime2Pre == 0
  void setRejectSingleTimeClusters(bool doReject);

  /// Switch to reject peaks in following timebins
  /// \param doReject - true to reject peak, false to NOT reject peaks in following bins
  void setRejectLaterTimebin(bool doReject);

  /// Switch for mode, how the charge should be shared among nearby clusters
  /// \param mode   0 for no splitting, charge is used for all found peaks,
  ///               1 for minimum contributes half to all peaks
  ///               2 for minimum contributes only to left/older peak
  void setSplittingMode(short mode);

 private:
  /*
   * Helper functions
   */

  /// HW Cluster Processor
  /// \param peakMask       VC-mask with only peaks enabled
  /// \param qMaxIndex      Buffer index of center pad
  /// \param centerPad     Pad number to be checked for cluster
  /// \param centerTime    Time to be checked for cluster
  /// \param row            Row number for cluster properties
  void hwClusterProcessor(Vc::uint_m peakMask, unsigned qMaxIndex, short centerPad, int centerTime, unsigned short row);

  /// HW Peak Finder
  /// \param qMaxIndex        Buffer index of central pad
  /// \param centerPad        Pad number to be checked for cluster
  /// \param mappedCenterTime Time to be checked for cluster, mapped in available time space
  /// \param row              Row number for cluster properties
  void hwPeakFinder(unsigned qMaxIndex, short centerPad, int mappedCenterTime, unsigned short row);

  /// Helper function to update cluster properties and MC labels
  /// \param selectionMask  VC-mask with slected pads enabled
  /// \param row            Current row
  /// \param centerPad     Pad of peak
  /// \param centerTime    Timebin of peak
  /// \param dp             delta pad
  /// \param dt             delta time
  /// \param qTot           Total charge
  /// \param pad            Weighted pad parameter
  /// \param time           Weighted time parameter
  /// \param sigmaPad2      Weighted sigma pad ^2 parameter
  /// \param sigmaTime2     Weighted sigma time ^2 parameter
  /// \param mcLabel        Vector with MClabel-counter-pairs
  void updateCluster(const Vc::uint_m selectionMask, int row, short centerPad, int centerTime, short dp, short dt, Vc::uint_v& qTot, Vc::int_v& pad, Vc::int_v& time, Vc::int_v& sigmaPad2, Vc::int_v& sigmaTime2, std::vector<std::unique_ptr<std::vector<std::pair<MCCompLabel, unsigned>>>>& mcLabels, Vc::uint_m splitMask = Vc::Mask<uint>(false));

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

  /// Clears the buffer at given timebin
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

  /*
   * class members
   */
  static const int mTimebinsInBuffer = 6;

  unsigned short mNumRows;               ///< Number of rows in this sector
  unsigned short mNumRowSets;            ///< Number of row sets (Number of rows / Vc::Size) in this sector
  short mCurrentMcContainerInBuffer;     ///< Bit field, where to find the current MC container in buffer
  short mSplittingMode;                  ///< Cluster splitting mode, 0 no splitting, 1 for minimum contributes half to both, 2 for miminum corresponds to left/older cluster
  int mClusterSector;                    ///< Sector to be processed
  int mLastTimebin;                      ///< Last time bin of previous event
  unsigned mLastHB;                      ///< Last HB bin of previous event
  unsigned mPeakChargeThreshold;         ///< Charge threshold for the central peak in ADC counts
  unsigned mContributionChargeThreshold; ///< Charge threshold for the contributing pads in ADC counts
  unsigned mClusterCounter;              ///< Cluster counter in output container for MC truth matching
  bool mIsContinuousReadout;             ///< Switch for continuous readout
  bool mRejectSinglePadClusters;         ///< Switch to reject single pad clusters, sigmaPad2Pre == 0
  bool mRejectSingleTimeClusters;        ///< Switch to reject single time clusters, sigmaTime2Pre == 0
  bool mRejectLaterTimebin;              ///< Switch to reject peaks in later timebins of the same pad

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
  MCLabelContainer* mClusterMcLabelArray;                  ///< Pointer to MC Label container
};

inline void HwClusterer::process(std::vector<o2::tpc::Digit> const& digits, o2::dataformats::MCTruthContainer<o2::MCCompLabel> const* mcDigitTruth)
{
  process(digits, mcDigitTruth, true);
}

inline void HwClusterer::finishProcess(std::vector<o2::tpc::Digit> const& digits, o2::dataformats::MCTruthContainer<o2::MCCompLabel> const* mcDigitTruth)
{
  finishProcess(digits, mcDigitTruth, true);
}

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

inline void HwClusterer::setRejectSinglePadClusters(bool doReject)
{
  mRejectSinglePadClusters = doReject;
}

inline void HwClusterer::setRejectSingleTimeClusters(bool doReject)
{
  mRejectSingleTimeClusters = doReject;
}

inline void HwClusterer::setRejectLaterTimebin(bool doReject)
{
  mRejectLaterTimebin = doReject;
}

inline void HwClusterer::setSplittingMode(short mode)
{
  mSplittingMode = mode;
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

} // namespace tpc
} // namespace o2

#endif

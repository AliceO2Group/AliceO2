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

  /// Swtich for requiring charge in at least one neighbouring pad
  /// \param reqPad - false for single pad clusters, true otherwise
  void setRequireNeighbouringPad(bool reqPad);

  /// Swtich for requiring charge in at least one neighbouring timebin
  /// \param reqPad - false for single timebin clusters, true otherwise
  void setRequireNeighbouringTimebin(bool reqTimebin);

 private:
  /*
   * Helper functions
   */

  /// HW Cluster Finder
  /// \param center_pad       Pad number to be checked for cluster
  /// \param center_time      Time to be checked for cluster
  /// \param row              Row number for cluster properties
  /// \param cluster          Reference to pair of a cluster and vector with MClabel-counter-pair to store found clusters in
  /// \return True if (center_pad,center_time) was a cluster, false if not
  bool hwClusterFinder(unsigned qMaxIndex, short center_pad, int center_time, unsigned short row, ClusterHardware& cluster, std::vector<std::pair<MCCompLabel, unsigned>>& mcLabels);

  /// Helper function to update cluster properties and MC labels
  /// \param row          Current row
  /// \param center_pad   Pad of peak
  /// \param center_time  Timebin of peak
  /// \param dp           delta pad
  /// \param dt           delta time
  /// \param qTot         Total charge
  /// \param pad          Weighted pad parameter
  /// \param time         Weighted time parameter
  /// \param sigmaPad2    Weighted sigma pad ^2 parameter
  /// \param sigmaTime2   Weighted sigma time ^2 parameter
  /// \param mcLabel      Vector with MClabel-counter-pair
  void updateCluster(int row, short center_pad, int center_time, short dp, short dt, unsigned& qTot, int& pad, int& time, int& sigmaPad2, int& sigmaTime2, std::vector<std::pair<MCCompLabel, unsigned>>& mcLabels);

  /// Writes clusters in temporary storage to cluster output
  /// \param timeOffset   Time offset of cluster container
  void writeOutputWithTimeOffset(int timeOffset);

  /// Does the Cluster Finding in all rows for given timebin
  /// \param timebin  Timebin to cluster peaks
  void findClusterForTime(int timebin);

  /// Searches for last remaining cluster and writes them out
  /// \param clear    Clears data buffer afterwards (for not continuous readout)
  void finishFrame(bool clear = false);

  /// Clears the buffer at given timebin TODO: and fills timebin with noise + pedestal
  /// \param timebin  Timebin to be cleared
  void clearBuffer(int timebin);

  /// Returns least significant set bit of mCurrentMcContainerInBuffer, only the 5 LSBs are checked
  /// \return LSB index which is set
  short getFirstSetBitOfField();

  /*
   * class members
   */
  unsigned short mNumRows;               ///< Number of rows in this sector
  short mCurrentMcContainerInBuffer;     ///< Bit field, where to find the current MC container in buffer
  int mClusterSector;                    ///< Sector to be processed
  int mLastTimebin;                      ///< Last time bin of previous event
  unsigned mLastHB;                      ///< Last HB bin of previous event
  unsigned mPeakChargeThreshold;         ///< Charge threshold for the central peak in ADC counts
  unsigned mContributionChargeThreshold; ///< Charge threshold for the contributing pads in ADC counts
  unsigned mClusterCounter;              ///< Cluster counter in output container for MC truth matching
  bool mRequireNeighbouringTimebin;      ///< Switch to disable single time cluster
  bool mRequireNeighbouringPad;          ///< Switch to disable single pad cluster
  bool mIsContinuousReadout;             ///< Switch for continuous readout

  std::vector<unsigned short> mPadsPerRow;                       ///< Number of pads for given row (offset of 2 pads on both sides is already added)
  std::vector<unsigned short> mGlobalRowToRegion;                ///< Mapping global row number to region
  std::vector<unsigned short> mGlobalRowToLocalRow;              ///< Converting global row number to local row number within region
  std::vector<std::vector<unsigned>> mDataBuffer;                ///< Buffer with digits (+noise +CM +...)
  std::vector<std::vector<int>> mIndexBuffer;                    ///< Buffer with digits indices for MC labels
  std::vector<std::shared_ptr<MCLabelContainer const>> mMCtruth; ///< MC truth information of timebins in buffer

  std::vector<std::unique_ptr<std::vector<std::pair<ClusterHardware, std::vector<std::pair<MCCompLabel, unsigned>>>>>> mTmpClusterArray; ///< Temporary cluster storage for each region to accumulate cluster before filling output container

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

inline void HwClusterer::setRequireNeighbouringPad(bool reqPad)
{
  mRequireNeighbouringPad = reqPad;
}

inline void HwClusterer::setRequireNeighbouringTimebin(bool reqTimebin)
{
  mRequireNeighbouringTimebin = reqTimebin;
}

inline short HwClusterer::getFirstSetBitOfField()
{
  for (short i = 0; i < 5; ++i) {
    if ((mCurrentMcContainerInBuffer >> i) & 0x1)
      return i;
  }
  return -1;
}
}
}


#endif

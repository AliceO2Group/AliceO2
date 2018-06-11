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
class ClusterHardware;

/// \class HwClusterer
/// \brief Class for TPC HW cluster finding
class HwClusterer : public Clusterer
{

  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

  public:

    /// Constructor
    /// \param clusterOutput is pointer to vector to be filled with clusters
    /// \param labelOutput is pointer to storage to be filled with MC labels
   /// \param sectorid is sector number to be processed
   HwClusterer(
     std::shared_ptr<std::vector<ClusterHardwareContainer8kb>> clusterOutput,
     std::shared_ptr<MCLabelContainer> labelOutput = nullptr,
     int sectorid = -1);

   /// Destructor
   ~HwClusterer() = default;

   /// Process digits
   /// @param digits Container with TPC digits
   /// @param mcDigitTruth MC Digit Truth container
   /// @param eventCount event counter
   /// @return Container with clusters
   void Process(std::vector<o2::TPC::Digit> const& digits, MCLabelContainer const& mcDigitTruth, int eventCount) override;

   /// Finish processing digits
   /// @param digits Container with TPC digits
   /// @param mcDigitTruth MC Digit Truth container
   /// @param eventCount event counter
   /// @return Container with clusters
   void FinishProcess(std::vector<o2::TPC::Digit> const& digits, MCLabelContainer const& mcDigitTruth, int eventCount) override;

   /// Switch for triggered / continuous readout
   /// \param isContinuous - false for triggered readout, true for continuous readout
   void setContinuousReadout(bool isContinuous);

  private:
    /*
     * Helper functions
     */

   /// HW Cluster Finder
   /// \param center_pad       Pad number to be checked for cluster
   /// \param center_time      Time to be checked for cluster
   /// \param row              Row number for cluster properties
   /// \param cluster          Field to store found cluster in
   /// \param sortedMcLabels   Sorted vector with MClabel-counter-pair
   /// \return True if (center_pad,center_time) was a cluster, false if not
   bool hwClusterFinder(unsigned short center_pad, unsigned center_time, unsigned short row, ClusterHardware& cluster, std::vector<std::pair<MCCompLabel, unsigned>>& sortedMcLabels);

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
   void updateCluster(int row, unsigned short center_pad, unsigned center_time, short dp, short dt, unsigned& qTot, int& pad, int& time, int& sigmaPad2, int& sigmaTime2, std::vector<std::pair<MCCompLabel, unsigned>>& mcLabels);

   /// Writes clusters in temporary storage to cluster output
   /// \param timeOffset   Time offset of cluster container
   void writeOutputForTimeOffset(unsigned timeOffset);

   /// Does the Cluster Finding in all rows for given timebin
   /// \param timebin  Timebin to cluster peaks
   void findClusterForTime(unsigned timebin);

   /// Searches for last remaining cluster and writes them out
   /// \param clear    Clears data buffer afterwards (for not continuous readout)
   void finishFrame(bool clear = false);

   /// Clears the buffer at given timebin TODO: and fills timebin with noise + pedestal
   /// \param timebin  Timebin to be cleared
   void clearBuffer(unsigned timebin);

   /*
     * class members
     */
   int mClusterSector;                    ///< Sector to be processed
   unsigned short mNumRows;               ///< Number of rows in this sector
   int mLastTimebin;                      ///< Last time bin of previous event
   unsigned mLastHB;                      ///< Last HB bin of previous event
   unsigned mPeakChargeThreshold;         ///< Charge threshold for the central peak in ADC counts
   unsigned mContributionChargeThreshold; ///< Charge threshold for the contributing pads in ADC counts
   bool mRequireNeighbouringTimebin;      ///< Switch to disable single time cluster
   bool mRequireNeighbouringPad;          ///< Switch to disable single pad cluster
   bool mIsContinuousReadout;             ///< Switch for continuous readout

   std::vector<unsigned short> mPadsPerRow;                       ///< Number of pads for given row (offset of 2 pads on both sides is already added)
   std::vector<unsigned short> mGlobalRowToRegion;                ///< Mapping global row number to region
   std::vector<unsigned short> mGlobalRowToLocalRow;              ///< Converting global row number to local row number within region
   std::vector<std::vector<unsigned>> mDataBuffer;                ///< Buffer with digits (+noise +CM +...)
   std::vector<std::vector<int>> mIndexBuffer;                    ///< Buffer with digits indices for MC labels
   std::vector<std::unique_ptr<MCLabelContainer const>> mMCtruth; ///< MC truth information of timebins in buffer
   std::vector<std::pair<MCCompLabel, int>> mMClabel;             ///< Vector to accumulate the MC labels

   std::vector<std::unique_ptr<std::vector<std::pair<std::shared_ptr<ClusterHardware>, std::unique_ptr<std::vector<std::pair<MCCompLabel, unsigned>>>>>>> mTmpClusterArray; ///< Temporary cluster storage for each region to accumulate cluster before filling output container

   std::shared_ptr<std::vector<ClusterHardwareContainer8kb>> mClusterArray; ///< Pointer to output cluster container
   std::shared_ptr<MCLabelContainer> mClusterMcLabelArray;                  ///< Pointer to MC Label container
};

inline void HwClusterer::setContinuousReadout(bool isContinuous)
{
  mIsContinuousReadout = isContinuous;
}
}
}


#endif

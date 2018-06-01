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
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_HWClusterer_H_
#define ALICEO2_TPC_HWClusterer_H_

#include "TPCReconstruction/Clusterer.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "TPCBase/CalDet.h"

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <vector>
#include <map>
#include <utility>
#include <tuple>
#include <memory>

namespace o2{
namespace TPC {

class HwClusterFinder;
class Digit;

/// \class HwClusterer
/// \brief Class for TPC HW cluster finding
class HwClusterer {

  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

  public:

    /// Constructor
    /// \param clusterOutput is pointer to vector to be filled with clusters
    /// \param labelOutput is pointer to storage to be filled with MC labels
    /// \param cru Number of CRUs to process
    /// \param minQDiff Min charge difference
    /// \param assignChargeUnique Avoid using same charge for multiple nearby clusters
    /// \param enableNoiseSim Enables the Noise simulation for empty pads (noise object has to be set)
    /// \param enablePedestalSubtraction Enables the Pedestal subtraction (pedestal object has to be set)
    /// \param padsPerCF Pads per cluster finder
    /// \param timebinsPerCF Time bins per cluster finder
    /// \param rowsMax Max number of rows to process
    /// \param padsMax Max number of pads to process
    /// \param timeBinsMax Max number of timebins to process
    /// \param minQMax Minimum peak charge for cluster
    /// \param requirePositiveCharge Positive charge is required
    /// \param requireNeighbouringPad Requires at least 2 adjecent pads with charge above threshold
    HwClusterer(
        std::vector<ClusterHardwareContainer8kb>* clusterOutput,
        MCLabelContainer *labelOutput = nullptr,
        int sectorid = -1);
//        int cruMin = 0,
//        int cruMax = 359,
//        float minQDiff = 0,
//        bool assignChargeUnique = true,//false,
//        bool enableNoiseSim = true,
//        bool enablePedestalSubtraction = true,
//        int padsPerCF = 8,
//        int timebinsPerCF = 8,
//        int rowsMax = 18,
//        int padsMax = 138,
//        int timeBinsMax = 1024,
//        int minQMax = 5,
//        bool requirePositiveCharge = true,
//        bool requireNeighbouringPad = true);

    /// Destructor
    ~HwClusterer() = default;

    /// Process digits
    /// @param digits Container with TPC digits
    /// @param mcDigitTruth MC Digit Truth container
    /// @param eventCount event counter
    /// @return Container with clusters
    void Process(std::vector<o2::TPC::Digit> const *digits, MCLabelContainer const* mcDigitTruth, int eventCount);

    /// Finish processing digits
    /// @param digits Container with TPC digits
    /// @param mcDigitTruth MC Digit Truth container
    /// @param eventCount event counter
    /// @return Container with clusters
    void FinishProcess(std::vector<o2::TPC::Digit> const *digits, MCLabelContainer const* mcDigitTruth, int eventCount);

//    /// Setter for noise object, noise will be added before cluster finding
//    /// \param noiseObject CalDet object, containing noise simulation
//    void setNoiseObject(std::shared_ptr<CalDet<float>> noiseObject) { mNoiseObject = noiseObject; };
//
//    /// Setter for pedestal object, pedestal value will be subtracted before cluster finding
//    /// \param pedestalObject CalDet object, containing pedestals for each pad
//    void setPedestalObject(std::shared_ptr<CalDet<float>> pedestalObject) { mPedestalObject = pedestalObject; };
//    void setPedestalObject(CalDet<float>* pedestalObject) {
//      LOG(DEBUG) << "Consider using std::shared_ptr for the pedestal object." << FairLogger::endl;
//      mPedestalObject = std::shared_ptr<CalDet<float>>(pedestalObject);
//    };

    /// Switch for triggered / continuous readout
    /// \param isContinuous - false for triggered readout, true for continuous readout
    void setContinuousReadout(bool isContinuous);

//    /// Setters for CRU range to be processed
//    /// \param cru ID of min/max CRU to be processed
//    void setCRUMin(int cru) { mCRUMin = cru; };
//    void setCRUMax(int cru) { mCRUMax = cru; };

  private:


    /*
     * Helper functions
     */

    /// HW Cluster Finder
    /// \param center_pad   Pad number to be checked for cluster
    /// \param center_time  Time to be checked for cluster
    /// \param row          Row number for cluster properties
    /// \param cluster      Field to store found cluster in
    /// \return True if (center_pad,center_time) was a cluster, false if not
    bool hwClusterFinder(unsigned short center_pad, unsigned center_time, unsigned short row, ClusterHardware& cluster);

    /// Helper function to update cluster properties
    /// \param charge       Charge
    /// \param dp           delta pad
    /// \param dt           delta time
    /// \param qTot         Total charge
    /// \param pad          Weighted pad parameter
    /// \param time         Weighted time parameter
    /// \param sigmaPad2    Weighted sigma pad ^2 parameter
    /// \param sigmaTime2   Weighted sigma time ^2 parameter
    void updateCluster(unsigned charge, short dp, short dt, unsigned& qTot, int& pad, int& time, int& sigmaPad2, int&sigmaTime2);

    /// Writes clusters in temorary storage to cluster output
    /// \param timeOffset   Time offset of cluster container
    void writeOutputForTimeOffset(unsigned timeOffset);

    /// Does the Cluster Finding in all rows for given timebin
    /// \param timebin  Timebin to cluster peaks
    void findClusterForTime(unsigned timebin);

    /// Searches for last remaining cluster and writes them out
    /// \param clear    Clears data buffer afterwards (for not continous readout)
    void finishFrame(bool clear = false);

    /// Clears the buffer at given timebin
    void clearBuffer(unsigned timebin);

    /*
     * class members
     */
    int mClusterSector;                     ///< Sector to be processed
    unsigned short mNumRows;                ///< Number of rows in this sector
    int mLastTimebin;                       ///< Last time bin of previous event
    unsigned mLastHB;                       ///< Last HB bin of previous event
    unsigned mPeakChargeThreshold;          ///< Charge threshold for the central peak in ADC counts
    unsigned mContributionChargeThreshold;  ///< Charge threshold for the contributing pads in ADC counts
    bool mRequireNeighbouringTimebin;       ///< Switch to disable single time cluster
    bool mRequireNeighbouringPad;           ///< Switch to disable single pad cluster
    bool mIsContinuousReadout;              ///< Switch for continuous readout

    std::vector<unsigned short> mPadsPerRow;
    std::vector<unsigned short> mGlobalRowToRegion;
    std::vector<unsigned short> mGlobalRowToLocalRow;
    std::vector<std::vector<unsigned>> mDataBuffer;

    std::vector<std::unique_ptr<std::vector<ClusterHardware>>> mTmpClusterArray;

    std::vector<ClusterHardwareContainer8kb>* mClusterArray;        ///< Pointer to output cluster storage
    MCLabelContainer* mClusterMcLabelArray;     ///< Pointer to MC Label storage






//    /// Configuration struct for the processDigits function
//    struct CfConfig {
//      unsigned iThreadID;               ///< Index of thread
//      unsigned iThreadMax;              ///< Total number of started threads
//      unsigned iCRUMin;                 ///< Minimum CRU number to process
//      unsigned iCRUMax;                 ///< Maximum CRU number to process
//      unsigned iMaxPads;                ///< Maximum number of pads per row
//      int iMinTimeBin;                  ///< Minumum digit time bin
//      int iMaxTimeBin;                  ///< Maximum digit time bin
//      bool iEnableNoiseSim;             ///< Noise simulation enable switch
//      bool iEnablePedestalSubtraction;  ///< Pedestal subtraction enable switch
//      bool iIsContinuousReadout;        ///< Continous simulation switch
//      std::shared_ptr<CalDet<float>> iNoiseObject;      ///< Pointer to noise object
//      std::shared_ptr<CalDet<float>> iPedestalObject;   ///< Pointer to pedestal object
//    };
//
//    /// Processing the digits, made static to allow for multithreading
//    /// \param digits Reference to digit container
//    /// \param clusterFinder Reference to container holding all cluster finder instances
//    /// \param cluster Reference to container for found clusters
//    /// \param label Reference to container for MC labels of found clusters
//    /// \param config Configuration for the cluster finding
//    static void processDigits(
//        const std::vector<std::vector<std::vector<std::tuple<Digit const*, int, int>>>>& digits,
//        const std::vector<std::vector<std::vector<std::shared_ptr<HwClusterFinder>>>>& clusterFinder,
//              std::vector<std::vector<Cluster>>& cluster,
//              std::vector<std::vector<std::vector<std::pair<int,int>>>>& label,
//              CfConfig config);
//
//    /// Handling of the parallel cluster finder threads
//    /// \param iTimeBinMin Minimum time bin to be processed
//    /// \param iTimeBinMax Maximum time bin to be processed
//    void ProcessTimeBins(int iTimeBinMin, int iTimeBinMax);
//
//
//
//
//    bool mAssignChargeUnique;               ///< Setting for CF to use charge only for one cluster
//    bool mEnableNoiseSim;                   ///< Switch for noise simulation
//    bool mEnablePedestalSubtraction;        ///< Switch for pedestal subtraction
//    unsigned mCRUMin;                       ///< Minimum CRU ID to be processed
//    unsigned mCRUMax;                       ///< Maximum CRU ID to be processed
//    unsigned mPadsPerCF;                    ///< Number of pads per cluster finder instance
//    unsigned mTimebinsPerCF;                ///< Number of time bins per cluster finder instance
    unsigned mNumThreads;                   ///< Number of parallel processing threads
//    float mMinQDiff;                        ///< Minimum charge difference between neighboring pads / time bins
//
//    std::vector<std::vector<std::vector<std::shared_ptr<HwClusterFinder>>>> mClusterFinder;     ///< Cluster finder container for each row in each CRU
//    std::vector<std::vector<std::vector<std::tuple<Digit const*, int, int>>>> mDigitContainer;  ///< Sorted digit container for each row in each CRU. Tuple consists of pointer to digit, original digit index and event count
//
//    std::vector<std::vector<Cluster>> mClusterStorage;                                          ///< Cluster storage for each CRU
//    std::vector<std::vector<std::vector<std::pair<int,int>>>> mClusterDigitIndexStorage;        ///< Container for digit indices, used in found clusters. Pair consists of original digit index and event count
//
//
//    std::shared_ptr<CalDet<float>> mNoiseObject;                ///< Pointer to the CalDet object for noise simulation
//    std::shared_ptr<CalDet<float>> mPedestalObject;             ///< Pointer to the CalDet object for the pedestal subtraction
//
//    std::map<int, std::unique_ptr<MCLabelContainer>> mLastMcDigitTruth; ///< Buffer for digit MC truth information
};

inline void HwClusterer::updateCluster(unsigned charge, short dp, short dt, unsigned& qTot, int& pad, int& time, int& sigmaPad2, int&sigmaTime2) {
  qTot          += charge;
  pad           += charge * dp;
  time          += charge * dt;
  sigmaPad2     += charge * dp * dp;
  sigmaTime2    += charge * dt * dt;
}

inline void HwClusterer::setContinuousReadout(bool isContinuous) {
  mIsContinuousReadout = isContinuous;
}

}
}


#endif

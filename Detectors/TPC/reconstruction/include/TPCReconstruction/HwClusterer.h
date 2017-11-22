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
#include "TPCReconstruction/Cluster.h"
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

class ClustererTask;
class HwClusterFinder;
class Digit;

/// \class HwClusterer
/// \brief Class for TPC HW cluster finding
class HwClusterer : public Clusterer {

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
        std::vector<o2::TPC::Cluster> *clusterOutput,
        MCLabelContainer *labelOutput = nullptr,
        int cruMin = 0,
        int cruMax = 359,
        float minQDiff = 0,
        bool assignChargeUnique = true,//false,
        bool enableNoiseSim = true,
        bool enablePedestalSubtraction = true,
        int padsPerCF = 8,
        int timebinsPerCF = 8,
        int rowsMax = 18,
        int padsMax = 138,
        int timeBinsMax = 1024,
        int minQMax = 5,
        bool requirePositiveCharge = true,
        bool requireNeighbouringPad = true);

    /// Destructor
    ~HwClusterer() override;

    /// Steer conversion of points to digits
    /// @param digits Container with TPC digits
    /// @param mcDigitTruth MC Digit Truth container
    /// @param eventCount event counter
    /// @return Container with clusters
    void Process(std::vector<o2::TPC::Digit> const &digits,
        MCLabelContainer const* mcDigitTruth, int eventCount) override;
    void Process(std::vector<std::unique_ptr<Digit>>& digits,
        MCLabelContainer const* mcDigitTruth, int eventCount) override;

    /// Setter for noise object, noise will be added before cluster finding
    /// \param noiseObject CalDet object, containing noise simulation
    void setNoiseObject(std::shared_ptr<CalDet<float>> noiseObject) { mNoiseObject = noiseObject; };

    /// Setter for pedestal object, pedestal value will be subtracted before cluster finding
    /// \param pedestalObject CalDet object, containing pedestals for each pad
    void setPedestalObject(std::shared_ptr<CalDet<float>> pedestalObject) { mPedestalObject = pedestalObject; };
    void setPedestalObject(CalDet<float>* pedestalObject) { 
      LOG(DEBUG) << "Consider using std::shared_ptr for the pedestal object." << FairLogger::endl; 
      mPedestalObject = std::shared_ptr<CalDet<float>>(pedestalObject); 
    };

    /// Switch for triggered / continuous readout
    /// \param isContinuous - false for triggered readout, true for continuous readout
    void setContinuousReadout(bool isContinuous) { mIsContinuousReadout = isContinuous; };

    /// Setters for CRU range to be processed
    /// \param cru ID of min/max CRU to be processed
    void setCRUMin(int cru) { mCRUMin = cru; };
    void setCRUMax(int cru) { mCRUMax = cru; };

    /// Set number of parallel threads
    /// \param threads Number to be set, if 0 hardware default value is used
    void setNumThreads(unsigned threads);

  private:

    /*
     * Helper functions
     */

    /// Configuration struct for the processDigits function
    struct CfConfig {
      unsigned iThreadID;               ///< Index of thread
      unsigned iThreadMax;              ///< Total number of started threads
      unsigned iCRUMin;                 ///< Minimum CRU number to process
      unsigned iCRUMax;                 ///< Maximum CRU number to process
      unsigned iMaxPads;                ///< Maximum number of pads per row
      int iMinTimeBin;                  ///< Minumum digit time bin
      int iMaxTimeBin;                  ///< Maximum digit time bin
      bool iEnableNoiseSim;             ///< Noise simulation enable switch
      bool iEnablePedestalSubtraction;  ///< Pedestal subtraction enable switch
      bool iIsContinuousReadout;        ///< Continous simulation switch
      std::shared_ptr<CalDet<float>> iNoiseObject;      ///< Pointer to noise object
      std::shared_ptr<CalDet<float>> iPedestalObject;   ///< Pointer to pedestal object
    };

    /// Processing the digits, made static to allow for multithreading
    /// \param digits Reference to digit container
    /// \param clusterFinder Reference to container holding all cluster finder instances
    /// \param cluster Reference to container for found clusters
    /// \param label Reference to container for MC labels of found clusters
    /// \param config Configuration for the cluster finding
    static void processDigits(
        const std::vector<std::vector<std::vector<std::tuple<Digit const*, int, int>>>>& digits,
        const std::vector<std::vector<std::vector<std::shared_ptr<HwClusterFinder>>>>& clusterFinder,
              std::vector<std::vector<Cluster>>& cluster,
              std::vector<std::vector<std::vector<std::pair<int,int>>>>& label,
              CfConfig config);

    /// Handling of the parallel cluster finder threads
    /// \param iTimeBinMin Minimum time bin to be processed
    /// \param iTimeBinMax Maximum time bin to be processed
    void ProcessTimeBins(int iTimeBinMin, int iTimeBinMax);

    /*
     * class members
     */
    bool mAssignChargeUnique;               ///< Setting for CF to use charge only for one cluster
    bool mEnableNoiseSim;                   ///< Switch for noise simulation
    bool mEnablePedestalSubtraction;        ///< Switch for pedestal subtraction
    bool mIsContinuousReadout;              ///< Switch for continuous readout
    int mLastTimebin;                       ///< Last time bin of previous event
    unsigned mCRUMin;                       ///< Minimum CRU ID to be processed
    unsigned mCRUMax;                       ///< Maximum CRU ID to be processed
    unsigned mPadsPerCF;                    ///< Number of pads per cluster finder instance
    unsigned mTimebinsPerCF;                ///< Number of time bins per cluster finder instance
    unsigned mNumThreads;                   ///< Number of parallel processing threads
    float mMinQDiff;                        ///< Minimum charge difference between neighboring pads / time bins

    std::vector<std::vector<std::vector<std::shared_ptr<HwClusterFinder>>>> mClusterFinder;     ///< Cluster finder container for each row in each CRU
    std::vector<std::vector<std::vector<std::tuple<Digit const*, int, int>>>> mDigitContainer;  ///< Sorted digit container for each row in each CRU. Tuple consists of pointer to digit, original digit index and event count

    std::vector<std::vector<Cluster>> mClusterStorage;                                          ///< Cluster storage for each CRU
    std::vector<std::vector<std::vector<std::pair<int,int>>>> mClusterDigitIndexStorage;        ///< Container for digit indices, used in found clusters. Pair consists of original digit index and event count

    std::vector<Cluster>* mClusterArray;        ///< Pointer to output cluster storage
    MCLabelContainer* mClusterMcLabelArray;     ///< Pointer to MC Label storage

    std::shared_ptr<CalDet<float>> mNoiseObject;                ///< Pointer to the CalDet object for noise simulation
    std::shared_ptr<CalDet<float>> mPedestalObject;             ///< Pointer to the CalDet object for the pedestal subtraction

    std::map<int, std::unique_ptr<MCLabelContainer>> mLastMcDigitTruth; ///< Buffer for digit MC truth information
  };
}
}


#endif

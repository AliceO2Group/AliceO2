// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  ClustererTask.h
//  ALICEO2
//
//
//

#ifndef __ALICEO2__ClustererTask__
#define __ALICEO2__ClustererTask__

#include "FairTask.h"  // for FairTask, InitStatus
#include "Rtypes.h"    // for ClustererTask::Class, ClassDef, etc
#include "TPCReconstruction/Clusterer.h"        // for Clusterer
#include "TPCReconstruction/BoxClusterer.h"     // for Clusterer
#include "TPCReconstruction/HwClusterer.h"      // for Clusterer
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <vector>
#include <memory>

namespace o2 {
namespace TPC{

class ClustererTask : public FairTask{

  using MCLabelContainer = o2::dataformats::MCTruthContainer<o2::MCCompLabel>;

  public:
    ClustererTask();
    ~ClustererTask() override;

    InitStatus Init() override;
    void Exec(Option_t *option) override;

    enum class ClustererType : int { HW, Box};

    /// Switch to enable individual clusterer
    /// \param type - Clusterer type, HW or Box
    /// \param val - Enable set to true or false
    void setClustererEnable(ClustererType type, bool val) {
      switch (type) {
        case ClustererType::HW:   mHwClustererEnable = val; break;
        case ClustererType::Box:  mBoxClustererEnable = val; break;
      };
    };

    /// Returns status of Cluster enable
    /// \param type - Clusterer type, HW or Box
    /// \return Enable status 
    bool isClustererEnable(ClustererType type) const {
      switch (type) {
        case ClustererType::HW:   return mHwClustererEnable;
        case ClustererType::Box:  return mBoxClustererEnable;
      };
    };

    /// Returns pointer to requested Clusterer type
    /// \param type - Clusterer type, HW or Box
    /// \return  Pointer to Clusterer, nullptr if Clusterer was not enabled during Init()
    Clusterer* getClusterer(ClustererType type) {
      switch (type) {
        case ClustererType::HW:   return mHwClusterer.get();
        case ClustererType::Box:  return mBoxClusterer.get();
      };
    };

    /// Returns pointer to Box Clusterer
    /// \return  Pointer to Clusterer, nullptr if Clusterer was not enabled during Init()
    BoxClusterer* getBoxClusterer()   const { return mBoxClusterer.get(); };
    
    /// Returns pointer to Hw Clusterer
    /// \return  Pointer to Clusterer, nullptr if Clusterer was not enabled during Init()
    HwClusterer* getHwClusterer()     const { return mHwClusterer.get(); };

  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  void setContinuousReadout(bool isContinuous);

  private:
    bool mBoxClustererEnable;   ///< Switch to enable Box Clusterfinder
    bool mHwClustererEnable;    ///< Switch to enable Hw Clusterfinder
    bool mIsContinuousReadout;  ///< Switch for continuous readout
    int mEventCount;            ///< Event counter

    std::unique_ptr<BoxClusterer> mBoxClusterer;    ///< Box Clusterfinder instance
    std::unique_ptr<HwClusterer> mHwClusterer;      ///< Hw Clusterfinder instance

    // Digit arrays
    std::vector<o2::TPC::Digit> const *mDigitsArray;    ///< Array of TPC digits
    MCLabelContainer const *mDigitMCTruthArray;         ///< Array for MCTruth information associated to digits in mDigitsArrray

    // Cluster arrays
    std::vector<o2::TPC::Cluster> *mClustersArray;              ///< Array of clusters found by Box Clusterfinder
    std::vector<o2::TPC::Cluster> *mHwClustersArray;            ///< Array of clusters found by Hw Clusterfinder
    std::unique_ptr<MCLabelContainer> mClustersMCTruthArray;      ///< Array for MCTruth information associated to cluster in mClustersArrays
    std::unique_ptr<MCLabelContainer> mHwClustersMCTruthArray;    ///< Array for MCTruth information associated to cluster in mHwClustersArrays

    ClassDefOverride(ClustererTask, 1)
};

inline
void ClustererTask::setContinuousReadout(bool isContinuous)
{
  mIsContinuousReadout = isContinuous;
}

}
}

#endif

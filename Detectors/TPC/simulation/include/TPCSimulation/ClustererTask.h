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

#include <cstdio>
#include "FairTask.h"  // for FairTask, InitStatus
#include "Rtypes.h"    // for ClustererTask::Class, ClassDef, etc
#include "TPCSimulation/Clusterer.h"       // for Clusterer
#include "TPCSimulation/BoxClusterer.h"       // for Clusterer
#include "TPCSimulation/HwClusterer.h"       // for Clusterer
#include <vector>

namespace o2 {
namespace TPC{
  
class ClustererTask : public FairTask{
  public:
    ClustererTask();
    ~ClustererTask() override;
    
    InitStatus Init() override;
    void Exec(Option_t *option) override;

    enum class ClustererType : int { HW, Box};
    void setClustererEnable(ClustererType type, bool val) {
      switch (type) {
        case ClustererType::HW:   mHwClustererEnable = val; break;
        case ClustererType::Box:  mBoxClustererEnable = val; break;
      };
    };

    bool isClustererEnable(ClustererType type) const { 
      switch (type) {
        case ClustererType::HW:   return mHwClustererEnable;
        case ClustererType::Box:  return mBoxClustererEnable;
      };
    };

    Clusterer* getClusterer(ClustererType type) { 
      switch (type) {
        case ClustererType::HW:   return mHwClusterer;
        case ClustererType::Box:  return mBoxClusterer;
      };
    };
    
    BoxClusterer* getBoxClusterer()   const { return mBoxClusterer; };
    HwClusterer* getHwClusterer()     const { return mHwClusterer; };
    //             Clusterer *GetClusterer() const { return fClusterer; }
    
  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  void setContinuousReadout(bool isContinuous);

  private:
    bool          mBoxClustererEnable;
    bool          mHwClustererEnable;
    bool          mIsContinuousReadout; ///< Switch for continuous readout

    BoxClusterer        *mBoxClusterer;
    HwClusterer         *mHwClusterer;
    
    std::vector<o2::TPC::Digit> const  *mDigitsArray;
    // produced data containers
    std::vector<o2::TPC::BoxCluster>  *mClustersArray;
    std::vector<o2::TPC::HwCluster>  *mHwClustersArray;
    
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

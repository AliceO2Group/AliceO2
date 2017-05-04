/// \file HwClusterer.h
/// \brief Class for TPC HW cluster finding
/// \author Sebastian Klewin
#ifndef ALICEO2_TPC_HWClusterer_H_
#define ALICEO2_TPC_HWClusterer_H_

#include "TPCSimulation/Clusterer.h"
#include "TPCBase/CalDet.h" 

#include <vector>

class TClonesArray;

namespace o2{
namespace TPC {
    
class ClusterContainer;
class ClustererTask;
class HwClusterFinder;
class HwCluster;
class DigitMC;

/// \class HwClusterer
/// \brief Class for TPC HW cluster finding
class HwClusterer : public Clusterer {
  public:
    enum class Processing : int { Sequential, Parallel};

    /// Default Constructor
    HwClusterer();

    /// Constructor
    /// \param processingType parallel or sequential
    /// \param globalTime value of first timebin
    /// \param cru Number of CRUs to process
    /// \param minQDiff Min charge differece 
    /// \param assignChargeUnique Avoid using same charge for multiple nearby clusters
    /// \param enableNoiseSim Enables the Noise simulation for empty pads (noise object has to be set)
    /// \param enablePedestalSubtraction Enables the Pedestal subtraction (pedestal object has to be set)
    /// \param padsPerCF Pads per cluster finder
    /// \param timebinsPerCF Timebins per cluster finder
    /// \param cfPerRow Number of cluster finder in each row
    HwClusterer(Processing processingType, int globalTime, int cru, float minQDiff,
      bool assignChargeUnique, bool enableNoiseSim, bool enablePedestalSubtraction, int padsPerCF, int timebinsPerCF, int cfPerRow);
    
    /// Destructor
    ~HwClusterer();
    
    // Should this really be a public member?
    // Maybe better to just call by process
    void Init() override;
    
    /// Steer conversion of points to digits
    /// @param digits Container with TPC digits
    /// @return Container with clusters
    ClusterContainer* Process(TClonesArray *digits) override;

    void setProcessingType(Processing processing)    { mProcessingType = processing; };   

    void setNoiseObject(CalDet<float>* noiseObject) { mNoiseObject = noiseObject; };
    void setPedestalObject(CalDet<float>* pedestalObject) { mPedestalObject = pedestalObject; };

    /// Switch for triggered / continuous readout
    /// \param isContinuous - false for triggered readout, true for continuous readout
    void setContinuousReadout(bool isContinuous) { mIsContinuousReadout = isContinuous; };
    
  private:
    // To be done
    /* BoxClusterer(const BoxClusterer &); */
    /* BoxClusterer &operator=(const BoxClusterer &); */

    struct CfConfig {
      int iCRU;
      int iMaxRows;
      int iMaxPads;
      int iMinTimeBin;
      int iMaxTimeBin;
      bool iEnableNoiseSim;
      bool iEnablePedestalSubtraction;
      bool iIsContinuousReadout;
      CalDet<float>* iNoiseObject;
      CalDet<float>* iPedestalObject;
    };
    
    static void processDigits(
        const std::vector<std::vector<DigitMC*>>& digits, 
        const std::vector<std::vector<HwClusterFinder*>>& clusterFinder, 
              std::vector<HwCluster>& cluster, 
              CfConfig config);
//              int iCRU,
//              int maxRows,
//              int maxPads, 
//              unsigned minTimeBin,
//              unsigned maxTimeBin);
    
    std::vector<std::vector<std::vector<HwClusterFinder*>>> mClusterFinder;
    std::vector<std::vector<std::vector<DigitMC*>>> mDigitContainer;

    std::vector<std::vector<HwCluster>> mClusterStorage;
    
    Processing    mProcessingType; 

    int     mGlobalTime;
    int     mCRUs;
    float   mMinQDiff;
    bool    mAssignChargeUnique;
    bool    mEnableNoiseSim;
    bool    mEnablePedestalSubtraction;
    bool    mIsContinuousReadout; ///< Switch for continuous readout
    int     mPadsPerCF;
    int     mTimebinsPerCF;
    int     mCfPerRow;
    int     mLastTimebin;

    CalDet<float>* mNoiseObject;
    CalDet<float>* mPedestalObject;
  };
}
}


#endif 

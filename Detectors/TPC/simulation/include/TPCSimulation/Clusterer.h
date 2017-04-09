/// \file Clusterer.h
/// \brief Base class for TPC clusterer
/// \author Sebastian klewin
#ifndef ALICEO2_TPC_Clusterer_H_
#define ALICEO2_TPC_Clusterer_H_

#include "TPCSimulation/ClusterContainer.h"

class TClonesArray;

namespace o2{
namespace TPC {
    
/// \class Clusterer
/// \brief Base Class fot TPC clusterer
class Clusterer {
  public:

    /// Default Constructor
    Clusterer();

    /// Constructor
    /// \param rowsMax Max number of rows to process
    /// \param padsMax Max number of pads to process
    /// \param timeBinsMax Max number of timebins to process
    /// \param minQMax Minimum peak charge for cluster
    /// \param requirePositiveCharge Positive charge is required
    /// \param requireNeighbouringPad Requires at least 2 adjecent pads with charge above threshold
    Clusterer(int rowsMax, int padsMax, int timeBinsMax, int minQMax, 
        bool requirePositiveCharge, bool requireNeighbouringPad);
    
    /// Destructor
    ~Clusterer() = default;
    
    /// Initialization function for clusterer
    virtual void Init() = 0;
    
    /// Processing all digits
    /// \param digits Container with TPC digits
    /// \return Container with clusters
    virtual ClusterContainer* Process(TClonesArray *digits) = 0;

    void setRowsMax(int val)                    { mRowsMax = val; };
    void setPadsMax(int val)                    { mPadsMax = val; };
    void setTimeBinsMax(int val)                { mTimeBinsMax = val; };
    void setMinQMax(float val)                  { mMinQMax = val; };
    void setRequirePositiveCharge(bool val)     { mRequirePositiveCharge = val; };
    void setRequireNeighbouringPad(bool val)    { mRequireNeighbouringPad = val; };

    int     getRowsMax()                  const { return mRowsMax; };
    int     getPadsMax()                  const { return mPadsMax; };
    int     getTimeBinsMax()              const { return mTimeBinsMax; };
    float   getMinQMax()                  const { return mMinQMax; };
    bool    hasRequirePositiveCharge()    const { return mRequirePositiveCharge; };
    bool    hasRequireNeighbouringPad()   const { return mRequireNeighbouringPad; };
    
  protected:
    
    ClusterContainer* mClusterContainer;    ///< Internal cluster storage
    
    int     mRowsMax;                       ///< Maximum row number
    int     mPadsMax;                       ///< Maximum pad number
    int     mTimeBinsMax;                   ///< Maximum time bin
    float   mMinQMax;                       ///< Minimun Qmax for cluster
    bool    mRequirePositiveCharge;         ///< If true, require charge > 0
    bool    mRequireNeighbouringPad;        ///< If true, require 2+ pads minimum
    
  };
}
}


#endif 

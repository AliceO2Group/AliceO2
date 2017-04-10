/// \file Cluster.h
/// \brief Cluster structure for TPC
#ifndef ALICEO2_TPC_CLUSTER_H
#define ALICEO2_TPC_CLUSTER_H

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>  // for base_object
#endif

#include "FairTimeStamp.h"                      // for FairTimeStamp
namespace boost { namespace serialization { class access; } }

namespace o2{
  namespace TPC{
    
    /// \class Cluster
    /// \brief Cluster class for the TPC
    ///
    class Cluster : public FairTimeStamp {
    public:
      
      /// Default constructor
      Cluster();
      
      /// Constructor, initializing all values
      /// \param cru CRU
      /// \param row Row
      /// \param q Total charge of cluster
      /// \param qmax Maximum charge in a single cell (pad, time)
      /// \param padmean Mean position of cluster in pad direction
      /// \param padsigma Sigma of cluster in pad direction
      /// \param timemean Mean position of cluster in time direction
      /// \param timesigma Sigma of cluster in time direction
      Cluster(short cru, short row, float q, float qmax, 
	      float padmean, float padsigma, float timemean, float timesigma);
      
      /// Destructor
      ~Cluster() override = default;

      /// Copy Constructor
      Cluster(const Cluster& other);

      /// Set parameters of cluster
      /// \param cru CRU
      /// \param row Row
      /// \param q Total charge of cluster
      /// \param qmax Maximum charge in a single cell (pad, time)
      /// \param padmean Mean position of cluster in pad direction
      /// \param padsigma Sigma of cluster in pad direction
      /// \param timemean Mean position of cluster in time direction
      /// \param timesigma Sigma of cluster in time direction
      void setParameters(short cru, short row, float q, float qmax, 
	      float padmean, float padsigma, float timemean, float timesigma);
            
      int getCRU() const { return mCRU; }
      int getRow() const { return mRow; }
      float getQ() const { return mQ; }
      float getQmax() const { return mQmax; }
      float getPadMean() const { return mPadMean; }
      float getTimeMean() const { return mTimeMean; }
      float getPadSigma() const { return mPadSigma; }
      float getTimeSigma() const { return mTimeSigma; }
      
      /// Print function: Print basic digit information on the  output stream
      /// \param output Stream to put the digit on
      /// \return The output stream
      friend std::ostream& operator<< (std::ostream& out, const Cluster &c) { return c.Print(out); }; 
//      std::ostream& Print(std::ostream &output) const;

    protected:      
      std::ostream& Print(std::ostream& out) const;
      
    private:
#ifndef __CINT__
      friend class boost::serialization::access;
#endif

      short   mCRU;
      short   mRow;
      float   mQ;
      float   mQmax;
      float   mPadMean;
      float   mPadSigma;
      float   mTimeMean;
      float   mTimeSigma;
            
      ClassDefOverride(Cluster, 1);
    };
  }
}

#endif /* ALICEO2_TPC_CLUSTER_H */   

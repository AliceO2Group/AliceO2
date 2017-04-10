/// \file BoxCluster.h
/// \brief Class to have some more info about the BoxClusterer clusters
#ifndef ALICEO2_TPC_BOXCLUSTER_H
#define ALICEO2_TPC_BOXCLUSTER_H

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>  // for base_object
#endif

#include "TPCSimulation/Cluster.h"
#include "FairTimeStamp.h"                      // for FairTimeStamp
#include "Rtypes.h"                             // for Double_t, ULong_t, etc

namespace boost { namespace serialization { class access; } }

namespace o2{
  namespace TPC{

    /// \class BoxCluster
    /// \brief Class to have some more info about the BoxClusterer clusters
    ///
    class BoxCluster : public Cluster {
    public:

      /// Default constructor
      BoxCluster();

      /// Constructor, initializing all values
      /// @param cru CRU (sector)
      /// @param row Row
      /// @param pad Pad with the maximum charge
      /// @param time Time bin with the maximum charge
      /// @param q Total charge of cluster
      /// @param qmax Maximum charge in a single cell (pad, time)
      /// @param padmean Mean position of cluster in pad direction
      /// @param padsigma Sigma of cluster in pad direction
      /// @param timemean Mean position of cluster in time direction
      /// @param timesigma Sigma of cluster in time direction
      /// @param size Number of pads * 10 + Number of time bins
      BoxCluster(Short_t cru, Short_t row, Short_t pad, Short_t time,
		 Float_t q, Float_t qmax, Float_t padmean,
		 Float_t padsigma, Float_t timemean, Float_t timesigma,
		 Short_t size);

      /// Destructor
      ~BoxCluster() override;

      /// Setter for special Box cluster parameters
      /// @param pad Pad with the maximum charge
      /// @param time Time bin with the maximum charge
      /// @param size Number of pads * 10 + Number of time bins
      void setBoxParameters(Short_t pad, Short_t time, Short_t size);

      Short_t getPad() const { return mPad; }
      Short_t getTime() const { return mTime; }
      Short_t getSize() const { return mSize; }
      // get number of pads covered by the cluster
      Short_t getNpads() const { return Short_t(mSize/10); }
      // get number of time bins covered by the cluster
      Short_t getNtimebins() const { return mSize%10; }


      /// Print function: Print basic information to the output stream
      /// @param output stream
      /// @return The output stream
      friend std::ostream& operator<< (std::ostream& out, const BoxCluster &c) { return c.Print(out); }
      std::ostream& Print(std::ostream &output) const;

    private:
#ifndef __CINT__
      friend class boost::serialization::access;
#endif

      Short_t                   mPad;
      Short_t                   mTime;
      Short_t                   mSize;

      ClassDefOverride(BoxCluster, 1);
    };
  }
}

#endif

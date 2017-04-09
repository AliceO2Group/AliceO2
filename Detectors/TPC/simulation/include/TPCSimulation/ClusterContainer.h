/// \file ClusterContainer.h
/// \brief Container class for TPC clusters
#ifndef _ALICEO2_TPC_ClusterContainer_
#define _ALICEO2_TPC_ClusterContainer_

#include "TPCSimulation/Cluster.h"
#include "Rtypes.h"
#include "TClonesArray.h"

namespace o2 {
  namespace TPC{
    class Cluster;

    /// \class ClusterContainer
    /// \brief Container class for TPC clusters
    class ClusterContainer{
    public:
      ClusterContainer();
      ~ClusterContainer();

      // Initialize the clones array
      // @param clusterType Possibility to store different types of clusters
      void InitArray(const Char_t* clusterType="o2::TPC::Cluster");

      // Empty array
      void Reset();

      /// Add cluster to array
      /// @param cru CRU (sector)
      /// @param row Row
      /// @param q Total charge of cluster
      /// @param qmax Maximum charge in a single cell (pad, time)
      /// @param padmean Mean position of cluster in pad direction
      /// @param padsigma Sigma of cluster in pad direction
      /// @param timemean Mean position of cluster in time direction
      /// @param timesigma Sigma of cluster in time direction
      Cluster* AddCluster(Int_t cru, Int_t row, Float_t qTot, Float_t qMax,
			  Float_t pad, Float_t time, Float_t sigmapad,
			  Float_t sigmatime);

      // Copy container info into the output container
      void FillOutputContainer(TClonesArray *outputcont);

      Int_t GetEntries() { return mClusterArray->GetEntries(); };

    private:
      Int_t         mNclusters;        // number of clusters
      TClonesArray* mClusterArray;      // array for clusters
    };
  }
}

#endif

/// \file TrackTPC.h
/// \brief Definition of the TPC Track
/// \author Thomas Klemenz, TU Muenchen, thomas.klemenz@tum.de


#ifndef ALICEO2_TPC_TRACKTPC
#define ALICEO2_TPC_TRACKTPC

#include "DetectorsBase/Track.h"

#include "TPCSimulation/Cluster.h"

#include <TClonesArray.h>

namespace o2 {
namespace TPC {

/// \class TrackTPC
/// This is the definition of the TPC Track Object


class TrackTPC final : public o2::Base::Track::TrackParBase {
  public:

    /// Default constructor
    TrackTPC();

    /// Constructor, initializing values for x, alpha and an array with Y, Z, sin(phi), tg(lambda) and q/pT
    /// \param x X of track evaluation
    /// \param alpha track frame angle
    /// \param std::array par contains Y, Z, sin(phi), tg(lambda) and q/pT
    TrackTPC(float x,float alpha, const std::array<float, o2::Base::Track::kNParams> &par);

    /// Constructor, initializing values for x,y and z in an array, momenta px, py and pz in an array
    /// \param std::array xyz contains x, y and z coordinates of the vertex of the origin
    /// \param std::array pxpypz contains momenta in x, y and z direction
    /// \param sign sign of the charge of the particle
    /// \param sectorAlpha false: angle of pT direction, true: angle of the sector from X, Y coordinate for r>1; angle of pT direction for r==0
    TrackTPC(const std::array<float,3> &xyz,const std::array<float,3> &pxpypz, int sign, bool sectorAlpha=true);

    /// Destructor
    ~TrackTPC() = default;

    /// Add a single cluster to the track
    void AddCluster(Cluster *c);

    /// Add an array of clusters to the track
    void AddClusterArray(TClonesArray *arr);

    /// Get the clusters which are associated with the track
    /// \return clusters of the track as a TClonesArray
    TClonesArray *GetClusterArray()  const { return mClusterArray; }

    /// Get the truncated mean energy loss of the track
    /// \param low low end of truncation
    /// \param high high end of truncation
    /// \param type 0 for Qmax, 1 for Q
    /// \param removeRows option to remove certain rows from the dEdx calculation
    /// \param nclPID pass any pointer to have the number of used clusters written to it
    /// \return mean energy loss
    float GetTruncatedMean(float low=0.05, float high=0.7, int type=1, int removeRows=0, int *nclPID=0x0) const;

  private:
    TClonesArray *mClusterArray;

};

inline
TrackTPC::TrackTPC()
  : TrackParBase()
  , mClusterArray(nullptr)
{}

inline
TrackTPC::TrackTPC(float x,float alpha, const std::array<float, o2::Base::Track::kNParams> &par)
  : TrackParBase(x, alpha, par)
  , mClusterArray(nullptr)
{
  mClusterArray = new TClonesArray("o2::TPC::Cluster");
}

inline
TrackTPC::TrackTPC(const std::array<float,3> &xyz,const std::array<float,3> &pxpypz, int sign, bool sectorAlpha)
  : TrackParBase(xyz, pxpypz, sign, sectorAlpha)
  , mClusterArray(nullptr)
{
  mClusterArray = new TClonesArray("o2::TPC::Cluster");
}

inline
void TrackTPC::AddCluster(Cluster *c)
{
  TClonesArray &clref = *mClusterArray;
  const size_t nCls = clref.GetEntriesFast();
  new(clref[nCls]) Cluster(*c);
  //c = (Cluster*)clref.ConstructedAt(nCls);
}

inline
void TrackTPC::AddClusterArray(TClonesArray *arr)
{
  mClusterArray->AbsorbObjects(arr);
}

}
}

#endif

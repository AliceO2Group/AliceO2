// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TPC_TRACKTPC
#define ALICEO2_TPC_TRACKTPC

#include "DetectorsBase/Track.h"

#include "TPCSimulation/Cluster.h"

#include <TClonesArray.h>


namespace o2 {
namespace TPC {

/// \class TrackTPC
/// This is the definition of the TPC Track Object


class TrackTPC {
  public:

    /// Default constructor
    TrackTPC();

    /// Constructor, initializing values for x, alpha and an array with Y, Z, sin(phi), tg(lambda) and q/pT
    /// \param x X of track evaluation
    /// \param alpha track frame angle
    /// \param std::array par contains Y, Z, sin(phi), tg(lambda) and q/pT
    TrackTPC(float x, float alpha, const std::array<float, o2::Base::Track::kNParams> &par, const std::array<float, o2::Base::Track::kCovMatSize> &cov);

    /// Constructor, initializing values for x,y and z in an array, momenta px, py and pz in an array
    /// \param std::array xyz contains x, y and z coordinates of the vertex of the origin
    /// \param std::array pxpypz contains momenta in x, y and z direction
    /// \param sign sign of the charge of the particle
    /// \param sectorAlpha false: angle of pT direction, true: angle of the sector from X, Y coordinate for r>1; angle of pT direction for r==0
    TrackTPC(const std::array<float,3> &xyz, const std::array<float,3> &pxpypz, const std::array<float, o2::Base::Track::kLabCovMatSize> &cv, int sign, bool sectorAlpha=true);

    /// Destructor
    ~TrackTPC() = default;

    /// Add a single cluster to the track
    void addCluster(Cluster &c);

    /// Add an array of clusters to the track
    void addClusterArray(TClonesArray *arr);

    /// Get the clusters which are associated with the track
    /// \return clusters of the track as a std::vector
    void getClusterVector(std::vector<Cluster> &clVec)  const { clVec = mClusterVector; }

    /// Get the truncated mean energy loss of the track
    /// \param low low end of truncation
    /// \param high high end of truncation
    /// \param type 0 for Qmax, 1 for Q
    /// \param removeRows option to remove certain rows from the dEdx calculation
    /// \param nclPID pass any pointer to have the number of used clusters written to it
    /// \return mean energy loss
    float getTruncatedMean(float low=0.05, float high=0.7, int type=1, int removeRows=0, int *nclPID=nullptr) const;


    float GetX()                         const { return mTrackParCov.GetX(); }
    float GetAlpha()                     const { return mTrackParCov.GetAlpha(); }
    float GetY()                         const { return mTrackParCov.GetY(); }
    float GetZ()                         const { return mTrackParCov.GetZ(); }
    float GetSnp()                       const { return mTrackParCov.GetSnp(); }
    float GetTgl()                       const { return mTrackParCov.GetTgl(); }
    float GetQ2Pt()                      const { return mTrackParCov.GetQ2Pt(); }

    // derived getters
    float GetCurvature(float b)          const { return mTrackParCov.GetCurvature(float(b));}
    float GetSign()                      const { return mTrackParCov.GetSign();}
    float GetPhi()                       const { return mTrackParCov.GetPhi();}
    float GetPhiPos()                    const { return mTrackParCov.GetPhiPos(); }

    float GetP()                         const { return mTrackParCov.GetP(); }
    float GetPt()                        const { return mTrackParCov.GetPt(); }
    void  GetXYZ(std::array<float,3> &xyz)           const { mTrackParCov.GetXYZ(xyz); }
    bool  GetPxPyPz(std::array<float,3> &pxyz)       const { return mTrackParCov.GetPxPyPz(pxyz); }
    bool  GetPosDir(std::array<float,9> &posdirp)    const { return mTrackParCov.GetPosDir(posdirp); }

    /// \todo implement getters for covariance (missing access to full covariance in Track.h)


  private:
    o2::Base::Track::TrackParCov mTrackParCov;
    std::vector<Cluster> mClusterVector;

};

inline
TrackTPC::TrackTPC()
  : mTrackParCov()
  , mClusterVector()
{}

inline
TrackTPC::TrackTPC(float x,float alpha, const std::array<float, o2::Base::Track::kNParams> &par, const std::array<float,o2::Base::Track::kCovMatSize> &cov)
  : mTrackParCov(x, alpha, par, cov)
  , mClusterVector()
{}

inline
TrackTPC::TrackTPC(const std::array<float,3> &xyz,const std::array<float,3> &pxpypz, const std::array<float,o2::Base::Track::kLabCovMatSize> &cv, int sign, bool sectorAlpha)
  : mTrackParCov(xyz, pxpypz, cv, sign, sectorAlpha)
  , mClusterVector()
{}

inline
void TrackTPC::addCluster(Cluster &c)
{
  mClusterVector.push_back(c);
}

inline
void TrackTPC::addClusterArray(TClonesArray *arr)
{
  for (auto clusterObject : *arr){
    Cluster *inputcluster = static_cast<Cluster*>(clusterObject);
    addCluster(*inputcluster);
  }
}

}
}

#endif

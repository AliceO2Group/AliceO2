// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterPEM.h
/// \brief Definition of a class to reconstruct clusters with the MLEM
/// algorithm
///
/// \author Gilles Grasseau, Subatech

#ifndef O2_MCH_CLUSTER_H_
#define O2_MCH_CLUSTER_H_

#include "MCHClustering/PadsPEM.h"

namespace o2
{
namespace mch
{
typedef std::pair<int, const double*> DataBlock_t;

typedef struct {
  PadIdx_t i;
  PadIdx_t j;
} MapKToIJ_t;

class ClusterPEM
{
 public:
  ClusterPEM();
  // Main constructor from initial data
  ClusterPEM(const double* x, const double* y, const double* dx, const double* dy,
             const double* q, const short* cathodes, const short* saturated,
             int chId, int nPads);
  ClusterPEM(Pads* pads0, Pads* pads1);
  // Extract a sub-cluster belonging to the group g
  ClusterPEM(ClusterPEM& cluster, Groups_t g);
  ~ClusterPEM();
  inline int getNbrOfPads(int c)
  {
    return (pads[c] == nullptr ? 0 : pads[c]->getNbrOfPads());
  };
  inline int getNbrOfPads()
  {
    return getNbrOfPads(0) + getNbrOfPads(1);
  };
  inline double getTotalCharge(int c)
  {
    return (pads[c] == nullptr ? 0 : pads[c]->getTotalCharge());
  };
  inline const double* getCharges(int c)
  {
    return (pads[c] == nullptr) ? nullptr : pads[c]->getCharges();
  }
  inline const Pads* getPads(int c) { return pads[c]; };
  inline const Groups_t* getCathGroup(int c) { return cathGroup[c]; };
  inline Groups_t* getProjPadGroup() { return projPadToGrp; };
  int getNbrOfPadsInGroup(int g);
  inline int getNbrOfProjectedPads()
  {
    return (projectedPads == nullptr ? -1 : projectedPads->getNbrOfPads());
  };
  // Unused - Old version
  // double *getProjPadsAsXYdXY( Groups_t group, const Mask_t* maskGrp, int
  // nbrProjPadsInTheGroup);
  const PadIdx_t* getMapCathPadToPad(int c)
  {
    return mapCathPadIdxToPadIdx[c];
  };
  // Build the 2 cathodes superosition geometry
  int buildProjectedGeometry(int includeAlonePads);
  const Pads* getProjectedPads() { return projectedPads; };
  // Project the charges of the 2 cathodes planes on the projected geometry
  double* projectChargeOnProjGeometry(int includeAlonePads);
  // Build Groups or geometrically independant Subclusters
  int buildGroupOfPads();
  // Add zero-charge pads to the cluster neighboring
  void addBoundaryPads();
  // Find local maximima with the PET algo
  int findLocalMaxWithPEM(double* thetaL, int nbrOfPadsInTheGroupCath);
  // Perform the fitting
  DataBlock_t fit(double* thetaInit, int K);
  // Not used in the Clustering/fitting
  // Just to check hit results
  int findLocalMaxWithBothCathodes(double* thetaOut, int kMax);

 private:
  // Cathodes planes
  int chamberId = -1;
  int singleCathPlaneID = -1;         // index of the unique cathode plane (0 or 1)
  int nbrOfCathodePlanes = 0;         // Nbr of Cathodes Plane
  Pads* pads[2] = {nullptr, nullptr}; // Two cathode-pads
  int nbrSaturated = 0;               // Number of saturated pads
  // Used for InspectModel
  PadIdx_t* mapCathPadIdxToPadIdx[2] = {
    nullptr, nullptr}; // Map cathode-pad index to pad index

  // Projection
  Pads* projectedPads = nullptr;     // Projected pads
  PadIdx_t* projNeighbors = nullptr; // Neighbors list of projected pads
  Groups_t* projPadToGrp = nullptr;  // Groups of projected pads
  int nbrOfProjGroups = 0;
  // Groups (i.e. cathode groups)
  Groups_t* cathGroup[2] = {nullptr, nullptr};
  int nbrOfCathGroups = 0;
  // Geometry
  PadIdx_t* IInterJ = nullptr;    // Compressed intersection matrix
  PadIdx_t* JInterI = nullptr;    // Compressed intersection matrix
  PadIdx_t* aloneIPads = nullptr; // Indexes of alone pads in cathode 0
  PadIdx_t* aloneJPads = nullptr; // Indexes of alone pads in cathode 1
  MapKToIJ_t* mapKToIJ =
    nullptr; // Mapping projected pads (k) to the 2 intersection pads
  PadIdx_t* mapIJToK =
    nullptr;                      // Inverse mapping (i,j) pads -> k (projected pads)
  PadIdx_t* aloneKPads = nullptr; // Alone (isolate) projected pads

  // Compute the projected geometry
  // Used by buildProjectedGeometry
  void computeProjectedPads(const Pads& pad0InfSup, const Pads& pad1InfSup,
                            int maxNbrOfProjPads, PadIdx_t* aloneIPads,
                            PadIdx_t* aloneJPads, PadIdx_t* aloneKPads,
                            int includeAlonePads);
  // Build groups from projected geometry (called proj-groups)
  int getConnectedComponentsOfProjPadsWOSinglePads();
  // Not Used - older version
  // void assignSingleCathPadsToGroup( short *padGroup, int nPads, int nGrp, int
  // nCath0, int nCath1, short *wellSplitGroup); Propagate the proj-groups to
  // the pads (of the 2 cathodes)
  int assignPadsToGroupFromProj(int nGrp);
  // Propagate the proj-groups to the pads (of the 2 cathodes)
  // Not used
  int assignGroupToCathPads();
  void updateProjectionGroups();
  // Unused - Old version
  int renumberGroupsFromMap(Groups_t* grpToGrp, int nGrp);
  // Renumber the groups id (starting from 0 and without gaps)
  int renumberGroups(Groups_t* grpToGrp, int nGrp);
  // Remove low charged groups
  void removeLowChargedGroups(int nGroups);
  // Keep the seeds inside the cluster area
  // Some fitting cases provide seeds outside of the cluster area
  int filterFitModelOnClusterRegion(Pads& pads, double* theta, int K,
                                    Mask_t* maskFilteredTheta);
  // Remove hits which are far from the strating position (theta0)
  int filterFitModelOnSpaceVariations(const double* theta0, int K0,
                                      double* theta, int K,
                                      Mask_t* maskFilteredTheta);

  // ???
  int getIndexByRow(const char* matrix, PadIdx_t N, PadIdx_t M, PadIdx_t* IIdx);
  int getIndexByColumns(const char* matrix, PadIdx_t N, PadIdx_t M,
                        PadIdx_t* JIdx);
  // Check the consistency of the different maps
  int checkConsistencyMapKToIJ(const char* intersectionMatrix,
                               const MapKToIJ_t* mapKToIJ,
                               const PadIdx_t* mapIJToK,
                               const PadIdx_t* aloneIPads,
                               const PadIdx_t* aloneJPads, int N0, int N1,
                               int nbrOfProjPads);

  // Not used in the Clustering/fitting
  // Just to check hit results
  int laplacian2D(const Pads& pads_, PadIdx_t* neigh, int chId,
                  PadIdx_t* sortedLocalMax, int kMax, double* smoothQ);
};

} // namespace mch
} // namespace o2

#endif // O2_MCH_CLUSTER_H_

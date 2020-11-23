// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  NearestNeighbour.h
/// \brief This class contains the a nearest neighbour search using a kdtree from CGAL
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Nov 23, 2020

#ifndef ALICEO2_TPC_NEARESTNEIGHBOUR_H_
#define ALICEO2_TPC_NEARESTNEIGHBOUR_H_

namespace o2
{
namespace tpc
{

template <typename DataT = double>
class NearestNeighbour
{
  /// for nearest neighbour search see: https://doc.cgal.org/latest/Spatial_searching/Spatial_searching_2searching_with_point_with_info_8cpp-example.html
 public:
  /// default constructor
  /// \param nPoints maximum number of points which will be filled in the tree
  NearestNeighbour(const int nPoints);

  /// add a point and their index to the tree
  void addPointAndIndex(const DataT z, const DataT r, const DataT phi, const unsigned int iZ, const unsigned int iR, const unsigned int iPhi) const;

  /// created the tree after all points are added
  void setTree() const;

  /// find the nearest neighbor/s for given query point
  /// \param z z position of query point
  /// \param r r position of query point
  /// \param phi phi position of query point
  /// \param zNearest z position of nearest point found
  /// \param rNearest r position of nearest point found
  /// \param phiNearest phi position of nearest point found
  /// \param iZNearest index in z direction of nearest point found
  /// \param iRNearest index in r direction of nearest point found
  /// \param iPhiNearest index in phi direction of nearest point found
  void query(const DataT z, const DataT r, const DataT phi, DataT& zNearest, DataT& rNearest, DataT& phiNearest, unsigned int& iZNearest, unsigned int& iRNearest, unsigned int& iPhiNearest) const;
};

} // namespace tpc
} // namespace o2

#endif

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  NearestNeighbour.cxx
/// \brief Definition of NearestNeighbour class
///
/// \author  Matthias Kleiner <mkleiner@ikf.uni-frankfurt.de>
/// \date Nov 23, 2020

#include "TPCSpaceCharge/NearestNeighbour.h"

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/property_map.h>
#include <boost/iterator/zip_iterator.hpp>
#include <vector>

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Point_Index = boost::tuple<Point_3, std::array<unsigned int, 3>>;
using Traits_base = CGAL::Search_traits_3<Kernel>;
using Traits = CGAL::Search_traits_adapter<Point_Index, CGAL::Nth_of_tuple_property_map<0, Point_Index>, Traits_base>;
using K_neighbor_search = CGAL::Orthogonal_k_neighbor_search<Traits>;
using Tree = K_neighbor_search::Tree;

class NearestNeighbourInternal
{
 public:
  /// constructor
  /// \param nPoints maximum number of points which will be filled in the tree
  NearestNeighbourInternal(const int nPoints)
  {
    mPoints.reserve(nPoints);
    mIndices.reserve(nPoints);
  }

  auto& getPoints() { return mPoints; }
  const auto& getPoints() const { return mPoints; }

  auto& getIndices() { return mIndices; }
  const auto& getIndices() const { return mIndices; }

  auto& getTree() { return mTree; }
  const auto& getTree() const { return mTree; }

 private:
  std::vector<Point_3> mPoints;                      ///< points which can be searched for their nearest neighbour
  std::vector<std::array<unsigned int, 3>> mIndices; /// indices of the points
  std::unique_ptr<Tree> mTree;                       ///< kd-tree which will be used for querying nearest neighbours
};

template <typename DataT>
inline o2::tpc::NearestNeighbour<DataT>::NearestNeighbour(const int nPoints)
{
  mInternal = std::make_unique<NearestNeighbourInternal>(nPoints);
}

template <typename DataT>
o2::tpc::NearestNeighbour<DataT>::~NearestNeighbour() = default;

template <typename DataT>
auto& o2::tpc::NearestNeighbour<DataT>::tree()
{
  return mInternal->getTree();
}

template <typename DataT>
const auto& o2::tpc::NearestNeighbour<DataT>::tree() const
{
  return mInternal->getTree();
}

template <typename DataT>
auto& o2::tpc::NearestNeighbour<DataT>::indices()
{
  return mInternal->getIndices();
}

template <typename DataT>
const auto& o2::tpc::NearestNeighbour<DataT>::indices() const
{
  return mInternal->getIndices();
}

template <typename DataT>
auto& o2::tpc::NearestNeighbour<DataT>::points()
{
  return mInternal->getPoints();
}

template <typename DataT>
const auto& o2::tpc::NearestNeighbour<DataT>::points() const
{
  return mInternal->getPoints();
}

template <typename DataT>
void o2::tpc::NearestNeighbour<DataT>::addPointAndIndex(const DataT z, const DataT r, const DataT phi, const unsigned int iZ, const unsigned int iR, const unsigned int iPhi)
{
  points().emplace_back(Point_3(z, r, phi));
  indices().emplace_back(std::array<unsigned int, 3>{iZ, iR, iPhi});
}

template <typename DataT>
void o2::tpc::NearestNeighbour<DataT>::query(const DataT z, const DataT r, const DataT phi, DataT& zNearest, DataT& rNearest, DataT& phiNearest, unsigned int& iZNearest, unsigned int& iRNearest, unsigned int& iPhiNearest) const
{
  const int neighbors = 1;
  const K_neighbor_search search(*tree(), Point_3(z, r, phi), neighbors);
  const K_neighbor_search::iterator it = search.begin();

  zNearest = static_cast<DataT>(boost::get<0>(it->first)[0]);
  rNearest = static_cast<DataT>(boost::get<0>(it->first)[1]);
  phiNearest = static_cast<DataT>(boost::get<0>(it->first)[2]);

  iZNearest = boost::get<1>(it->first)[0];
  iRNearest = boost::get<1>(it->first)[1];
  iPhiNearest = boost::get<1>(it->first)[2];
}

template <typename DataT>
void o2::tpc::NearestNeighbour<DataT>::setTree()
{
  tree() = std::make_unique<Tree>(boost::make_zip_iterator(boost::make_tuple(points().begin(), indices().begin())), boost::make_zip_iterator(boost::make_tuple(points().end(), indices().end())));
}

template class o2::tpc::NearestNeighbour<double>;
template class o2::tpc::NearestNeighbour<float>;

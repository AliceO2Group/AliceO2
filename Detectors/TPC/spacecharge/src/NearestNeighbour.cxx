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

std::unique_ptr<std::vector<Point_3>> mPoints;
std::unique_ptr<std::vector<std::array<unsigned int, 3>>> mIndices;
std::unique_ptr<Tree> mTree;

template <typename DataT>
inline o2::tpc::NearestNeighbour<DataT>::NearestNeighbour(const int nPoints)
{
  mPoints = std::make_unique<std::vector<Point_3>>();
  mIndices = std::make_unique<std::vector<std::array<unsigned int, 3>>>();
  mPoints->reserve(nPoints);
  mIndices->reserve(nPoints);
}

template <typename DataT>
void o2::tpc::NearestNeighbour<DataT>::addPointAndIndex(const DataT z, const DataT r, const DataT phi, const unsigned int iZ, const unsigned int iR, const unsigned int iPhi) const
{
  mPoints->emplace_back(Point_3(z, r, phi));
  mIndices->emplace_back(std::array<unsigned int, 3>{iZ, iR, iPhi});
}

template <typename DataT>
void o2::tpc::NearestNeighbour<DataT>::query(const DataT z, const DataT r, const DataT phi, DataT& zNearest, DataT& rNearest, DataT& phiNearest, unsigned int& iZNearest, unsigned int& iRNearest, unsigned int& iPhiNearest) const
{
  const int neighbors = 1;
  const K_neighbor_search search(*mTree, Point_3(z, r, phi), neighbors);
  const K_neighbor_search::iterator it = search.begin();

  zNearest = static_cast<DataT>(boost::get<0>(it->first)[0]);
  rNearest = static_cast<DataT>(boost::get<0>(it->first)[1]);
  phiNearest = static_cast<DataT>(boost::get<0>(it->first)[2]);

  iZNearest = boost::get<1>(it->first)[0];
  iRNearest = boost::get<1>(it->first)[1];
  iPhiNearest = boost::get<1>(it->first)[2];
}

template <typename DataT>
void o2::tpc::NearestNeighbour<DataT>::setTree() const
{
  mTree = std::make_unique<Tree>(boost::make_zip_iterator(boost::make_tuple(mPoints->begin(), mIndices->begin())), boost::make_zip_iterator(boost::make_tuple(mPoints->end(), mIndices->end())));
}

template class o2::tpc::NearestNeighbour<double>;
template class o2::tpc::NearestNeighbour<float>;

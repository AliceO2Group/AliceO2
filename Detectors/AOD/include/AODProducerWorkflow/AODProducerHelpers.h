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

/// @file   AODProducerHelpers.h
/// common helpers for AOD producers

#ifndef O2_AODPRODUCER_HELPERS
#define O2_AODPRODUCER_HELPERS

#include <boost/functional/hash.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/unordered_map.hpp>
#include <string>
#include <vector>
#include <Framework/AnalysisHelpers.h>

namespace o2::aodhelpers
{

typedef boost::tuple<int, int, int> Triplet_t;

struct TripletHash {
  std::size_t operator()(Triplet_t const& e) const
  {
    std::size_t seed = 0;
    boost::hash_combine(seed, e.get<0>());
    boost::hash_combine(seed, e.get<1>());
    boost::hash_combine(seed, e.get<2>());
    return seed;
  }
};

struct TripletEqualTo {
  bool operator()(Triplet_t const& x, Triplet_t const& y) const
  {
    return (x.get<0>() == y.get<0>() &&
            x.get<1>() == y.get<1>() &&
            x.get<2>() == y.get<2>());
  }
};

typedef boost::unordered_map<Triplet_t, int, TripletHash, TripletEqualTo> TripletsMap_t;

template <typename T>
framework::Produces<T> createTableCursor(framework::ProcessingContext& pc)
{
  framework::Produces<T> c;
  c.resetCursor(pc.outputs()
                  .make<framework::TableBuilder>(framework::OutputForTable<T>::ref()));
  c.setLabel(o2::aod::MetadataTrait<T>::metadata::tableLabel());
  return c;
}
} // namespace o2::aodhelpers

#endif /* O2_AODPRODUCER_HELPERS */

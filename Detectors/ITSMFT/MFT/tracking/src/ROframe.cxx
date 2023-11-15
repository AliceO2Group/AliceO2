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
///
/// \file ROframe.cxx
///

#include "MFTTracking/ROframe.h"

#include <iostream>

namespace o2
{
namespace mft
{

template <typename T>
Int_t ROframe<T>::getTotalClusters() const
{
  size_t totalClusters{0};
  for (auto& clusters : mClusters) {
    totalClusters += clusters.size();
  }
  return Int_t(totalClusters);
}

template class ROframe<o2::mft::TrackLTF>;
template class ROframe<o2::mft::TrackLTFL>;

} // namespace mft
} // namespace o2

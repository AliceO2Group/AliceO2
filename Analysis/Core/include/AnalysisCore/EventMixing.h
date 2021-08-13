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

#ifndef ANALYSIS_CORE_EVENTMIXING_H_
#define ANALYSIS_CORE_EVENTMIXING_H_

namespace eventmixing
{
/// Calculate hash for an element based on 2 properties and their bins.
/// \tparam T1 Data type of the configurable of the z-vertex and multiplicity bins
/// \tparam T2 Data type of the value of the z-vertex and multiplicity
/// \param vtxBins Binning in z-vertex
/// \param multBins Binning in multiplicity
/// \param vtx Value of the z-vertex of the collision
/// \param mult Multiplicity of the collision
/// \return Hash of the event
template <typename T1, typename T2>
static int getMixingBin(const T1& vtxBins, const T1& multBins, const T2& vtx, const T2& mult)
{
  // underflow
  if (vtx < vtxBins.at(0)) {
    return -1;
  }
  if (mult < multBins.at(0)) {
    return -1;
  }

  for (int i = 1; i < vtxBins.size(); i++) {
    if (vtx < vtxBins.at(i)) {
      for (int j = 1; j < multBins.size(); j++) {
        if (mult < multBins.at(j)) {
          return i + j * (vtxBins.size() + 1);
        }
      }
    }
  }
  // overflow
  return -1;
}
}; // namespace eventmixing

#endif /* ANALYSIS_CORE_EVENTMIXING_H_ */

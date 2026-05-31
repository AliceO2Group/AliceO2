// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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
/// Holder of auxiallary information used for tuning
///

#include "ITStracking/Constants.h"
#include "GPUCommonRtypes.h"

namespace o2::its
{
struct TrackletMC final {
  float tgl{constants::UnsetValue}; // tanLambda
  float phi{constants::UnsetValue}; // phi
  float rIn{constants::UnsetValue};
  float zIn{constants::UnsetValue};
  float phiIn{constants::UnsetValue};
  float rOut{constants::UnsetValue};
  float zOut{constants::UnsetValue};
  float phiOut{constants::UnsetValue};
  float dr{constants::UnsetValue};
  float dz{constants::UnsetValue};
  float dPhi{constants::UnsetValue};
  bool ok{false}; // truth
  /// below only metrics valid if ok
  bool prim{false};                 // primary
  float dXY{constants::UnsetValue}; // transverse distance to event
  float dZ{constants::UnsetValue};  // longitudinal distance to event
  float deltaZEvent{constants::UnsetValue};
  float tglEvent{constants::UnsetValue};
  ClassDefNV(TrackletMC, 2);
};

} // namespace o2::its

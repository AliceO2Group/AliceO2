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

/// \file DCAFitterN.cxx
/// \brief Defintions for N-prongs secondary vertex fit
/// \author ruben.shahoyan@cern.ch, adapted from central barrel to fwd rapidities by Rita Sadek, rita.sadek@cern.ch

#include "DCAFitter/FwdDCAFitterN.h"

namespace o2
{
namespace vertexing
{

void __test_instance__()
{
  FwdDCAFitter2 ft2;
  FwdDCAFitter3 ft3;
  o2::track::TrackParCovFwd tr;
  ft2.process(tr, tr);
  ft3.process(tr, tr, tr);
}

} // namespace vertexing
} // namespace o2

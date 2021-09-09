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
/// \file Label.cxx
/// \brief
///

#include "ITStracking/Label.h"

namespace o2
{
namespace its
{

Label::Label(const int mcId, const float pT, const float phi, const float eta, const int pdg, const int ncl)
  : monteCarloId{mcId},
    transverseMomentum{pT},
    phiCoordinate{phi},
    pseudorapidity{eta},
    pdgCode{pdg},
    numberOfClusters{ncl}
{
  // Nothing to do
}

std::ostream& operator<<(std::ostream& outputStream, const Label& label)
{
  outputStream << label.monteCarloId << "\t" << label.transverseMomentum << "\t" << label.phiCoordinate << "\t"
               << label.pseudorapidity << "\t" << label.pdgCode << "\t" << label.numberOfClusters;

  return outputStream;
}
} // namespace its
} // namespace o2

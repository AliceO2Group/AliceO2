// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \file Label.cxx
/// \brief 
///

#include "ITSReconstruction/CA/Label.h"

namespace o2
{
namespace ITS
{
namespace CA
{

Label::Label(const int monteCarloId, const float transverseMomentum, const float phiCoordinate,
    const float pseudorapidity, const int pdgCode, const int numberOfClusters)
    : monteCarloId { monteCarloId }, transverseMomentum { transverseMomentum }, phiCoordinate { phiCoordinate }, pseudorapidity {
        pseudorapidity }, pdgCode { pdgCode }, numberOfClusters { numberOfClusters }
{
  // Nothing to do
}

std::ostream& operator<<(std::ostream& outputStream, const Label& label)
{
  outputStream << label.monteCarloId << "\t" << label.transverseMomentum << "\t" << label.phiCoordinate << "\t"
      << label.pseudorapidity << "\t" << label.pdgCode << "\t" << label.numberOfClusters;

  return outputStream;
}

}
}
}

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PreCluster.cxx
/// \brief Implementation of the MCH precluster minimal structure
///
/// \author Philippe Pillot, Subatech

#include "MCHBase/PreCluster.h"

namespace o2
{
namespace mch
{

using namespace std;

//_________________________________________________________________
void PreCluster::print(std::ostream& stream, gsl::span<const Digit> digits) const
{
  /// print the precluster, getting the associated digits from the provided span

  if (lastDigit() >= digits.size()) {
    stream << "the vector of digits is too small to contain the digits of this precluster" << endl;
  }

  int i(0);
  stream << "  nDigits = " << nDigits << endl;
  for (const auto& digit : digits.subspan(firstDigit, nDigits)) {
    stream << "  digit[" << i++ << "] = " << digit.getDetID() << "," << digit.getPadID() << "," << digit.getADC()
           << "," << digit.getTime().bunchCrossing << "-" << digit.getTime().sampaTime << endl;
  }
}

} // namespace mch
} // namespace o2

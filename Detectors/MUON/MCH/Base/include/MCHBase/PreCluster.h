// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PreCluster.h
/// \brief Definition of the MCH precluster minimal structure
///
/// \author Philippe Pillot, Subatech

#ifndef ALICEO2_MCH_PRECLUSTER_H_
#define ALICEO2_MCH_PRECLUSTER_H_

#include <iostream>

#include <gsl/span>

#include "DataFormatsMCH/Digit.h"

namespace o2
{
namespace mch
{

/// precluster minimal structure
struct PreCluster {
  uint32_t firstDigit; ///< index of first associated digit in the ordered vector of digits
  uint32_t nDigits;    ///< number of digits attached to this precluster

  /// return the index of last associated digit in the ordered vector of digits
  uint32_t lastDigit() const { return firstDigit + nDigits - 1; }

  void print(std::ostream& stream, gsl::span<const Digit> digits) const;
};

} // namespace mch
} // namespace o2

#endif // ALICEO2_MCH_PRECLUSTER_H_

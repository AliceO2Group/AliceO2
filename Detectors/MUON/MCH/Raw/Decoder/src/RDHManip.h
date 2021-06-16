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

#ifndef O2_MCH_RAW_RDH_MANIP_H
#define O2_MCH_RAW_RDH_MANIP_H

#include <cstdint>
#include <functional>
#include <gsl/span>
#include <vector>
#include <cstddef>
#include "Headers/RDHAny.h"

namespace o2::mch::raw
{

/// Append bytes from RDH to the buffer
void appendRDH(std::vector<std::byte>& buffer, const o2::header::RDHAny& rdh);

/// Create a RDH of the given version from the buffer
/// (which should have the right size and content to be a valid RDH)
o2::header::RDHAny createRDH(gsl::span<const std::byte> buffer, int version);

/// Count the number of RDHs in the buffer
int countRDHs(gsl::span<const std::byte> buffer);

/// Dump the RDHs found in the buffer
int showRDHs(gsl::span<const std::byte> buffer);

} // namespace o2::mch::raw

#endif

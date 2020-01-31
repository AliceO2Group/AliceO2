// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file digit.h
/// \author Felix Weiglhofer
//
#if !defined(SHARED_DIGIT_H)
#define SHARED_DIGIT_H

#include "clusterFinderDefs.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
namespace deprecated
{

struct PackedDigit {
  float charge;
  Timestamp time;
  Pad pad;
  Row row;
};

using Digit = PackedDigit;

} // namespace deprecated
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif //!defined(SHARED_DIGIT_H)

// vim: set ts=4 sw=4 sts=4 expandtab:

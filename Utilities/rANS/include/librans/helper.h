// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   helper.h
/// @author Michael Lettrich
/// @since  2019-05-21
/// @brief  various helper functions

#include <cstddef>

#ifndef RANS_HELPER_H
#define RANS_HELPER_H

namespace o2
{
namespace rans
{

template <typename T>
constexpr bool needs64Bit()
{
  return sizeof(T) > 4;
}

constexpr size_t bitsToRange(size_t bits)
{
  return 1 << bits;
}

} // namespace rans
} // namespace o2

#endif /* RANS_HELPER_H */

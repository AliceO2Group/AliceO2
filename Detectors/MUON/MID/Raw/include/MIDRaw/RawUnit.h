// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDRaw/RawUnit.h
/// \brief  Raw data format MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   30 September 2019
#ifndef O2_MID_RAWUNIT_H
#define O2_MID_RAWUNIT_H

#include <cstdint>
#include <cstddef>
#include "Headers/RAWDataHeader.h"

namespace o2
{
namespace mid
{
namespace raw
{
typedef uint32_t RawUnit;

// Buffer size
static constexpr size_t sElementSizeInBytes = sizeof(RawUnit);
static constexpr size_t sMaxBufferSize = 8192 / sElementSizeInBytes;
static constexpr size_t sElementSizeInBits = 8 * sElementSizeInBytes;

// Header size
static constexpr size_t sHeaderSizeInBytes = sizeof(header::RAWDataHeader);
static constexpr size_t sHeaderSizeInElements = sHeaderSizeInBytes / sElementSizeInBytes;

} // namespace raw
} // namespace mid
} // namespace o2

#endif /* O2_MID_RAWUNIT_H */
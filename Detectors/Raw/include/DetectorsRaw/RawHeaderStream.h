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

/// \file RawHeaderStream.h
/// \brief Input stream operators for raw header 4 to 7 from binary file
///
/// Helpers to define input stream operator for raw headers v4 and v7 from
/// binary file input stream, used in RawReaderFile

#ifndef DETECTOR_RAW_RAWHEADERSTREAM_H
#define DETECTOR_RAW_RAWHEADERSTREAM_H

#include <iosfwd>
#include "Headers/RAWDataHeader.h"

namespace o2
{

namespace raw
{

std::istream& operator>>(std::istream& stream, o2::header::RAWDataHeaderV4& header);
std::istream& operator>>(std::istream& stream, o2::header::RAWDataHeaderV5& header);
std::istream& operator>>(std::istream& stream, o2::header::RAWDataHeaderV6& header);
std::istream& operator>>(std::istream& stream, o2::header::RAWDataHeaderV7& header);

std::ostream& operator<<(std::ostream& stream, const o2::header::RAWDataHeaderV4& header);
std::ostream& operator<<(std::ostream& stream, const o2::header::RAWDataHeaderV5& header);
std::ostream& operator<<(std::ostream& stream, const o2::header::RAWDataHeaderV6& header);
std::ostream& operator<<(std::ostream& stream, const o2::header::RAWDataHeaderV7& header);

} // namespace raw

} // namespace o2
#endif // DETECTOR_RAW_RAWHEADERSTREAM_H

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   rans.h
/// @author Michael Lettrich
/// @since  2019-05-21
/// @brief  header for public api

#ifndef RANS_RANS_H
#define RANS_RANS_H

#include "rANS/SymbolStatistics.h"
#include "rANS/Encoder.h"
#include "rANS/Decoder.h"

namespace o2
{
namespace rans
{
template <typename source_T>
using Encoder32 = Encoder<uint32_t, uint8_t, source_T>;
template <typename source_T>
using Encoder64 = Encoder<uint64_t, uint32_t, source_T>;

template <typename source_T>
using Decoder32 = Decoder<uint32_t, uint8_t, source_T>;
template <typename source_T>
using Decoder64 = Decoder<uint64_t, uint32_t, source_T>;

} // namespace rans
} // namespace o2

#endif /* RANS_RANS_H */

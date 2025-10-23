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
/// \file Definitions.h
/// \brief

#ifndef TRACKINGITS_DEFINITIONS_H_
#define TRACKINGITS_DEFINITIONS_H_

#include <type_traits>

#include "ReconstructionDataFormats/Vertex.h"

#ifdef CA_DEBUG
#define CA_DEBUGGER(x) x
#else
#define CA_DEBUGGER(x) \
  do {                 \
  } while (0)
#endif

namespace o2::its
{

enum class TrackletMode {
  Layer0Layer1 = 0,
  Layer1Layer2 = 2
};

using Vertex = o2::dataformats::Vertex<o2::dataformats::TimeStamp<int>>;

template <bool IsConst, typename T>
using maybe_const = typename std::conditional<IsConst, const T, T>::type;

} // namespace o2::its

#endif

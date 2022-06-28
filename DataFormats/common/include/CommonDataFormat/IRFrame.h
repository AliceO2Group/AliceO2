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

/// \file IRFrame.h
/// \brief Class to delimit start and end IR of certain time period
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_IRFRAME_H
#define ALICEO2_IRFRAME_H

#include "MathUtils/Primitive2D.h"
#include "CommonDataFormat/InteractionRecord.h"

namespace o2
{
namespace dataformats
{

// Bracket of 2 IRs.
// We could just alias it to the bracket specialization, but this would create
// problems with fwd.declaration
struct IRFrame : public o2::math_utils::detail::Bracket<o2::InteractionRecord> {
  using o2::math_utils::detail::Bracket<o2::InteractionRecord>::Bracket;

  uint64_t info = 0;

  ClassDefNV(IRFrame, 2);
};

} // namespace dataformats
} // namespace o2

#endif

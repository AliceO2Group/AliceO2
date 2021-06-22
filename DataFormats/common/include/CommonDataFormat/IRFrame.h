// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  ClassDefNV(IRFrame, 1);
};

} // namespace dataformats
} // namespace o2

#endif

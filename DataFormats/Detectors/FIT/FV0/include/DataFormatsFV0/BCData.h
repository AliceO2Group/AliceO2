// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _FV0_BC_DATA_H_
#define _FV0_BC_DATA_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include <Rtypes.h>
#include <gsl/span>

/// \file BCData.h
/// \brief Class to describe fired triggered and/or stored channels for the BC and to refer to channel data
/// \author ruben.shahoyan@cern.ch -> maciej.slupecki@cern.ch

namespace o2
{
namespace fv0
{
class ChannelData;

struct BCData {
  /// we are going to refer to at most 48 channels, so 6 bits for the number of channels and 26 for the reference
  o2::dataformats::RangeRefComp<6> ref;
  o2::InteractionRecord ir;

  BCData() = default;
  BCData(int first, int ne, o2::InteractionRecord iRec)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    ir = iRec;
  }

  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;
  void print() const;

  ClassDefNV(BCData, 1);
};
} // namespace fv0
} // namespace o2

#endif

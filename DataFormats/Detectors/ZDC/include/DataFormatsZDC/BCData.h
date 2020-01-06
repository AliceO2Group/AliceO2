// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _ZDC_BC_DATA_H_
#define _ZDC_BC_DATA_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <gsl/span>

/// \file BCData.h
/// \brief Class to describe fired triggered and/or stored channels for the BC and to refer to channel data
/// \author ruben.shahoyan@cern.ch

namespace o2
{
namespace zdc
{
class ChannelData;

struct BCData {
  /// we are going to refer to at most 26 channels, so 5 bits for the NChannels and 27 for the reference
  o2::dataformats::RangeRefComp<5> ref;
  o2::InteractionRecord ir;
  uint32_t channels = 0; // pattern of channels it refers to
  uint32_t triggers = 0; // pattern of triggered channels (not necessarily stored) in this BC

  BCData() = default;
  BCData(int first, int ne, o2::InteractionRecord iRec, uint32_t chSto, uint32_t chTrig)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    ir = iRec;
    channels = chSto;
    triggers = chTrig;
  }

  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;
  void print() const;

  ClassDefNV(BCData, 1);
};
} // namespace zdc
} // namespace o2

#endif

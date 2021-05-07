// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _ZDC_BC_REC_DATA_H
#define _ZDC_BC_REC_DATA_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <gsl/span>

/// \file BCData.h
/// \brief Class to describe reconstructed data and refer to channel data
/// \author ruben.shahoyan@cern.ch, pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{
class ChannelData;


struct BCRecData {
  /// we are going to refer to at most 26 channels, so 5 bits for the NChannels and 27 for the reference
  o2::dataformats::RangeRefComp<5> refe;
  o2::dataformats::RangeRefComp<5> reft;
  o2::InteractionRecord ir;
  uint32_t flags;

  BCRecData() = default;
  BCRecData(int firste, int ne, int firstt, int nt, o2::InteractionRecord iRec, uint32_t fl)
  {
    refe.setFirstEntry(firste);
    refe.setEntries(ne);
    reft.setFirstEntry(firstt);
    reft.setEntries(nt);
    ir = iRec;
    flags = fl;
  }

  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;
  void print() const;

  ClassDefNV(BCRecData, 1 b);
};
} // namespace zdc
} // namespace o2

#endif

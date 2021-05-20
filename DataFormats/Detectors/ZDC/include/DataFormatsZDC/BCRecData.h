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
  /// we are going to refer to at most 26 channels, so 5 bits are reserved
  o2::dataformats::RangeRefComp<5> refe; // Reference to reconstructed energy
  o2::dataformats::RangeRefComp<5> reft; // Reference to reconstructed TDC
  o2::dataformats::RangeRefComp<5> refm; // Reference to reconstruction error/information flags
  o2::InteractionRecord ir;

  BCRecData() = default;
  BCRecData(int firste, int firstt, int firstm, o2::InteractionRecord iRec)
  {
    refe.setFirstEntry(firste);
    refe.setEntries(0);
    reft.setFirstEntry(firstt);
    reft.setEntries(0);
    refm.setFirstEntry(firstm);
    refm.setEntries(0);
    ir = iRec;
  }

  inline void addEnergy(){
    refe.setEntries(refe.getEntries()+1);
  }
  void print() const;

  ClassDefNV(BCRecData, 1);
};
} // namespace zdc
} // namespace o2

#endif

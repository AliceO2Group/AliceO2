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

#ifndef O2_ZDC_BC_REC_DATA_H
#define O2_ZDC_BC_REC_DATA_H

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "ZDCBase/Constants.h"
#include <Rtypes.h>
#include <gsl/span>

/// \file BCRecData.h
/// \brief Class to refer to the reconstructed information
/// \author ruben.shahoyan@cern.ch, pietro.cortese@cern.ch

namespace o2
{
namespace zdc
{
class ChannelData;

struct BCRecData {
  o2::InteractionRecord ir;
  uint32_t channels = 0;                 // Pattern of channels acquired
  uint32_t triggers = 0;                 // Pattern of channels with autotrigger bit
  o2::dataformats::RangeRefComp<5> refe; // Reference to reconstructed energy
  o2::dataformats::RangeRefComp<5> reft; // Reference to reconstructed TDC
  o2::dataformats::RangeRefComp<5> refi; // Reference to reconstruction error/information flags
  o2::dataformats::RangeRefComp<5> refw; // Reference to waveform interpolated data

  BCRecData() = default;

  // Insert interaction record for new event (will set later the number of entries)
  BCRecData(int firste, int firstt, int firsti, o2::InteractionRecord iRec)
  {
    refe.setFirstEntry(firste);
    refe.setEntries(0);
    reft.setFirstEntry(firstt);
    reft.setEntries(0);
    refi.setFirstEntry(firsti);
    refi.setEntries(0);
    refw.setFirstEntry(0);
    refw.setEntries(0);
    ir = iRec;
  }

  BCRecData(int firste, int firstt, int firsti, int firstd, o2::InteractionRecord iRec)
  {
    refe.setFirstEntry(firste);
    refe.setEntries(0);
    reft.setFirstEntry(firstt);
    reft.setEntries(0);
    refi.setFirstEntry(firsti);
    refi.setEntries(0);
    refw.setFirstEntry(firstd);
    refw.setEntries(0);
    ir = iRec;
  }

  // Update counter of energy entries
  inline void addEnergy()
  {
    refe.setEntries(refe.getEntries() + 1);
  }

  // Update counter of TDC entries
  inline void addTDC()
  {
    reft.setEntries(reft.getEntries() + 1);
  }

  // Update counter of Info entries
  inline void addInfo()
  {
    refi.setEntries(refi.getEntries() + 1);
  }

  // Update counter of Waveform entries
  inline void addWaveform()
  {
    refw.setEntries(refw.getEntries() + 1);
  }

  // Get information about event
  inline void getRef(int& firste, int& ne, int& firstt, int& nt, int& firsti, int& ni)
  {
    firste = refe.getFirstEntry();
    firstt = reft.getFirstEntry();
    firsti = refi.getFirstEntry();
    ne = refe.getEntries();
    nt = reft.getEntries();
    ni = refi.getEntries();
  }

  inline void getRef(int& firste, int& ne, int& firstt, int& nt, int& firsti, int& ni, int& firstw, int& nw)
  {
    firste = refe.getFirstEntry();
    firstt = reft.getFirstEntry();
    firsti = refi.getFirstEntry();
    firstw = refw.getFirstEntry();
    ne = refe.getEntries();
    nt = reft.getEntries();
    ni = refi.getEntries();
    nw = refw.getEntries();
  }
  inline void getRefE(int& firste, int& ne)
  {
    firste = refe.getFirstEntry();
    ne = refe.getEntries();
  }
  inline void getRefT(int& firstt, int& nt)
  {
    firstt = reft.getFirstEntry();
    nt = reft.getEntries();
  }
  inline void getRefI(int& firsti, int& ni)
  {
    firsti = refi.getFirstEntry();
    ni = refi.getEntries();
  }
  inline void getRefW(int& firstw, int& nw)
  {
    firstw = refw.getFirstEntry();
    nw = refw.getEntries();
  }

  void print() const;

  ClassDefNV(BCRecData, 2);
};
} // namespace zdc
} // namespace o2

#endif

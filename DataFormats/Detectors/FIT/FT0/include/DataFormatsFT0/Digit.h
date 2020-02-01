// Copyright CERN and copyright holders of ALICE O2.This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef _FT0_DIGIT_H_
#define _FT0_DIGIT_H_

#include "CommonDataFormat/InteractionRecord.h"
#include "CommonDataFormat/RangeReference.h"
#include "CommonDataFormat/TimeStamp.h"
#include "DataFormatsFT0/ChannelData.h"
#include <Rtypes.h>
#include <gsl/span>
#include <bitset>

/// \file Digit.h
/// \brief Class to describe fired triggered and/or stored channels for the BC and to refer to channel data
/// \author ruben.shahoyan@cern.ch

namespace o2
{
namespace ft0
{
class ChannelData;

struct Triggers {
  union {
    int64_t word = 0;
    struct {
      int64_t orC : 1,
        orA : 1,
        sCen : 1,
        cen : 1,
        vertex : 1,
        nChanA : 7,
        nChanC : 7,
        amplA : 12,
        amplC : 12,
        timeA : 9,
        timeC : 9,
        rest1 : 3;
    };
  };
  ClassDefNV(Triggers, 1);
};

struct Digit
{
  o2::dataformats::RangeRefComp<8> ref;

  Triggers mTriggers;               // pattern of triggers  in this BC
  o2::InteractionRecord mIntRecord; // Interaction record (orbit, bc)


  Digit() = default;
  Digit(int first, int ne, o2::InteractionRecord iRec, int64_t chTrig)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    mIntRecord = iRec;
    mTriggers.word = chTrig;
  }
  Digit(int first, int ne, o2::InteractionRecord iRec, Triggers chTrig)
  {
    ref.setFirstEntry(first);
    ref.setEntries(ne);
    mIntRecord = iRec;
    mTriggers = chTrig;
  }

  // ~Digit() = default;


  //  uint32_t getOrbit() const { return o2::InteractionRecord::orbit; }
  uint16_t getBC() const { return mIntRecord.bc; }
  o2::InteractionRecord getIntRecord() { return  mIntRecord;};
  gsl::span<const ChannelData> getBunchChannelData(const gsl::span<const ChannelData> tfdata) const;
  void print() const;

  ClassDefNV(Digit, 1);
};
} // namespace ft0
} // namespace o2

#endif

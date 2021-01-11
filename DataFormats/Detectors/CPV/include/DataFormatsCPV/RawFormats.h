// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CPV_RAWFORMATS_H
#define ALICEO2_CPV_RAWFORMATS_H

namespace o2
{

namespace cpv
{

union PadWord {
  uint32_t mDataWord;
  struct {
    uint32_t charge : 11; ///< Bits  0 - 10 : charge
    uint32_t address : 6; ///< Bits 12 - 17 : address (0..47)
    uint32_t dilogic : 4; ///< Bits 18 - 21 : dilogic (1..10)
    uint32_t row : 6;     ///< Bits 22 - 26 : raw     (1..24)
    uint32_t zero : 1;    ///< Bits 27 - 27 : zeroed so we can distinguish it from the EoE
  };
};

union EoEWord {
  uint32_t mDataWord;
  struct {
    uint32_t nword : 7;    ///< Bits  0 - 6 : word counter (0...47)
    uint32_t en : 11;      ///< Bits 7 - 17 : event number -- not used
    uint32_t dilogic : 4;  ///< Bits 18 - 21 : dilogic (1..10)
    uint32_t row : 6;      ///< Bits 22 - 26 : raw     (1..24)
    uint32_t checkbit : 1; ///< Bits 27 - 27 : bit 27 is always 1 by definition of EoE
  };
};

union SegMarkerWord {
  uint32_t mDataWord;
  struct {
    uint32_t row : 8;     ///< Bits  0 - 7 : segment 0,1,2 charge
    uint32_t nwords : 12; ///< Bits 8 - 19 : number of words in the segment
    uint32_t marker : 12; ///< Bits 20 - 31: ab0 the segment marker word
  };
};

union RowMarkerWord {
  uint32_t mDataWord;
  struct {
    uint32_t marker : 16; ///< Bits  0,15);    //the marker word
    uint32_t nwords : 16; ///< Bits 16 - 31 : number of words written after row marker (digits and EoE)
  };
};

} // namespace cpv

} // namespace o2

#endif

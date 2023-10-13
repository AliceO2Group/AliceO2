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

#ifndef ALICEO2_TRD_PHDATA_H_
#define ALICEO2_TRD_PHDATA_H_

#include <cstdint>
#include "Rtypes.h"

namespace o2::trd
{

/*
  This data type is used to send around the information required to fill PH plots per chamber

  |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
  -------------------------------------------------------------------------------------------------
  |type |nNeighb |   time bin   |        detector number      |     ADC sum for all neigbours     |
  -------------------------------------------------------------------------------------------------
*/

class PHData
{
 public:
  enum Origin : uint8_t {
    ITSTPCTRD,
    TPCTRD,
    TRACKLET,
    OTHER
  };

  PHData() = default;
  PHData(int adc, int det, int tb, int nb, int type) { set(adc, det, tb, nb, type); }

  void set(int adc, int det, int tb, int nb, int type)
  {
    mData = ((type & 0x3) << 30) | ((nb & 0x7) << 27) | ((tb & 0x1f) << 22) | ((det & 0x3ff) << 12) | (adc & 0xfff);
  }

  // the ADC sum for given time bin for up to three neighbours
  int getADC() const { return mData & 0xfff; }
  // the TRD detector number
  int getDetector() const { return (mData >> 12) & 0x3ff; }
  // the given time bin
  int getTimebin() const { return (mData >> 22) & 0x1f; }
  // number of neighbouring digits for which the ADC is accumulated
  int getNneighbours() const { return (mData >> 27) & 0x7; }
  // the origin of this point: digit on ITS-TPC-TRD track, ... (see enum Origin above)
  int getType() const { return (mData >> 30) & 0x3; }

 private:
  uint32_t mData{0}; // see comment above for data content

  ClassDefNV(PHData, 1);
};
} // namespace o2::trd

#endif // ALICEO2_TRD_PHDATA_H_

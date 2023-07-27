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

// #include "TRDBase/GeometryBase.h"
// #include "DetectorsCommonDataFormats/DetMatrixCache.h"
// #include "DetectorsCommonDataFormats/DetID.h"

#ifndef O2_TRD_TRACKLET64_H
#define O2_TRD_TRACKLET64_H

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TRD Raw tracklet                                                       //
// class for TRD tracklets                                                //
//   loosely based on the old TrackletMCM                                 //
//   It still returns the old TrackletWord of run2, rebuilt on calling.   //
// Authors                                                                //
//  Sean Murray (murrays@cern.ch)                                         //
//
////////////////////////////////////////////////////////////////////////////

#ifndef GPUCA_GPUCODE_DEVICE
#include <cstdint>
#endif
#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
#include "GPUCommonMath.h"
#include "DataFormatsTRD/Constants.h"

namespace o2
{
namespace trd
{
/*      |63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47|46|45|44|43|42|41|40|39|38|37|36|35|34|33|32|
        -------------------------------------------------------------------------------------------------
Word 0  |   Format  |              HCID              |  padrow   | col |            position            |
        -------------------------------------------------------------------------------------------------
        |31|30|29|28|27|26|25|24|23|22|21|20|19|18|17|16|15|14|13|12|11|10|09|08|07|06|05|04|03|02|01|00|
        -------------------------------------------------------------------------------------------------
Word 0  |  slope                |    Q2                 |    Q1                 |         Q0            |
        -------------------------------------------------------------------------------------------------

        Note: In the tracklet word wich is sent from the FEE both the position and the slope have one bit inverted.
              This avoids the mis-interpretation of the word as tracklet end marker. The raw reader flips these bits
              back so that the position and slope stored in the Tracklet64 can be used without requiring to invert
              any bit.
*/
class Tracklet64
{

 public:
  GPUdDefault() Tracklet64() = default;
  GPUd() Tracklet64(uint64_t trackletword) { mtrackletWord = trackletword; }
  GPUdDefault() Tracklet64(const Tracklet64&) = default;
  GPUd() Tracklet64(uint64_t format, uint64_t hcid, uint64_t padrow, uint64_t col, uint64_t position,
                    uint64_t slope, uint64_t Q0, uint64_t Q1, uint64_t Q2)
  {
    mtrackletWord = ((format << formatbs) & formatmask) |
                    ((hcid << hcidbs) & hcidmask) |
                    ((padrow << padrowbs) & padrowmask) |
                    ((col << colbs) & colmask) |
                    ((position << posbs) & posmask) |
                    ((slope << slopebs) & slopemask) |
                    ((Q2 << Q2bs) & Q2mask) |
                    ((Q1 << Q1bs) & Q1mask) |
                    ((Q0 << Q0bs) & Q0mask);
  }

  Tracklet64(uint64_t format, uint64_t hcid, uint64_t padrow, uint64_t col, uint64_t position,
             uint64_t slope, uint64_t pid)
  {
    mtrackletWord = ((format << formatbs) & formatmask) |
                    ((hcid << hcidbs) & hcidmask) |
                    ((padrow << padrowbs) & padrowmask) |
                    ((col << colbs) & colmask) |
                    ((position << posbs) & posmask) |
                    ((slope << slopebs) & slopemask) |
                    (pid & PIDmask);
  }

  GPUdDefault() ~Tracklet64() = default;
  GPUdDefault() Tracklet64& operator=(const Tracklet64& rhs) = default;
  GPUd() bool operator==(const Tracklet64& o) const { return mtrackletWord == o.mtrackletWord; }

  // ----- Getters -----
  GPUd() uint64_t getTrackletWord() const { return mtrackletWord; }

  // position / direction information

  // half-chamber number [0..1079]
  GPUd() int getHCID() const { return ((mtrackletWord & hcidmask) >> hcidbs); };

  // chamber number [0..539]
  GPUd() int getDetector() const { return getHCID() / 2; }

  // pad row number on ROB [0..15]
  GPUd() int getPadRow() const { return ((mtrackletWord & padrowmask) >> padrowbs); };

  // MCM position in column direction on ROB [0..3]
  GPUd() int getColumn() const { return ((mtrackletWord & colmask) >> colbs); };

  // in units of 1/40 pads, 11 bit granularity (MSP is used to indicate sign)
  GPUd() int getPosition() const { return ((mtrackletWord & posmask) >> posbs); };

  // applies two's complement to getPosition() to obtain signed value
  GPUd() int getPositionBinSigned() const;

  // the position as float in unit of pads relative to MCM center
  GPUd() float getPositionFloat() const { return getPositionBinSigned() * constants::GRANULARITYTRKLPOS; }

  // in units of 1/128 pads/timebin, 8 bit granularity (MSP is used to indicate sign)
  GPUd() int getSlope() const { return ((mtrackletWord & slopemask) >> slopebs); };

  // applies two's complement to getSlope() to obtain signed value
  GPUd() int getSlopeBinSigned() const;

  // as float in units of pads per time bin
  GPUd() float getSlopeFloat() const { return getSlopeBinSigned() * constants::GRANULARITYTRKLSLOPE / constants::ADDBITSHIFTSLOPE; }

  // MCM position on ROB [0..15]
  GPUd() int getMCM() const { return constants::NMCMROBINROW * (getPadRow() % constants::NMCMROBINCOL) + getColumn(); }

  // ROB number [0..5] for C0 chamber and [0..7] for C1 chamber
  GPUd() int getROB() const { return (getHCID() % 2) ? (getPadRow() / constants::NMCMROBINROW) * 2 + 1 : (getPadRow() / constants::NMCMROBINROW) * 2; }

  // MCM number in pad column direction [0..7]
  GPUd() int getMCMCol() const { return (getMCM() % constants::NMCMROBINCOL) + constants::NMCMROBINCOL * (getROB() % 2); }

  // pad width in cm required for transformation from pad coordinates to tracking coordinates
  GPUd() float getPadWidth() const { return 0.635f + 0.03f * (getDetector() % constants::NLAYER); }

  // pad column number inside pad row as float
  GPUd() float getPadColFloat() const { return getPositionFloat() + getMCMCol() * constants::NCOLMCM + constants::NADCMCM / 2.f; }

  // pad column number inside pad row as int can be off by +-1 pad (same function name as for TRD digit)
  // FIXME: understand why the offset seems to be 8 pads and not nChannels / 2 = 10.5
  GPUd() int getPadCol() const { return GPUCA_NAMESPACE::gpu::CAMath::Nint(getPadColFloat() - 2); }

  // translate local position into global y (in cm) not taking into account calibrations (ExB, vDrift, t0)
  GPUd() float getUncalibratedY() const
  {
    // one pad column has 144 pads, the offset of -63 is the center of the first MCM in that column
    // which is connected to the pads -63 - 9 = -72 to -63 + 9 = -54
    int offsetInNumberOfPads = -63 + constants::NCOLMCM * getMCMCol();
    return (offsetInNumberOfPads + getPositionFloat()) * getPadWidth();
  }

  // translate local slope into dy/dx with dx=3m (drift length) and default drift time in time bins (19.4 timebins / 3cm)
  GPUd() float getUncalibratedDy(float nTbDrift = 19.4f) const { return getSlopeFloat() * getPadWidth() * nTbDrift; }

  // PID related getters
  GPUd() int getFormat() const { return ((mtrackletWord & formatmask) >> formatbs); };
  GPUd() int getPID() const { return ((mtrackletWord & PIDmask)); };
  GPUd() int getDynamicCharge(unsigned int charge) const
  {
    int shift = (charge >> 6) & 0x3;
    if (shift == 0) {
      shift = 8;
    } else {
      shift = shift << 1;
    }
    charge = charge << shift;
    return charge;
  }; // no unit
  GPUd() int getQ0() const
  {
    if ((getFormat() & 0x1) == 0) {
      return ((mtrackletWord & Q0mask) >> Q0bs);
    } else {
      return getDynamicCharge((mtrackletWord & Q0mask) >> Q0bs);
    }
  }; // no unit
  GPUd() int getQ1() const
  {
    if ((getFormat() & 0x1) == 0) {
      return ((mtrackletWord & Q1mask) >> Q1bs);
    } else {
      return getDynamicCharge((mtrackletWord & Q1mask) >> Q1bs);
    }
  }; // no unit
  GPUd() int getQ2() const
  {
    if ((getFormat() & 0x1) == 0) {
      return ((mtrackletWord & Q2mask) >> Q2bs);
    } else {
      return getDynamicCharge((mtrackletWord & Q2mask) >> Q2bs);
    }
  }; // no unit

  // ----- Setters for tracklet word manipulation -----
  GPUd() void setTrackletWord(uint64_t trackletword) { mtrackletWord = trackletword; }
  GPUd() void setQ0(int charge) { mtrackletWord = (mtrackletWord & ~Q0mask) | ((charge & Q0mask) << Q0bs); }
  GPUd() void setQ1(int charge) { mtrackletWord = (mtrackletWord & ~Q1mask) | ((charge & Q1mask) << Q1bs); }
  GPUd() void setQ2(int charge) { mtrackletWord = (mtrackletWord & ~Q2mask) | ((charge & Q2mask) << Q2bs); }
  GPUd() void setPID(int pid) { mtrackletWord = (mtrackletWord & ~PIDmask) | ((pid & PIDmask) << PIDbs); }
  GPUd() void setPosition(int position) { mtrackletWord = (mtrackletWord & ~posmask) | ((position & posmask) << posbs); }
  GPUd() void setSlope(int slope) { mtrackletWord = (mtrackletWord & ~slopemask) | ((slope & slopemask) << slopebs); }

  GPUd() void print() const;
#ifndef GPUCA_GPUCODE_DEVICE
  void printStream(std::ostream& stream) const;
#endif // GPUCA_GPUCODE_DEVICE

  // bit masks for the above raw data;
  static constexpr uint64_t formatmask = 0xf000000000000000;
  static constexpr uint64_t hcidmask = 0x0ffe000000000000;
  static constexpr uint64_t padrowmask = 0x0001e00000000000;
  static constexpr uint64_t colmask = 0x0000180000000000;
  static constexpr uint64_t posmask = 0x000007ff00000000;
  static constexpr uint64_t slopemask = 0x00000000ff000000;
  static constexpr uint64_t Q2mask = 0x0000000000ff0000;
  static constexpr uint64_t Q1mask = 0x000000000000ff00;
  static constexpr uint64_t Q0mask = 0x00000000000000ff;
  static constexpr uint64_t PIDmask = 0x0000000000ffffff;
  // bit shifts for the above raw data
  static constexpr uint64_t formatbs = 60;
  static constexpr uint64_t hcidbs = 49;
  static constexpr uint64_t padrowbs = 45;
  static constexpr uint64_t colbs = 43;
  static constexpr uint64_t posbs = 32;
  static constexpr uint64_t slopebs = 24;
  static constexpr uint64_t PIDbs = 0;
  static constexpr uint64_t Q2bs = 16;
  static constexpr uint64_t Q1bs = 8;
  static constexpr uint64_t Q0bs = 0;

 protected:
  uint64_t mtrackletWord; // the 64 bit word holding all the tracklet information for run3.
 private:
  ClassDefNV(Tracklet64, 2);
};

GPUdi() int Tracklet64::getPositionBinSigned() const
{
  int padLocalBin = getPosition();
  int padLocal = 0;
  if (padLocalBin & (1 << (constants::NBITSTRKLPOS - 1))) {
    padLocal = -((~(padLocalBin - 1)) & ((1 << constants::NBITSTRKLPOS) - 1));
  } else {
    padLocal = padLocalBin & ((1 << constants::NBITSTRKLPOS) - 1);
  }
  return padLocal;
}

GPUdi() int Tracklet64::getSlopeBinSigned() const
{
  int slopeBin = getSlope();
  int slope = 0;
  if (slopeBin & (1 << (constants::NBITSTRKLSLOPE - 1))) {
    slope = -((~(slopeBin - 1)) & ((1 << constants::NBITSTRKLSLOPE) - 1));
  } else {
    slope = slopeBin & ((1 << constants::NBITSTRKLSLOPE) - 1);
  }
  return -slope;
}

#ifndef GPUCA_GPUCODE_DEVICE
std::ostream& operator<<(std::ostream& stream, const Tracklet64& trg);
#endif // GPUCA_GPUCODE_DEVICE

} // namespace trd
} // namespace o2

#endif

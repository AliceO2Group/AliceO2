// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//#include "TRDBase/GeometryBase.h"
//#include "DetectorsCommonDataFormats/DetMatrixCache.h"
//#include "DetectorsCommonDataFormats/DetID.h"

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

#include "GPUCommonDef.h"
#include "GPUCommonRtypes.h"
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

  // ----- Getters for contents of tracklet word -----
  GPUd() uint64_t getFormat() const { return ((mtrackletWord & formatmask) >> formatbs); }; // no units 0..1077
  GPUd() uint64_t getHCID() const { return ((mtrackletWord & hcidmask) >> hcidbs); };       // no units 0..1077
  GPUd() uint64_t getPadRow() const { return ((mtrackletWord & padrowmask) >> padrowbs); }; // pad row number [0..15]
  GPUd() uint64_t getColumn() const { return ((mtrackletWord & colmask) >> colbs); };       // column refers to MCM position in column direction on readout board [0..3]
  GPUd() uint64_t getPosition() const { return ((mtrackletWord & posmask) >> posbs); };     // in units of 1/80 pads, 11 bit granularity [-12.8..12.8] relative to MCM center
  GPUd() uint64_t getSlope() const { return ((mtrackletWord & slopemask) >> slopebs); };    // in units of 1/1000 pads/timebin, 8 bit granularity [-0.128 to 0.128]
  GPUd() uint64_t getPID() const { return ((mtrackletWord & PIDmask)); };                   // no unit, all 3 charge windows combined
  GPUd() uint64_t getQ0() const { return ((mtrackletWord & Q0mask) >> Q0bs); };             // no unit
  GPUd() uint64_t getQ1() const { return ((mtrackletWord & Q1mask) >> Q1bs); };             // no unit
  GPUd() uint64_t getQ2() const { return ((mtrackletWord & Q2mask) >> Q2bs); };             // no unit

  GPUd() void setTrackletWord(uint64_t trackletword) { mtrackletWord = trackletword; }

  // ----- Getters for tracklet information -----
  GPUd() int getMCM() const { return 4 * (getPadRow() % 4) + getColumn(); }                                 // returns MCM position on ROB [0..15]
  GPUd() int getROB() const { return (getHCID() % 2) ? (getPadRow() / 4) * 2 + 1 : (getPadRow() / 4) * 2; } // returns ROB number [0..5] for C0 chamber and [0..7] for C1 chamber
  GPUd() float getUncalibratedY() const;                                                                    // translate local position into global y (in cm) not taking into account calibrations (ExB, vDrift, t0)
  GPUd() float getUncalibratedDy(float nTbDrift = 19.4f) const;                                             // translate local slope into dy/dx with dx=3m (drift length) and default drift time in time bins (19.4 timebins / 3cm)

  // ----- Getters for offline corresponding values -----
  GPUd() int getDetector() const { return getHCID() / 2; }

  GPUd() uint64_t getTrackletWord() const { return mtrackletWord; }

  GPUd() void setQ0(int charge)
  {
    mtrackletWord &= ~Q0mask;
    mtrackletWord |= ((charge << Q0bs) & Q0mask);
  }
  GPUd() void setQ1(int charge)
  {
    mtrackletWord &= ~Q1mask;
    mtrackletWord |= ((charge << Q1bs) & Q1mask);
  }
  GPUd() void setQ2(int charge)
  {
    mtrackletWord &= ~Q2mask;
    mtrackletWord |= ((charge << Q2bs) & Q2mask);
  }
  GPUd() void setPID(uint64_t pid)
  {
    // set the entire pid area of the trackletword, all the 3 Q's
    mtrackletWord &= ~PIDmask;
    mtrackletWord |= ((pid << PIDbs) & PIDmask);
  }
  GPUd() void setPosition(uint64_t position)
  {
    mtrackletWord &= ~posmask;
    mtrackletWord |= ((position << posbs) & posmask);
  }
  GPUd() void setSlope(uint64_t slope)
  {
    mtrackletWord &= ~slopemask;
    mtrackletWord |= ((slope << slopebs) & slopemask);
  }

  GPUd() bool operator==(const Tracklet64& o) const { return mtrackletWord == o.mtrackletWord; }

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
  //bit shifts for the above raw data
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
  ClassDefNV(Tracklet64, 1);
};

GPUdi() float Tracklet64::getUncalibratedY() const
{
  int padLocalBin = getPosition();
  int padLocal = 0;
  if (padLocalBin & (1 << (constants::NBITSTRKLPOS - 1))) {
    padLocal = -((~(padLocalBin - 1)) & ((1 << constants::NBITSTRKLPOS) - 1));
  } else {
    padLocal = padLocalBin & ((1 << constants::NBITSTRKLPOS) - 1);
  }
  int mcmCol = (getMCM() % constants::NMCMROBINCOL) + constants::NMCMROBINCOL * (getROB() % 2);
  float offset = -63.f + ((float)constants::NCOLMCM) * mcmCol;
  float padWidth = 0.635f + 0.03f * (getDetector() % constants::NLAYER);
  return (offset + padLocal * constants::GRANULARITYTRKLPOS) * padWidth;
}

GPUdi() float Tracklet64::getUncalibratedDy(float nTbDrift) const
{
  float dy;
  int dyLocalBin = getSlope();
  if (dyLocalBin & (1 << (constants::NBITSTRKLSLOPE - 1))) {
    dy = (~(dyLocalBin - 1)) & ((1 << constants::NBITSTRKLSLOPE) - 1);
    dy *= -1.f;
  } else {
    dy = dyLocalBin & ((1 << constants::NBITSTRKLSLOPE) - 1);
  }
  float padWidth = 0.635f + 0.03f * (getDetector() % constants::NLAYER);
  return dy * constants::GRANULARITYTRKLSLOPE * padWidth * nTbDrift;
}

#ifndef GPUCA_GPUCODE_DEVICE
std::ostream& operator<<(std::ostream& stream, const Tracklet64& trg);
#endif // GPUCA_GPUCODE_DEVICE

} //namespace trd
} //namespace o2

#endif

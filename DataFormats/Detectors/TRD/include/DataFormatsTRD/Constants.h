// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Constants.h
/// \brief Global TRD definitions and constants
/// \author ole.schmidt@cern.ch

#ifndef AliceO2_TRD_Constants_H
#define AliceO2_TRD_Constants_H

namespace o2
{
namespace trd
{
namespace constants
{
constexpr int NSECTOR = 18;        // the number of sectors
constexpr int NSTACK = 5;          // the number of stacks per sector
constexpr int NLAYER = 6;          // the number of layers
constexpr int NCHAMBERPERSEC = 30; // the number of chambers per sector
constexpr int MAXCHAMBER = 540;    // the maximum number of installed chambers
constexpr int NCHAMBER = 521;      // the number of chambers actually installed
constexpr int NHALFCRU = 72;       // the number of half cru (link bundles)
constexpr int NLINKSPERHALFCRU = 15; // the number of links per half cru or cru end point.
constexpr int NRU = 36;              // the number of CRU we have
constexpr int NFLP = 12;             // the number of FLP we have.
constexpr int NCRUPERFLP = 3;        // the number of CRU per FLP
constexpr int TRDLINKID = 15;        // hard coded link id, specific to TRD

constexpr int NCOLUMN = 144; // the number of pad columns for each chamber
constexpr int NROWC0 = 12;   // the number of pad rows for chambers of type C0 (installed stack 0,1,3 and 4)
constexpr int NROWC1 = 16;   // the number of pad rows for chambers of type C1 (installed in stack 2)

constexpr int NMCMROB = 16;     // the number of MCMs per ROB
constexpr int NMCMHCMAX = 64;   // the maximum number of MCMs for one half chamber (C1 type)
constexpr int NMCMROBINROW = 4; // the number of MCMs per ROB in row direction
constexpr int NMCMROBINCOL = 4; // the number of MCMs per ROB in column direction
constexpr int NROBC0 = 6;       // the number of ROBs per C0 chamber
constexpr int NROBC1 = 8;       // the number of ROBs per C1 chamber
constexpr int NADCMCM = 21;     // the number of ADC channels per MCM
constexpr int NCOLMCM = 18;     // the number of pads per MCM

constexpr int NBITSTRKLPOS = 11;                   // number of bits for position in tracklet64 word
constexpr int NBITSTRKLSLOPE = 8;                  // number of bits for slope in tracklet64 word
constexpr float PADGRANULARITYTRKLPOS = 80.f;
constexpr float PADGRANULARITYTRKLSLOPE = 1000.f;
constexpr float GRANULARITYTRKLPOS = 1.f / PADGRANULARITYTRKLPOS;     // granularity of position in tracklet64 word in pad-widths
constexpr float GRANULARITYTRKLSLOPE = 1.f / PADGRANULARITYTRKLSLOPE; // granularity of slope in tracklet64 word in pads/timebin

// OS: Should this not be flexible for example in case of Kr calib?
constexpr int TIMEBINS = 30; // the number of time bins
constexpr float MAXIMPACTANGLE = 25.f; // the maximum impact angle for tracks relative to the TRD detector plane to be considered for vDrift and ExB calibration
constexpr int NBINSANGLEDIFF = 25;     // the number of bins for the track angle used for the vDrift and ExB calibration based on the tracking

// Trigger parameters
constexpr double READOUT_TIME = 3000;                  // the time the readout takes, as 30 TB = 3 micro-s.
constexpr double DEAD_TIME = 200;                      // trigger deadtime, 2 micro-s
constexpr double BUSY_TIME = READOUT_TIME + DEAD_TIME; // the time for which no new trigger can be received in nanoseconds

// array size to store incoming half cru payload.
constexpr int HBFBUFFERMAX = 1048576;         // max buffer size for data read from a half cru, (all events)
constexpr int CRUPADDING32 = 0xeeeeeeee;      // padding word used in the cru.
constexpr int TRACKLETENDMARKER = 0x10001000; // marker for the end of tracklets in raw data, 2 of these.
constexpr int DIGITENDMARKER = 0x0;           // marker for the end of digits in raw data, 2 of these
constexpr int MAXDATAPERLINK32 = 13824;       // max number of 32 bit words per link ((21x12+2+4)*64) 64 mcm, 21 channels, 10 words per channel 2 header words(DigitMCMHeader DigitMCMADCmask) 4 words for tracklets.
constexpr int MAXDATAPERLINK256 = 1728;       // max number of linkwords per cru link. (256bit words)

} //namespace constants
} // namespace trd
} // namespace o2

#endif

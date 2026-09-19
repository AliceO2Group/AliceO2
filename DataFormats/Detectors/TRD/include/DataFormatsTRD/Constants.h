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

/// \file Constants.h
/// \brief Global TRD definitions and constants
/// \author ole.schmidt@cern.ch

#ifndef AliceO2_TRD_Constants_H
#define AliceO2_TRD_Constants_H

#include "GPUCommonDef.h"

namespace o2
{
namespace trd
{
namespace constants
{
GPUglobalconstexpr() int NSECTOR = 18;          ///< the number of sectors
GPUglobalconstexpr() int NSTACK = 5;            ///< the number of stacks per sector
GPUglobalconstexpr() int NLAYER = 6;            ///< the number of layers
GPUglobalconstexpr() int NCHAMBERPERSEC = 30;   ///< the number of chambers per sector
GPUglobalconstexpr() int NHCPERSEC = 60;        ///< the number of half-chambers per sector
GPUglobalconstexpr() int MAXCHAMBER = 540;      ///< the maximum number of installed chambers
GPUglobalconstexpr() int MAXHALFCHAMBER = 1080; ///< the maximum number of installed half-chambers
GPUglobalconstexpr() int NCHAMBER = 521;        ///< the number of chambers actually installed
GPUglobalconstexpr() int NHALFCRU = 72;         ///< the number of half cru (link bundles)
GPUglobalconstexpr() int NLINKSPERHALFCRU = 15; ///< the number of links per half cru or cru end point.
GPUglobalconstexpr() int NLINKSPERCRU = 30;     ///< the number of links per CRU (two CRUs serve one supermodule)
GPUglobalconstexpr() int NCRU = 36;             ///< the number of CRU we have
GPUglobalconstexpr() int NFLP = 12;             ///< the number of FLP we have.
GPUglobalconstexpr() int NCRUPERFLP = 3;        ///< the number of CRU per FLP
GPUglobalconstexpr() int TRDLINKID = 15;        ///< hard coded link id, specific to TRD

GPUglobalconstexpr() int NCOLUMN = 144; ///< the number of pad columns for each chamber
GPUglobalconstexpr() int NROWC0 = 12;   ///< the number of pad rows for chambers of type C0 (installed in stack 2)
GPUglobalconstexpr() int NROWC1 = 16;   ///< the number of pad rows for chambers of type C1 (installed in stacks 0, 1, 3 and 4)
GPUglobalconstexpr() int FIRSTROW[NSTACK] = {0, 16, 32, 44, 60}; ///< first pad row for each stack

GPUglobalconstexpr() int NMCMROB = 16;     ///< the number of MCMs per ROB
GPUglobalconstexpr() int NMCMHCMAX = 64;   ///< the maximum number of MCMs for one half chamber (C1 type)
GPUglobalconstexpr() int NMCMROBINROW = 4; ///< the number of MCMs per ROB in row direction
GPUglobalconstexpr() int NMCMROBINCOL = 4; ///< the number of MCMs per ROB in column direction
GPUglobalconstexpr() int NROBC0 = 6;       ///< the number of ROBs per C0 chamber
GPUglobalconstexpr() int NROBC1 = 8;       ///< the number of ROBs per C1 chamber
GPUglobalconstexpr() int NADCMCM = 21;     ///< the number of ADC channels per MCM
GPUglobalconstexpr() int NCOLMCM = 18;     ///< the number of pads per MCM
GPUglobalconstexpr() int NCHANNELSPERROW = NMCMROBINCOL * 2 * NADCMCM;                                                    ///< the number of readout channels per pad row
GPUglobalconstexpr() int NCHANNELSC0 = NROWC0 * NCHANNELSPERROW;                                                          ///< the number of readout channels per C0 chamber
GPUglobalconstexpr() int NCHANNELSC1 = NROWC1 * NCHANNELSPERROW;                                                          ///< the number of readout channels per C1 chamber
GPUglobalconstexpr() int NCHANNELSTOTAL = NSECTOR * NLAYER * (NSTACK - 1) * NCHANNELSC1 + NSECTOR * NLAYER * NCHANNELSC0; ///< the total number of readout channels for TRD
GPUglobalconstexpr() int NCHANNELSPERSECTOR = NCHANNELSTOTAL / NSECTOR;                                                   ///< then number of readout channels per sector
GPUglobalconstexpr() int NCHANNELSPERLAYER = NCHANNELSPERSECTOR / NLAYER;                                                 ///< then number of readout channels per layer
GPUglobalconstexpr() int NCPU = 4;         ///< the number of CPUs inside the TRAP chip
GPUglobalconstexpr() int NCHARGES = 3;     ///< the number of charges per tracklet (Q0/1/2)

// the values below should come out of the TRAP config in the future
GPUglobalconstexpr() int NBITSTRKLPOS = 11;                                      ///< number of bits for position in tracklet64 word
GPUglobalconstexpr() int NBITSTRKLSLOPE = 8;                                     ///< number of bits for slope in tracklet64 word
GPUglobalconstexpr() int ADDBITSHIFTSLOPE = 1 << 3;                              ///< in the TRAP the slope is shifted by 3 additional bits compared to the position
GPUglobalconstexpr() int PADGRANULARITYTRKLPOS = 40;                             ///< tracklet position is stored in units of 1/40 pad
GPUglobalconstexpr() int PADGRANULARITYTRKLSLOPE = 128;                          ///< tracklet deflection is stored in units of 1/128 pad per time bin
GPUglobalconstexpr() float GRANULARITYTRKLPOS = 1.f / PADGRANULARITYTRKLPOS;     ///< granularity of position in tracklet64 word in pad-widths
GPUglobalconstexpr() float GRANULARITYTRKLSLOPE = 1.f / PADGRANULARITYTRKLSLOPE; ///< granularity of slope in tracklet64 word in pads/timebin
GPUglobalconstexpr() int ADCBASELINE = 10;                                       ///< baseline in ADC units

// OS: Should this not be flexible for example in case of Kr calib?
GPUglobalconstexpr() int TIMEBINS = 30;            ///< the number of time bins
GPUglobalconstexpr() float MAXIMPACTANGLE = 25.f;  ///< the maximum impact angle for tracks relative to the TRD detector plane to be considered for vDrift and ExB calibration
GPUglobalconstexpr() int NBINSANGLEDIFF = 25;      ///< the number of bins for the track angle used for the vDrift and ExB calibration based on the tracking
#ifndef GPUCA_GPUCODE_DEVICE
// calibration defaults, host only: these are double and never used in device code
constexpr double VDRIFTDEFAULT = 1.546; ///< default value for vDrift
constexpr double VDRIFTMIN = 0.4;       ///< min value for vDrift
constexpr double VDRIFTMAX = 2.0;       ///< max value for vDrift
constexpr double EXBDEFAULT = 0.0;      ///< default value for LorentzAngle
constexpr double EXBMIN = -0.4;         ///< min value for LorentzAngle
constexpr double EXBMAX = 0.4;          ///< max value for LorentzAngle
#endif
GPUglobalconstexpr() int NBINSGAINCALIB = 320;     ///< number of bins in the charge (Q0+Q1+Q2) histogram for gain calibration
GPUglobalconstexpr() float MPVDEDXDEFAULT = 42.;   ///< default Most Probable Value of TRD dEdx
GPUglobalconstexpr() float T0DEFAULT = 1.2;        ///< default value for t0

// array size to store incoming half cru payload.
GPUglobalconstexpr() int HBFBUFFERMAX = 1048576;                 ///< max buffer size for data read from a half cru, (all events)
GPUglobalconstexpr() unsigned int CRUPADDING32 = 0xeeeeeeee;     ///< padding word used in the cru.
GPUglobalconstexpr() int CHANNELNRNOTRKLT = 23;                  ///< this marks channels in the ADC mask which don't contribute to a tracklet
GPUglobalconstexpr() int NOTRACKLETFIT = 31;                     ///< this value is assigned to the fit pointer in case no tracklet is available
GPUglobalconstexpr() int TRACKLETENDMARKER = 0x10001000;         ///< marker for the end of tracklets in raw data, 2 of these.
GPUglobalconstexpr() int PADDINGWORD = 0xeeeeeeee;               ///< half-CRU links will be padded with this words to get an even number of 256bit words
GPUglobalconstexpr() int DIGITENDMARKER = 0x0;                   ///< marker for the end of digits in raw data, 2 of these
GPUglobalconstexpr() int MAXDATAPERLINK32 = 13824;               ///< max number of 32 bit words per link ((21x12+2+4)*64) 64 mcm, 21 channels, 10 words per channel 2 header words(DigitMCMHeader DigitMCMADCmask) 4 words for tracklets.
GPUglobalconstexpr() int MAXDATAPERLINK256 = 1728;               ///< max number of linkwords per cru link. (256bit words)
GPUglobalconstexpr() int MAXEVENTCOUNTERSEPERATION = 200;        ///< how far apart can subsequent mcmheader event counters be before we flag for concern, used as a sanity check in rawreader.
GPUglobalconstexpr() int MAXMCMCOUNT = 69120;                    ///< at most mcm count maxchamber x nrobc1 nmcmrob
GPUglobalconstexpr() int MAXLINKERRORHISTOGRAMS = 10;            ///< size of the array holding the link error plots from the raw reader
GPUglobalconstexpr() int MAXPARSEERRORHISTOGRAMS = 60;           ///< size of the array holding the parsing error plots from the raw reader
GPUglobalconstexpr() unsigned int ETYPEPHYSICSTRIGGER = 0x2;     ///< CRU Half Chamber header eventtype definition
GPUglobalconstexpr() unsigned int ETYPECALIBRATIONTRIGGER = 0x3; ///< CRU Half Chamber header eventtype definition
GPUglobalconstexpr() int MAXCRUERRORVALUE = 0x2;                 ///< Max possible value for a CRU Halfchamber link error. As of may 2022, can only be 0x0, 0x1, and 0x2, at least that is all so far(may2022).
GPUglobalconstexpr() int INVALIDPRETRIGGERPHASE = 0xf;           ///< Invalid value for phase, used to signify there is no hcheader.

} // namespace constants
} // namespace trd
} // namespace o2

#endif

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

#ifndef ALICEO2_TOF_GEO_H
#define ALICEO2_TOF_GEO_H

#include "Rtypes.h"
#include <array>
#include <vector>
#include "CommonConstants/LHCConstants.h"
//#include "DetectorsRaw/HBFUtils.h"

namespace o2
{
namespace tof
{
/// \class Geo
/// \brief TOF geo parameters (only statics)

class Geo
{
 public:
  //  static void updateNSinTF() { NS_IN_TF = o2::constants::lhc::LHCOrbitNS * o2::raw::HBFUtils::getNOrbitsPerTF(); }

  // FLP <-> CRU <-> LINKS mapping
  static Int_t getCRU(int link)
  {
    return CRUFROMLINK[link];
  }
  static Int_t getCRUlink(int link)
  {
    return CRULINK[link];
  }
  static Int_t getCRUendpoint(int link)
  {
    return CRUENDPOINT[link];
  }
  static Int_t getCONETlink(int link) { return (link % 4); }
  static Int_t getCRUid(int link)
  {
    return CRUID[CRUFROMLINK[link]];
  }
  static Int_t getCRUid(int iflp, int icru) { return CRUFROMFLP[iflp][icru]; }
  static Int_t getFLPid(int link)
  {
    return FLPFROMCRU[CRUFROMLINK[link]];
  }
  static Int_t getFEEid(int link)
  {
    return FEEID[link];
  }
  static Int_t getFLP(int iflp) { return FLP[iflp]; }

  // From AliTOFGeometry
  static void translate(Float_t* xyz, Float_t translationVector[3]);
  static void translate(Float_t& x, Float_t& y, Float_t& z, Float_t translationVector[3]);
  static void rotate(Float_t* xyz, Double_t rotationAngles[6]);

  static void rotateToSector(Float_t* xyz, Int_t isector);
  static void rotateToStrip(Float_t* xyz, Int_t iplate, Int_t istrip, Int_t isector);
  static void antiRotateToSector(Float_t* xyz, Int_t isector);
  static void antiRotateToStrip(Float_t* xyz, Int_t iplate, Int_t istrip, Int_t isector);

  static void antiRotate(Float_t* xyz, Double_t rotationAngles[6]);
  static void getDetID(Float_t* pos, Int_t* det);
  static Int_t getIndex(const Int_t* detId);               // Get channel index from det Id (for calibration mainly)
  static void getVolumeIndices(Int_t index, Int_t* detId); // Get volume index from channel index

  static void getPos(Int_t* det, Float_t* pos);
  static std::string getVolumePath(const Int_t* ind);
  static Int_t getStripNumberPerSM(Int_t iplate, Int_t istrip);
  static void getStripAndModule(Int_t iStripPerSM, Int_t& iplate, Int_t& istrip); // Return the module and strip per module corresponding to the strip number per SM

  static Float_t getAngles(Int_t iplate, Int_t istrip) { return ANGLES[iplate][istrip]; }
  static Float_t getHeights(Int_t iplate, Int_t istrip) { return HEIGHTS[iplate][istrip]; }
  static Float_t getDistances(Int_t iplate, Int_t istrip) { return DISTANCES[iplate][istrip]; }
  static Float_t getGeoHeights(Int_t isector, Int_t iplate, Int_t istrip) { return mGeoHeights[isector][iplate][istrip]; }
  static Float_t getGeoX(Int_t isector, Int_t iplate, Int_t istrip) { return mGeoX[isector][iplate][istrip]; }
  static Float_t getGeoDistances(Int_t isector, Int_t iplate, Int_t istrip) { return mGeoDistances[isector][iplate][istrip]; }

  static void getPadDxDyDz(const Float_t* pos, Int_t* det, Float_t* DeltaPos, int sector = -1);
  enum {
    // DAQ characteristics
    // cfr. TOF-TDR pag. 105 for Glossary
    // TARODA : TOF-ALICE Read Out and Data Acquisition system
    kNDDL = 4,    // Number of DDL (Detector Data Link) per sector
    kNTRM = 12,   // Number of TRM ( Readout Module) per DDL
    kNTdc = 15,   // Number of Tdc (Time to Digital Converter) per TRM
    kNChain = 2,  // Number of chains per TRM
    kNCrate = 72, // Number of Crates
    kNCh = 8      // Number of channels per Tdc
  };

  static constexpr Int_t RAW_PAGE_MAX_SIZE = 8192;

  static constexpr Double_t BC_TIME = o2::constants::lhc::LHCBunchSpacingNS; // bunch crossing in ns
  static constexpr Double_t BC_TIME_INV = 1. / BC_TIME;                      // inv bunch crossing in ns
  static constexpr Double_t BC_TIME_INPS = BC_TIME * 1000;                   // bunch crossing in ps
  static constexpr Double_t BC_TIME_INPS_INV = 1. / BC_TIME_INPS;            // inv bunch crossing in ps
  static constexpr int BC_IN_ORBIT = o2::constants::lhc::LHCMaxBunches;     // N. bunch crossing in 1 orbit

  static constexpr Int_t NPADX = 48;
  static constexpr Float_t NPADX_INV_INT = 1. / NPADX;
  static constexpr Int_t NPADZ = 2;
  static constexpr Int_t NPADS = NPADX * NPADZ;
  static constexpr Float_t NPADS_INV_INT = 1. / NPADS;
  static constexpr Int_t NSTRIPA = 15;
  static constexpr Int_t NSTRIPB = 19;
  static constexpr Int_t NSTRIPC = 19;
  static constexpr Int_t NMAXNSTRIP = 20;
  static constexpr Int_t NSTRIPXSECTOR = NSTRIPA + 2 * NSTRIPB + 2 * NSTRIPC;
  static constexpr Float_t NSTRIPXSECTOR_INV_INT = 1. / NSTRIPXSECTOR;
  static constexpr Int_t NPADSXSECTOR = NSTRIPXSECTOR * NPADS;

  static constexpr Int_t NSECTORS = 18;
  static constexpr Int_t NSTRIPS = NSECTORS * NSTRIPXSECTOR;
  static constexpr Int_t NPLATES = 5;

  static constexpr int NCHANNELS = NSTRIPS * NPADS;
  static constexpr int N_ELECTRONIC_CHANNELS = 72 << 12;

  static constexpr Float_t MAXHZTOF = 370.6;      // Max half z-size of TOF (cm)
  static constexpr Float_t ZLENA = MAXHZTOF * 2.; // length (cm) of the A module
  static constexpr Float_t ZLENB = 146.5;         // length (cm) of the B module
  static constexpr Float_t ZLENC = 170.45;        // length (cm) of the C module

  static constexpr Float_t XTOF = 372.00; // Inner radius of the TOF for Reconstruction (cm)
  static constexpr Float_t RMIN = 371;
  static constexpr Float_t RMAX = 400.05;

  static constexpr Float_t RMIN2 = RMIN * RMIN;
  static constexpr Float_t RMAX2 = RMAX * RMAX;

  static constexpr Float_t XPAD = 2.5;
  static constexpr Float_t XHALFSTRIP = XPAD * NPADX * 0.5;
  static constexpr Float_t ZPAD = 3.5;
  static constexpr Float_t STRIPLENGTH = 122;

  static constexpr Float_t SIGMAFORTAIL1 = 2.;   // Sig1 for simulation of TDC tails
  static constexpr Float_t SIGMAFORTAIL12 = 0.5; // Sig2 for simulation of TDC tails

  static constexpr Float_t PHISEC = 20; // sector Phi width (deg)
  static constexpr Float_t PHISECINV = 1. / PHISEC; // sector Phi width (deg)

  static constexpr Float_t TDCBIN = o2::constants::lhc::LHCBunchSpacingNS * 1E3 / 1024; ///< TDC bin width [ps]
  static constexpr Float_t NTDCBIN_PER_PS = 1. / TDCBIN;     ///< number of TDC bins in 1 ns
  static constexpr Int_t RATIO_TOT_TDC_BIN = 2;              // ratio between TDC and TOT bin sizes
  static constexpr Float_t TOTBIN = TDCBIN * RATIO_TOT_TDC_BIN; // time-over-threshold bin width [ps]
  static constexpr Float_t TOTBIN_NS = TOTBIN * 1E-3;        // time-over-threshold bin width [ns]
  static constexpr Float_t NTOTBIN_PER_NS = 1000. / TOTBIN;  // number of time-over-threshold bin in 1 ns
  static constexpr Float_t BUNCHCROSSINGBIN = TDCBIN * 1024; // bunch-crossing bin width [ps]

  static constexpr Float_t SLEWTOTMIN = 10.; // min TOT for slewing correction [ns]
  static constexpr Float_t SLEWTOTMAX = 16.; // max TOT for slewing correction [ns]

  static constexpr Float_t DEADTIME = 25E+03;               // Single channel dead time (ps)
  static constexpr Float_t DEADTIMETDC = DEADTIME / TDCBIN; ///< Single channel TDC dead time (ps)
  static constexpr int NWINDOW_IN_ORBIT = 3;                //< Number of tof window in 1 orbit
  static constexpr Double_t READOUTWINDOW = o2::constants::lhc::LHCOrbitNS / NWINDOW_IN_ORBIT; // Readout window (ns) - time between two consecutive triggers = 1/3 orbit
  static constexpr int BC_IN_WINDOW = BC_IN_ORBIT / NWINDOW_IN_ORBIT;                         // N. bunch crossing in 1 tof window
  static constexpr double BC_IN_WINDOW_INV = 1. / BC_IN_WINDOW;
  static constexpr Double_t READOUTWINDOW_INV = 1. / READOUTWINDOW;                           // Readout window (ns)

  static constexpr Int_t READOUTWINDOW_IN_BC = BC_IN_ORBIT / NWINDOW_IN_ORBIT;                             // round down in case
  static constexpr Int_t LATENCYWINDOW_TOF = 1196;                                                         // Latency window  in BC (larger than 1/3 orbit 1188 BC)
  static constexpr Int_t LATENCY_ADJUSTEMENT_IN_BC = -114;                                                 // Latency adj wrt LHC (to compare with fill scheme)
  static constexpr Int_t LATENCYWINDOW_IN_BC = LATENCYWINDOW_TOF + LATENCY_ADJUSTEMENT_IN_BC;              // Latency window  in BC (larger than 1/3 orbit 1188 BC)
  static constexpr Int_t LATENCY_ADJ_LHC_IN_BC = 0;                                                        // Latency adj wrt LHC (to compare with fill scheme) -> should be zero!
  static constexpr Int_t MATCHINGWINDOW_IN_BC = 1200;                                                      // Latency window  in BC (larger than 1/3 orbit 1188 BC)
  static constexpr Int_t OVERLAP_IN_BC = MATCHINGWINDOW_IN_BC - READOUTWINDOW_IN_BC;                       // overlap between two readout window in BC
  static constexpr Double_t LATENCYWINDOW = LATENCYWINDOW_IN_BC * o2::constants::lhc::LHCBunchSpacingNS;   // Latency window  in ns
  static constexpr Double_t MATCHINGWINDOW = MATCHINGWINDOW_IN_BC * o2::constants::lhc::LHCBunchSpacingNS; // Matching window  in ns
  static constexpr Double_t WINDOWOVERLAP = MATCHINGWINDOW - READOUTWINDOW;                                // overlap between two consecutive matchingwindow

  static constexpr Int_t CRUFROMLINK[kNCrate] = {
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};

  static constexpr Int_t FEEID[kNCrate] = {
    327680, 327681, 327682, 327683, 327684, 327685, 327686, 327687, 327688, 327689, 327690, 327691, 327692, 327693, 327694, 327695, 327696, 327697,
    327698, 327699, 327700, 327701, 327702, 327703, 327704, 327705, 327706, 327707, 327708, 327709, 327710, 327711, 327712, 327713, 327714, 327715,
    327716, 327717, 327718, 327719, 327720, 327721, 327722, 327723, 327724, 327725, 327726, 327727, 327728, 327729, 327730, 327731, 327732, 327733,
    327734, 327735, 327736, 327737, 327738, 327739, 327740, 327741, 327742, 327743, 327744, 327745, 327746, 327747, 327748, 327749, 327750, 327751};

  static constexpr Int_t FLP[2] = {178, 179};
  static constexpr Int_t CRUFROMFLP[2][2] = {{227, 228}, {225, 226}};
  static constexpr Int_t FLPFROMCRU[4] = {179, 179, 178, 178};
  static constexpr Int_t CRUID[4] = {225, 226, 227, 228};

  static constexpr Int_t CRULINK[kNCrate] = {
    11, 10, 0, 1, 9, 8, 2, 3, 7, 6, 4, 5, 5, 4, 6, 7, 3, 2,
    3, 9, 11, 10, 0, 1, 9, 8, 2, 3, 7, 6, 4, 5, 5, 4, 6, 7,
    11, 10, 0, 1, 9, 8, 2, 3, 7, 6, 4, 5, 5, 4, 6, 7, 3, 2,
    8, 9, 11, 10, 0, 1, 9, 8, 2, 3, 7, 6, 4, 5, 5, 4, 6, 7};

  static constexpr Int_t CRUENDPOINT[kNCrate] = {
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1,
    0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1};

  static constexpr Float_t ANGLES[NPLATES][NMAXNSTRIP] = { // Strip Tilt Angles
    {43.99, 43.20, 42.40, 41.59, 40.77, 39.94, 39.11, 38.25, 37.40, 36.53,
     35.65, 34.76, 33.87, 32.96, 32.05, 31.13, 30.19, 29.24, 12.33, 0.00},
    {27.26, 26.28, 25.30, 24.31, 23.31, 22.31, 21.30, 20.29, 19.26, 18.24,
     17.20, 16.16, 15.11, 14.05, 13.00, 11.93, 10.87, 9.80, 8.74, 0.00},
    {0.00, 6.30, 5.31, 4.25, 3.19, 2.12, 1.06, 0.00, -1.06, -2.12,
     -3.19, -4.25, -5.31, -6.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00},
    {-8.74, -9.80, -10.87, -11.93, -13.00, -14.05, -15.11, -16.16, -17.20, -18.24,
     -19.26, -20.29, -21.30, -22.31, -23.31, -24.31, -25.30, -26.28, -27.26, 0.00},
    {-12.33, -29.24, -30.19, -31.13, -32.05, -32.96, -33.87, -34.76, -35.65, -36.53,
     -37.40, -38.25, -39.11, -39.94, -40.77, -41.59, -42.40, -43.20, -43.99, 0.00}};

  static constexpr Float_t HEIGHTS[NPLATES][NMAXNSTRIP] = {
    {-8.405, -7.725, -8.405, -7.765, -8.285, -7.745, -7.865, -7.905, -7.895, -7.885,
     -7.705, -7.395, -7.525, -7.645, -7.835, -7.965, -8.365, -9.385, -3.255, 0.000},
    {-7.905, -8.235, -8.605, -9.045, -10.205, -3.975, -5.915, -7.765, -10.205, -3.635,
     -5.885, -8.005, -10.505, -4.395, -7.325, -10.235, -4.655, -7.495, -10.515, 0.000},
    {-2.705, -10.645, -5.165, -10.095, -4.995, -10.085, -4.835, -10.385, -4.835, -10.085,
     -4.995, -10.095, -5.165, -10.645, -2.705, 0.000, 0.000, 0.000, 0.000, 0.000},
    {-10.515, -7.495, -4.655, -10.235, -7.325, -4.395, -10.505, -8.005, -5.885, -3.635,
     -10.205, -7.765, -5.915, -3.975, -10.205, -9.045, -8.605, -8.235, -7.905, 0.000},
    {-3.255, -9.385, -8.365, -7.965, -7.835, -7.645, -7.525, -7.395, -7.705, -7.885,
     -7.895, -7.905, -7.865, -7.745, -8.285, -7.765, -8.405, -7.725, -8.405, 0.000}};

  static constexpr Float_t DISTANCES[NPLATES][NMAXNSTRIP] = {
    {364.14, 354.88, 344.49, 335.31, 325.44, 316.51, 307.11, 297.91, 288.84, 279.89,
     271.20, 262.62, 253.84, 245.20, 236.56, 228.06, 219.46, 210.63, 206.09, 0.00},
    {194.57, 186.38, 178.25, 170.13, 161.78, 156.62, 148.10, 139.72, 131.23, 125.87,
     117.61, 109.44, 101.29, 95.46, 87.36, 79.37, 73.17, 65.33, 57.71, 0.00},
    {49.28, 41.35, 35.37, 27.91, 21.20, 13.94, 7.06, 0.00, -7.06, -13.94,
     -21.20, -27.91, -35.37, -41.35, -49.28, 0.00, 0.00, 0.00, 0.00, 0.00},
    {-57.71, -65.33, -73.17, -79.37, -87.36, -95.46, -101.29, -109.44, -117.61, -125.87,
     -131.23, -139.72, -148.10, -156.62, -161.78, -170.13, -178.25, -186.38, -194.57, 0.00},
    {-206.09, -210.63, -219.46, -228.06, -236.56, -245.20, -253.84, -262.62, -271.20, -279.89,
     -288.84, -297.91, -307.11, -316.51, -325.44, -335.31, -344.49, -354.88, -364.14, 0.00}};

  // from AliTOFv6T0 class
  static constexpr Bool_t FEAWITHMASKS[NSECTORS] =
    // TOF sectors with Nino masks: 0, 8, 9, 10, 16
    {kTRUE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kTRUE,
     kTRUE, kTRUE, kFALSE, kFALSE, kFALSE, kFALSE, kFALSE, kTRUE, kFALSE};
  ; // Selecting TOF sectors containing FEA cooling masks

  static constexpr Float_t MODULEWALLTHICKNESS = 0.33;  // wall thickness (cm) (nofe: 0.3 as in AliTOFGeometry?)
  static constexpr Float_t INTERCENTRMODBORDER1 = 49.5; // 1st distance of
  // border between
  // central and
  // intermediate
  // modules respect to
  // the central module
  // centre (cm)
  static constexpr Float_t INTERCENTRMODBORDER2 = 57.5; // 2nd distance of
  // border between the
  // central and
  // intermediate
  // modules respect to
  // the central module
  // centre (cm)
  static constexpr Float_t EXTERINTERMODBORDER1 = 196.0; // 1st distance of
  // border between the
  // intermediate and
  // external modules
  // respect to the
  // central module
  // centre (cm)
  static constexpr Float_t EXTERINTERMODBORDER2 = 203.5; // 2nd distance of
  // border between the
  // intermediate and
  // external
  // modules respect to
  // the central module
  // centre (cm)
  static constexpr Float_t LENGTHINCEMODBORDERU = 5.0; // height of upper border
  // between the central
  // and intermediate
  // modules (cm)
  static constexpr Float_t LENGTHINCEMODBORDERD = 7.0; // height of lower border
  // between the central
  // and intermediate
  // modules (cm)
  static constexpr Float_t LENGTHEXINMODBORDER = 5.0; // height of border
  // between the
  // intermediate and
  // external modules
  // (cm)
  static constexpr Float_t MODULECOVERTHICKNESS = 2.0; // thickness of cover
  // modules zone (cm)
  static constexpr Float_t FEAWIDTH1 = 19.0;      // mother volume width of each of
                                                  // two external FEA in a
                                                  // supermodule (cm)
  static constexpr Float_t FEAWIDTH2 = 39.5;      // mother volume width of two
                                                  // internal FEA in a supermodule
                                                  // (cm)
  static constexpr Float_t SAWTHICKNESS = 1.0;    // services alluminium wall
                                                  // thickness (cm)
  static constexpr Float_t CBLW = 13.5;           // cables&tubes block width (cm)
  static constexpr Float_t CBLH1 = 2.0;           // min. height of cables&tubes block
                                                  // (cm)
  static constexpr Float_t CBLH2 = 12.3;          // max. height of cables&tubes block
                                                  // (cm)
  static constexpr Float_t BETWEENLANDMASK = 0.1; // distance between the L
  // element and the Nino
  // mask (cm)
  static constexpr Float_t AL1PARAMETERS[3] = {static_cast<Float_t>(FEAWIDTH1 * 0.5), 0.4, 0.2}; // (cm)
  static constexpr Float_t AL2PARAMETERS[3] = {7.25, 0.75, 0.25};                                // (cm)
  static constexpr Float_t AL3PARAMETERS[3] = {3., 4., 0.1};                                     // (cm)
  static constexpr Float_t ROOF1PARAMETERS[3] = {AL1PARAMETERS[0], AL1PARAMETERS[2], 1.45};      // (cm)
  static constexpr Float_t ROOF2PARAMETERS[3] = {AL3PARAMETERS[0], 0.1, 1.15};                   // (cm)
  static constexpr Float_t FEAPARAMETERS[3] = {static_cast<Float_t>(FEAWIDTH1 * 0.5), 5.6, 0.1}; // (cm)
  static constexpr Float_t BAR[3] = {8.575, 0.6, 0.25};                                          // (cm)
  static constexpr Float_t BAR1[3] = {BAR[0], BAR[1], 0.1};                                      // (cm)
  static constexpr Float_t BAR2[3] = {BAR[0], 0.1, static_cast<Float_t>(BAR[1] - 2. * BAR1[2])}; // (cm)
  static constexpr Float_t BARS[3] = {2., BAR[1], BAR[2]};                                       // (cm)
  static constexpr Float_t BARS1[3] = {BARS[0], BAR1[1], BAR1[2]};                               // (cm)
  static constexpr Float_t BARS2[3] = {BARS[0], BAR2[1], BAR2[2]};                               // (cm)

  static constexpr Float_t HHONY = 1.0;           // height of HONY Layer
  static constexpr Float_t HPCBY = 0.08;          // height of PCB Layer
  static constexpr Float_t HRGLY = 0.055;         // height of RED GLASS Layer
  static constexpr Float_t HFILIY = 0.125;        // height of FISHLINE Layer
  static constexpr Float_t HGLASSY = 0.160 * 0.5; // semi-height of GLASS Layer
  static constexpr Float_t HCPCBY = 0.16;         // height of PCB  Central Layer
  static constexpr Float_t WHONZ = 8.1;           // z dimension of HONEY Layer
  static constexpr Float_t WPCBZ1 = 10.64;        // z dimension of PCB Lower Layer
  static constexpr Float_t WPCBZ2 = 11.6;         // z dimension of PCB Upper Layer
  static constexpr Float_t WCPCBZ = 12.4;         // z dimension of PCB Central Layer
  static constexpr Float_t WRGLZ = 8.;            // z dimension of RED GLASS Layer
  static constexpr Float_t WGLFZ = 7.;            // z dimension of GLASS Layer
  static constexpr Float_t HSENSMY = 0.0105;      // height of Sensitive Layer

  static Float_t getCableLength(Int_t icrate, Int_t islot, Int_t ichain, Int_t itdc) { return CABLELENGTH[icrate][islot - 3][ichain][itdc / 3]; }
  static Float_t getCableTimeShift(Int_t icrate, Int_t islot, Int_t ichain, Int_t itdc) { return getCableLength(icrate, islot, ichain, itdc) * CABLEPROPAGATIONDELAY; }
  static Int_t getCableTimeShiftBin(Int_t icrate, Int_t islot, Int_t ichain, Int_t itdc) { return int(getCableTimeShift(icrate, islot, ichain, itdc) * 1000 / TDCBIN); }
  static Float_t getPropagationDelay() { return CABLEPROPAGATIONDELAY; };
  static Int_t getIndexFromEquipment(Int_t icrate, Int_t islot, Int_t ichain, Int_t itdc); // return TOF channel index

  static Int_t getCrateFromECH(int ech) { return ech >> 12; }
  static Int_t getTRMFromECH(int ech) { return ((ech % 4096) >> 8) + 3; }
  static Int_t getChainFromECH(int ech) { return (ech % 256) >> 7; }
  static Int_t getTDCFromECH(int ech) { return (ech % 128) >> 3; }
  static Int_t getTDCChFromECH(int ech) { return (ech % 8); }
  static Int_t getECHFromIndexes(int crate, int trm, int chain, int tdc, int chan) { return (crate << 12) + ((trm - 3) << 8) + (chain << 7) + (tdc << 3) + chan; }
  static Int_t getECHFromCH(int chan) { return CHAN_TO_ELCHAN[chan]; }
  static Int_t getCHFromECH(int echan) { return ELCHAN_TO_CHAN[echan]; }

  static void Init();
  static void InitIdeal();
  static void InitIndices();
  static void getPosInSectorCoord(int ch, float* pos);
  static void getPosInStripCoord(int ch, float* pos);
  static void getPosInPadCoord(int ch, float* pos);
  static void getPosInSectorCoord(const Int_t* detId, float* pos);
  static void getPosInStripCoord(const Int_t* detId, float* pos);
  static void getPosInPadCoord(const Int_t* detId, float* pos);

 private:
  static Int_t getSector(const Float_t* pos);
  static Int_t getPlate(const Float_t* pos);
  static Int_t getPadZ(const Float_t* pos);
  static Int_t getPadX(const Float_t* pos);

  static void fromGlobalToSector(Float_t* pos, Int_t isector);              // change coords to Sector reference
  static Int_t fromPlateToStrip(Float_t* pos, Int_t iplate, Int_t isector); // change coord to Strip reference and return strip number

  static Bool_t mToBeInit;
  static Bool_t mToBeInitIndexing;
  static Float_t mRotationMatrixSector[NSECTORS + 1][3][3]; // rotation matrixes
  static Float_t mRotationMatrixPlateStrip[NSECTORS][NPLATES][NMAXNSTRIP][3][3];
  static Float_t mPadPosition[NSECTORS][NPLATES][NMAXNSTRIP][NPADZ][NPADX][3];
  static Float_t mGeoDistances[NSECTORS][NPLATES][NMAXNSTRIP];
  static Float_t mGeoHeights[NSECTORS][NPLATES][NMAXNSTRIP];
  static Float_t mGeoX[NSECTORS][NPLATES][NMAXNSTRIP];
  static Int_t mPlate[NSTRIPXSECTOR];
  static Int_t mStripInPlate[NSTRIPXSECTOR];
  static std::array<std::vector<float>, 5> mDistances[NSECTORS];

  // cable length map
  static constexpr Float_t CABLEPROPAGATIONDELAY = 0.0513;           // Propagation delay [ns/cm]
  static const Float_t CABLELENGTH[kNCrate][10][kNChain][kNTdc / 3]; // not constexpr as we initialize it in CableLength.cxx at run time
  static const Int_t CHAN_TO_ELCHAN[NCHANNELS];
  static const Int_t ELCHAN_TO_CHAN[N_ELECTRONIC_CHANNELS];

  ClassDefNV(Geo, 3);
};
} // namespace tof
} // namespace o2

#endif

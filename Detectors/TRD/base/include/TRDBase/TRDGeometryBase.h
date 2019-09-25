// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_TRDGEOMETRYBASE_H
#define O2_TRDGEOMETRYBASE_H

#include "GPUCommonDef.h"
#include "TRDBase/TRDCommonParam.h"
#include "TRDBase/TRDPadPlane.h"

namespace o2
{
namespace trd
{
class TRDGeometryBase
{
 public:
  ~TRDGeometryBase() = default;

  static constexpr int MAXMATRICES = 521;

  GPUd() int isVersion() { return 1; }
  GPUd() bool isHole(int la, int st, int se) const { return (((se == 13) || (se == 14) || (se == 15)) && (st == 2)); }
  GPUd() bool isOnBoundary(int det, float y, float z, float eps = 0.5) const;

  GPUd() void setSMstatus(int sm, bool status)
  {
    if (status) {
      mSMStatus |= 0x3ffff & (0x1 << sm);
    } else {
      mSMStatus &= ~(0x3ffff & (0x1 << sm));
    }
  }
  GPUd() bool getSMstatus(int sm) const { return (mSMStatus & (0x1 << sm)) != 0; }
  GPUd() static int getDetectorSec(int det) { return (det % (kNlayer * kNstack)); }
  GPUd() static int getDetectorSec(int layer, int stack) { return (layer + stack * kNlayer); }
  GPUd() static int getDetector(int layer, int stack, int sector) { return (layer + stack * kNlayer + sector * kNlayer * kNstack); }
  GPUd() static int getLayer(int det) { return (det % kNlayer); }
  GPUd() static int getStack(int det) { return ((det % (kNlayer * kNstack)) / kNlayer); }
  GPUd() int getStack(float z, int layer) const;

  GPUd() const TRDPadPlane* getPadPlane(int layer, int stack) const { return &mPadPlanes[getDetectorSec(layer, stack)]; }
  GPUd() const TRDPadPlane* getPadPlane(int det) const { return &mPadPlanes[getDetectorSec(det)]; }

  GPUd() int getRowMax(int layer, int stack, int /*sector*/) const { return getPadPlane(layer, stack)->getNrows(); }
  GPUd() int getColMax(int layer) const { return getPadPlane(layer, 0)->getNcols(); }
  GPUd() float getRow0(int layer, int stack, int /*sector*/) const { return getPadPlane(layer, stack)->getRow0(); }
  GPUd() float getCol0(int layer) const { return getPadPlane(layer, 0)->getCol0(); }

  static constexpr int getSector(int det) { return (det / (kNlayer * kNstack)); }
  static constexpr float getTime0(int layer) { return TIME0[layer]; }
  static constexpr float getXtrdBeg() { return XTRDBEG; }
  static constexpr float getXtrdEnd() { return XTRDEND; }
  static constexpr float getChamberWidth(int layer) { return CWIDTH[layer]; }
  static constexpr float getChamberLength(int layer, int stack) { return CLENGTH[layer][stack]; }
  static constexpr float getAlpha() { return 2.0 * 3.14159265358979324 / kNsector; }
  static constexpr float cheight() { return CH; }
  static constexpr float cheightSV() { return CHSV; }
  static constexpr float cspace() { return VSPACE; }
  static constexpr float craHght() { return CRAH; }
  static constexpr float cdrHght() { return CDRH; }
  static constexpr float camHght() { return CAMH; }
  static constexpr float croHght() { return CROH; }
  static constexpr float csvHght() { return CSVH; }
  static constexpr float croWid() { return CROW; }
  static constexpr float anodePos() { return ANODEPOS; }
  static constexpr float myThick() { return RMYTHICK; }
  static constexpr float drThick() { return DRTHICK; }
  static constexpr float amThick() { return AMTHICK; }
  static constexpr float drZpos() { return DRZPOS; }
  static constexpr float rpadW() { return RPADW; }
  static constexpr float cpadW() { return CPADW; }
  static constexpr float cwidcha() { return (SWIDTH2 - SWIDTH1) / SHEIGHT * (CH + VSPACE); }
  static constexpr int MCMmax() { return MCMMAX; }
  static constexpr int MCMrow() { return MCMROW; }
  static constexpr int ROBmaxC0() { return ROBMAXC0; }
  static constexpr int ROBmaxC1() { return ROBMAXC1; }
  static constexpr int ADCmax() { return ADCMAX; }
  static constexpr int TBmax() { return TBMAX; }
  static constexpr int padmax() { return PADMAX; }
  static constexpr int colmax() { return COLMAX; }
  static constexpr int rowmaxC0() { return ROWMAXC0; }
  static constexpr int rowmaxC1() { return ROWMAXC1; }

 protected:
  TRDGeometryBase() = default;

  static constexpr float TLENGTH = 751.0; ///< Total length of the TRD mother volume

  // Parameter of the super module mother volumes
  static constexpr float SHEIGHT = 77.9;    ///<  Height of the supermodule
  static constexpr float SWIDTH1 = 94.881;  ///< Lower width of the supermodule
  static constexpr float SWIDTH2 = 122.353; ///< Upper width of the supermodule
  static constexpr float SLENGTH = 702.0;   ///< Length of the supermodule

  // Length of the additional space in front of the supermodule used for services
  static constexpr float FLENGTH = (TLENGTH - SLENGTH) / 2.0;

  static constexpr float SMPLTT = 0.2; ///< Thickness of the super module side plates

  static constexpr float VSPACE = 1.784; ///< Vertical spacing of the chambers
  static constexpr float HSPACE = 2.0;   ///< Horizontal spacing of the chambers
  static constexpr float VROCSM = 1.2;   ///< Radial distance of the first ROC to the outer plates of the SM

  static constexpr float CRAH = 4.8;                     ///<  Height of the radiator part of the chambers
  static constexpr float CDRH = 3.0;                     ///<  Height of the drift region of the chambers
  static constexpr float CAMH = 0.7;                     ///<  Height of the amplification region of the chambers
  static constexpr float CROH = 2.316;                   ///<  Height of the readout of the chambers
  static constexpr float CROW = 0.9;                     ///< Additional width of the readout chamber frames
  static constexpr float CSVH = VSPACE - 0.742;          ///< Height of the services on top of the chambers
  static constexpr float CH = CRAH + CDRH + CAMH + CROH; ///< Total height of the chambers (w/o services)
  static constexpr float CHSV = CH + CSVH;               ///< Total height of the chambers (with services)

  // Distance of anode wire plane relative to middle of alignable volume
  static constexpr float ANODEPOS = CRAH + CDRH + CAMH / 2.0 - CHSV / 2.0;

  static constexpr float CALT = 0.4;    ///< Thicknesses of different parts of the chamber frame Lower aluminum frame
  static constexpr float CCLST = 0.21;  ///< Thickness of the lower Wacosit frame sides
  static constexpr float CCLFT = 1.0;   ///< Thickness of the lower Wacosit frame front
  static constexpr float CGLT = 0.25;   ///< Thichness of the glue around the radiator
  static constexpr float CCUTA = 1.0;   ///< Upper Wacosit frame around amplification region
  static constexpr float CCUTB = 0.8;   ///< Thickness of the upper Wacosit frame around amp. region
  static constexpr float CAUT = 1.5;    ///< Al frame of back panel
  static constexpr float CALW = 2.5;    ///< Width of additional aluminum ledge on lower frame
  static constexpr float CALH = 0.4;    ///< Height of additional aluminum ledge on lower frame
  static constexpr float CALWMOD = 0.4; ///< Width of additional aluminum ledge on lower frame
  static constexpr float CALHMOD = 2.5; ///< Height of additional aluminum ledge on lower frame
  static constexpr float CWSW = 1.2;    ///< Width of additional wacosit ledge on lower frame
  static constexpr float CWSH = 0.3;    ///< Height of additional wacosit ledge on lower frame

  static constexpr float CPADW = 0.0; ///>Difference of outer chamber width and pad plane width
  static constexpr float RPADW = 1.0; ///<Difference of outer chamber width and pad plane width

  //
  // Thickness of the the material layers
  //
  static constexpr float DRTHICK = CDRH;              ///< Thickness of the drift region
  static constexpr float AMTHICK = CAMH;              ///< Thickness of the amplification region
  static constexpr float XETHICK = DRTHICK + AMTHICK; ///< Thickness of the gas volume
  static constexpr float WRTHICK = 0.00011;           ///< Thickness of the wire planes

  static constexpr float RMYTHICK = 0.0015;                                        ///< Thickness of the mylar layers in the radiator
  static constexpr float RCBTHICK = 0.0055;                                        ///< Thickness of the carbon layers in the radiator
  static constexpr float RGLTHICK = 0.0065;                                        ///< Thickness of the glue layers in the radiator
  static constexpr float RRHTHICK = 0.8;                                           ///< Thickness of the rohacell layers in the radiator
  static constexpr float RFBTHICK = CRAH - 2.0 * (RMYTHICK + RCBTHICK + RRHTHICK); ///< Thickness of the fiber layers in the radiator

  static constexpr float PPDTHICK = 0.0025;                                                                                  ///< Thickness of copper of the pad plane
  static constexpr float PPPTHICK = 0.0356;                                                                                  ///< Thickness of PCB board of the pad plane
  static constexpr float PGLTHICK = 0.1428;                                                                                  ///< Thickness of the glue layer
  static constexpr float PCBTHICK = 0.019;                                                                                   ///< Thickness of the carbon layers
  static constexpr float PPCTHICK = 0.0486;                                                                                  ///< Thickness of the PCB readout boards
  static constexpr float PRBTHICK = 0.0057;                                                                                  ///< Thickness of the PCB copper layers
  static constexpr float PELTHICK = 0.0029;                                                                                  ///< Thickness of all other electronics components (caps, etc.)
  static constexpr float PHCTHICK = CROH - PPDTHICK - PPPTHICK - PGLTHICK - PCBTHICK * 2.0 - PPCTHICK - PRBTHICK - PELTHICK; ///< Thickness of the honeycomb support structure

  //
  // Position of the material layers
  //
  static constexpr float DRZPOS = 2.4;                     ///< Position of the drift region
  static constexpr float AMZPOS = 0.0;                     ///< Position of the amplification region
  static constexpr float WRZPOSA = 0.0;                    ///< Position of the wire planes
  static constexpr float WRZPOSB = -AMTHICK / 2.0 + 0.001; ///< Position of the wire planes
  static constexpr float CALZPOS = 0.3;                    ///< Position of the additional aluminum ledges

  static constexpr int MCMMAX = 16;   ///< Maximum number of MCMs per ROB
  static constexpr int MCMROW = 4;    ///< Maximum number of MCMs per ROB Row
  static constexpr int ROBMAXC0 = 6;  ///< Maximum number of ROBs per C0 chamber
  static constexpr int ROBMAXC1 = 8;  ///< Maximum number of ROBs per C1 chamber
  static constexpr int ADCMAX = 21;   ///< Maximum number of ADC channels per MCM
  static constexpr int TBMAX = 60;    ///< Maximum number of Time bins
  static constexpr int PADMAX = 18;   ///< Maximum number of pads per MCM
  static constexpr int COLMAX = 144;  ///< Maximum number of pads per padplane row
  static constexpr int ROWMAXC0 = 12; ///< Maximum number of Rows per C0 chamber
  static constexpr int ROWMAXC1 = 16; ///< Maximum number of Rows per C1 chamber

  static constexpr float TIME0BASE = 300.65; ///< Base value for calculation of Time-position of pad 0
  // Time-position of pad 0
  static constexpr float TIME0[6] = {TIME0BASE + 0 * (CH + VSPACE),
                                     TIME0BASE + 1 * (CH + VSPACE),
                                     TIME0BASE + 2 * (CH + VSPACE),
                                     TIME0BASE + 3 * (CH + VSPACE),
                                     TIME0BASE + 4 * (CH + VSPACE),
                                     TIME0BASE + 5 * (CH + VSPACE)};

  static constexpr float XTRDBEG = 288.43; ///< X-coordinate in tracking system of begin of TRD mother volume
  static constexpr float XTRDEND = 366.33; ///< X-coordinate in tracking system of end of TRD mother volume

  // The outer width of the chambers
  static constexpr float CWIDTH[kNlayer] = {90.4, 94.8, 99.3, 103.7, 108.1, 112.6};

  // The outer lengths of the chambers
  // Includes the spacings between the chambers!
  static constexpr float CLENGTH[kNlayer][kNstack] = {
    {124.0, 124.0, 110.0, 124.0, 124.0},
    {124.0, 124.0, 110.0, 124.0, 124.0},
    {131.0, 131.0, 110.0, 131.0, 131.0},
    {138.0, 138.0, 110.0, 138.0, 138.0},
    {145.0, 145.0, 110.0, 145.0, 145.0},
    {147.0, 147.0, 110.0, 147.0, 147.0}};

  TRDPadPlane mPadPlanes[kNlayer * kNstack];

  int mSMStatus = 0x3ffff;

  ClassDefNV(TRDGeometryBase, 1);
};
} // end namespace trd
} // end namespace o2
#endif

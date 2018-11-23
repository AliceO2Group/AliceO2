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

#include "TRDBase/TRDCommonParam.h"

namespace o2
{
namespace trd
{
class TRDPadPlane;

class TRDGeometryBase
{
 public:
  ~TRDGeometryBase() = default;

  int isVersion() { return 1; }
  bool isHole(int la, int st, int se) const;
  bool isOnBoundary(int det, float y, float z, float eps = 0.5) const;

  void setSMstatus(int sm, bool status) {
    if (status) {
      mSMStatus |= 0x3ffff&(0x1<<sm);
    }
    else {
      mSMStatus &= ~(0x3ffff&(0x1<<sm));
    }
  }
  bool getSMstatus(int sm) const { return (mSMStatus&(0x1<<sm))!=0; }
  int getDetectorSec(int layer, int stack) const;
  int getDetector(int layer, int stack, int sector) const;
  int getLayer(int det) const;
  int getStack(int det) const;
  int getStack(float z, int layer) const;

  TRDPadPlane* getPadPlane(int layer, int stack) const;
  TRDPadPlane* getPadPlane(int det) const { return getPadPlane(getLayer(det), getStack(det)); }
  int getRowMax(int layer, int stack, int /*sector*/) const;
  int getColMax(int layer) const;
  float getRow0(int layer, int stack, int /*sector*/) const;
  float getCol0(int layer) const;

  static constexpr int getSector(int det) {return (det / (kNlayer * kNstack));}
  static constexpr float getTime0(int layer) { return fgkTime0[layer]; }
  static constexpr float getXtrdBeg() { return fgkXtrdBeg; }
  static constexpr float getXtrdEnd() { return fgkXtrdEnd; }
  static constexpr float getChamberWidth(int layer) { return fgkCwidth[layer]; }
  static constexpr float getChamberLength(int layer, int stack) { return fgkClength[layer][stack]; }
  static constexpr float getAlpha() { return 2.0 * 3.14159265358979324 / kNsector; }
  static constexpr float cheight() { return fgkCH; }
  static constexpr float cheightSV() { return fgkCHsv; }
  static constexpr float cspace() { return fgkVspace; }
  static constexpr float craHght() { return fgkCraH; }
  static constexpr float cdrHght() { return fgkCdrH; }
  static constexpr float camHght() { return fgkCamH; }
  static constexpr float croHght() { return fgkCroH; }
  static constexpr float csvHght() { return fgkCsvH; }
  static constexpr float croWid() { return fgkCroW; }
  static constexpr float anodePos() { return fgkAnodePos; }
  static constexpr float myThick() { return fgkRMyThick; }
  static constexpr float drThick() { return fgkDrThick; }
  static constexpr float amThick() { return fgkAmThick; }
  static constexpr float drZpos() { return fgkDrZpos; }
  static constexpr float rpadW() { return fgkRpadW; }
  static constexpr float cpadW() { return fgkCpadW; }
  static constexpr float cwidcha() { return (fgkSwidth2 - fgkSwidth1) / fgkSheight * (fgkCH + fgkVspace); }
  static constexpr int MCMmax() { return fgkMCMmax; }
  static constexpr int MCMrow() { return fgkMCMrow; }
  static constexpr int ROBmaxC0() { return fgkROBmaxC0; }
  static constexpr int ROBmaxC1() { return fgkROBmaxC1; }
  static constexpr int ADCmax() { return fgkADCmax; }
  static constexpr int TBmax() { return fgkTBmax; }
  static constexpr int padmax() { return fgkPadmax; }
  static constexpr int colmax() { return fgkColmax; }
  static constexpr int rowmaxC0() { return fgkRowmaxC0; }
  static constexpr int rowmaxC1() { return fgkRowmaxC1; }
 protected:
  TRDGeometryBase() = default;
  
  static constexpr float fgkTlength = 751.0; ///< Total length of the TRD mother volume

  // Parameter of the super module mother volumes
  static constexpr float fgkSheight = 77.9;    ///<  Height of the supermodule
  static constexpr float fgkSwidth1 = 94.881;  ///< Lower width of the supermodule
  static constexpr float fgkSwidth2 = 122.353; ///< Upper width of the supermodule
  static constexpr float fgkSlength = 702.0;   ///< Length of the supermodule

  // Length of the additional space in front of the supermodule used for services
  static constexpr float fgkFlength = (fgkTlength - fgkSlength) / 2.0;

  static constexpr float fgkSMpltT = 0.2;   ///< Thickness of the super module side plates

  static constexpr float fgkVspace = 1.784; ///< Vertical spacing of the chambers
  static constexpr float fgkHspace = 2.0;   ///< Horizontal spacing of the chambers
  static constexpr float fgkVrocsm = 1.2;   ///< Radial distance of the first ROC to the outer plates of the SM

  static constexpr float fgkCraH = 4.8;     ///<  Height of the radiator part of the chambers
  static constexpr float fgkCdrH = 3.0;     ///<  Height of the drift region of the chambers
  static constexpr float fgkCamH = 0.7;     ///<  Height of the amplification region of the chambers
  static constexpr float fgkCroH = 2.316;   ///<  Height of the readout of the chambers
  static constexpr float fgkCroW = 0.9;     ///< Additional width of the readout chamber frames
  static constexpr float fgkCsvH = fgkVspace - 0.742; ///< Height of the services on top of the chambers
  static constexpr float fgkCH = fgkCraH + fgkCdrH + fgkCamH + fgkCroH; ///< Total height of the chambers (w/o services)
  static constexpr float fgkCHsv = fgkCH + fgkCsvH; ///< Total height of the chambers (with services)

  // Distance of anode wire plane relative to middle of alignable volume
  static constexpr float fgkAnodePos = fgkCraH + fgkCdrH + fgkCamH / 2.0 - fgkCHsv / 2.0;

  static constexpr float fgkCalT = 0.4;    ///< Thicknesses of different parts of the chamber frame Lower aluminum frame
  static constexpr float fgkCclsT = 0.21;  ///< Thickness of the lower Wacosit frame sides
  static constexpr float fgkCclfT = 1.0;   ///< Thickness of the lower Wacosit frame front
  static constexpr float fgkCglT = 0.25;   ///< Thichness of the glue around the radiator
  static constexpr float fgkCcuTa = 1.0;   ///< Upper Wacosit frame around amplification region
  static constexpr float fgkCcuTb = 0.8;   ///< Thickness of the upper Wacosit frame around amp. region
  static constexpr float fgkCauT = 1.5;    ///< Al frame of back panel
  static constexpr float fgkCalW = 2.5;    ///< Width of additional aluminum ledge on lower frame
  static constexpr float fgkCalH = 0.4;    ///< Height of additional aluminum ledge on lower frame
  static constexpr float fgkCalWmod = 0.4; ///< Width of additional aluminum ledge on lower frame
  static constexpr float fgkCalHmod = 2.5; ///< Height of additional aluminum ledge on lower frame
  static constexpr float fgkCwsW = 1.2;    ///< Width of additional wacosit ledge on lower frame
  static constexpr float fgkCwsH = 0.3;    ///< Height of additional wacosit ledge on lower frame

  static constexpr float fgkCpadW = 0.0;   ///>Difference of outer chamber width and pad plane width
  static constexpr float fgkRpadW = 1.0;   ///<Difference of outer chamber width and pad plane width

  //
  // Thickness of the the material layers
  //
  static constexpr float fgkDrThick = fgkCdrH; ///< Thickness of the drift region
  static constexpr float fgkAmThick = fgkCamH; ///< Thickness of the amplification region
  static constexpr float fgkXeThick = fgkDrThick + fgkAmThick; ///< Thickness of the gas volume
  static constexpr float fgkWrThick = 0.00011; ///< Thickness of the wire planes

  static constexpr float fgkRMyThick = 0.0015; ///< Thickness of the mylar layers in the radiator
  static constexpr float fgkRCbThick = 0.0055; ///< Thickness of the carbon layers in the radiator
  static constexpr float fgkRGlThick = 0.0065; ///< Thickness of the glue layers in the radiator
  static constexpr float fgkRRhThick = 0.8;    ///< Thickness of the rohacell layers in the radiator
  static constexpr float fgkRFbThick = fgkCraH - 2.0 * (fgkRMyThick + fgkRCbThick + fgkRRhThick); ///< Thickness of the fiber layers in the radiator

  static constexpr float fgkPPdThick = 0.0025; ///< Thickness of copper of the pad plane
  static constexpr float fgkPPpThick = 0.0356; ///< Thickness of PCB board of the pad plane
  static constexpr float fgkPGlThick = 0.1428; ///< Thickness of the glue layer
  static constexpr float fgkPCbThick = 0.019;  ///< Thickness of the carbon layers
  static constexpr float fgkPPcThick = 0.0486; ///< Thickness of the PCB readout boards
  static constexpr float fgkPRbThick = 0.0057; ///< Thickness of the PCB copper layers
  static constexpr float fgkPElThick = 0.0029; ///< Thickness of all other electronics components (caps, etc.)
  static constexpr float fgkPHcThick =
    fgkCroH - fgkPPdThick - fgkPPpThick - fgkPGlThick - fgkPCbThick * 2.0 - fgkPPcThick - fgkPRbThick - fgkPElThick; ///< Thickness of the honeycomb support structure

  //
  // Position of the material layers
  //
  static constexpr float fgkDrZpos = 2.4;  ///< Position of the drift region
  static constexpr float fgkAmZpos = 0.0;  ///< Position of the amplification region
  static constexpr float fgkWrZposA = 0.0; ///< Position of the wire planes
  static constexpr float fgkWrZposB = -fgkAmThick / 2.0 + 0.001; ///< Position of the wire planes
  static constexpr float fgkCalZpos = 0.3; ///< Position of the additional aluminum ledges

  static constexpr int fgkMCMmax = 16;     ///< Maximum number of MCMs per ROB
  static constexpr int fgkMCMrow = 4;      ///< Maximum number of MCMs per ROB Row
  static constexpr int fgkROBmaxC0 = 6;    ///< Maximum number of ROBs per C0 chamber
  static constexpr int fgkROBmaxC1 = 8;    ///< Maximum number of ROBs per C1 chamber
  static constexpr int fgkADCmax = 21;     ///< Maximum number of ADC channels per MCM
  static constexpr int fgkTBmax = 60;      ///< Maximum number of Time bins
  static constexpr int fgkPadmax = 18;     ///< Maximum number of pads per MCM
  static constexpr int fgkColmax = 144;    ///< Maximum number of pads per padplane row
  static constexpr int fgkRowmaxC0 = 12;   ///< Maximum number of Rows per C0 chamber
  static constexpr int fgkRowmaxC1 = 16;   ///< Maximum number of Rows per C1 chamber

  static constexpr float fgkTime0Base = 300.65; ///< Base value for calculation of Time-position of pad 0
  // Time-position of pad 0
  static constexpr float fgkTime0[6] = { fgkTime0Base + 0 * (fgkCH + fgkVspace),
                                           fgkTime0Base + 1 * (fgkCH + fgkVspace),
                                           fgkTime0Base + 2 * (fgkCH + fgkVspace),
                                           fgkTime0Base + 3 * (fgkCH + fgkVspace),
                                           fgkTime0Base + 4 * (fgkCH + fgkVspace),
                                           fgkTime0Base + 5 * (fgkCH + fgkVspace) };

  static constexpr float fgkXtrdBeg = 288.43; ///< X-coordinate in tracking system of begin of TRD mother volume
  static constexpr float fgkXtrdEnd = 366.33; ///< X-coordinate in tracking system of end of TRD mother volume

  // The outer width of the chambers
  static constexpr float fgkCwidth[kNlayer] = { 90.4, 94.8, 99.3, 103.7, 108.1, 112.6 };

  // The outer lengths of the chambers
  // Includes the spacings between the chambers!
  static constexpr float fgkClength[kNlayer][kNstack] = {
    { 124.0, 124.0, 110.0, 124.0, 124.0 }, { 124.0, 124.0, 110.0, 124.0, 124.0 }, { 131.0, 131.0, 110.0, 131.0, 131.0 },
    { 138.0, 138.0, 110.0, 138.0, 138.0 }, { 145.0, 145.0, 110.0, 145.0, 145.0 }, { 147.0, 147.0, 110.0, 147.0, 147.0 }
  };

  int mSMStatus = 0x3ffff;
  
  TRDPadPlane* mPadPlaneArray = nullptr;

 private:
  ClassDefNV(TRDGeometryBase, 1) //  TRD geometry class
};
} // end namespace trd
} // end namespace o2
#endif

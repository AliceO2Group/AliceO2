// @(#) $Id$
// Original: AliHLTTransform.h,v 1.37 2005/06/14 10:55:21 cvetan 

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCGEOMETRY_H
#define ALIHLTTPCGEOMETRY_H

#ifdef use_aliroot
  class AliRunLoader;
#endif

#include "Rtypes.h"

class AliHLTTPCGeometry {

 public:
    AliHLTTPCGeometry();
  enum VersionType { kVdefault=0, kVdeprecated=1, kValiroot=10, kVcosmics=100};

 private:
  static const Double_t fgkBFACT; //bfield
  static const Double_t fgkPi;    //pi
  static const Double_t fgkPi2;   //2pi
  static const Double_t fgk2Pi;   //pi/2
  static const Double_t fgkAnodeWireSpacing; //anode wire spacing 
  static const Double_t fgkToDeg; //rad to deg

  static Int_t fgNPatches;   //6 (dont change this) 
  static Int_t fgRows[6][2]; //rows per patch
  static Int_t fgNRows[6];   //rows per patch

  static Double_t fgBField;         //field
  static Double_t fgBFieldFactor;   //field 
  static Double_t fgSolenoidBField; //field
  static Int_t fgNTimeBins;  //ntimebins
  static Int_t fgNRowLow;    //nrows
  static Int_t fgNRowUp;     //nrows
  static Int_t fgNRowUp1;    //nrows
  static Int_t fgNRowUp2;    //nrows
  static Int_t fgNSectorLow; //nsector
  static Int_t fgNSectorUp;  //nsector
  static Int_t fgSlice2Sector[36][2]; //nslice
  static Int_t fgSector2Slice[72];    //nslice
  static Int_t fgSectorLow[72];       //nsector
  static Double_t fgPadPitchWidthLow; //pad pitch
  static Double_t fgPadPitchWidthUp;  //pad pitch
  static Double_t fgZWidth;  //width
  static Double_t fgZSigma;  //sigma
  static Double_t fgZLength; //length
  static Double_t fgZOffset; //offset
  static Int_t fgNSector; //72  (dont change this)
  static Int_t fgNSlice;  //36  (dont change this)
  static Int_t fgNRow;    //159 (dont change this)
  static Double_t fgNRotShift; //Rotation shift (eg. 0.5 for 10 degrees)
  static Int_t fgNPads[159]; //fill this following Init and fVersion
  static Double_t fgX[159];  //X position in local coordinates
  static Int_t fgVersion;  //flags the version
  static Double_t fgDiffT; //Transversal diffusion constant
  static Double_t fgDiffL; //Longitudinal diffusion constant
  static Double_t fgOmegaTau; //ExB effects
  static Double_t fgInnerPadLength;  //innner pad length
  static Double_t fgOuter1PadLength; //outer pad length
  static Double_t fgOuter2PadLength; //outer pad length
  static Double_t fgInnerPRFSigma;   //inner pad response function
  static Double_t fgOuter1PRFSigma;  //outer pad response function
  static Double_t fgOuter2PRFSigma;  //outer pad response function
  static Double_t fgTimeSigma; //Minimal longitudinal width
  static Int_t fgADCSat; //ADC Saturation (1024 = 10 bit)
  static Int_t fgZeroSup; //Zero suppression threshold
  static Double_t fgCos[36]; //stores the cos value for local to global rotations  
  static Double_t fgSin[36]; //stores the sin value for local to global rotations  

 public:
  virtual ~AliHLTTPCGeometry() {}

  //setters
  static void SetNPatches(Int_t i){fgNPatches = i;}
  static void SetNRows(Int_t s[6]){for(Int_t i=0;i<fgNPatches;i++) fgNRows[i] = s[i];}
  static void SetRows(Int_t s[6][2]){
    for(Int_t i=0;i<fgNPatches;i++){
      fgRows[i][0] = s[i][0];
      fgRows[i][1] = s[i][1];
    }
  }
  static void SetBField(Double_t f) {fgBField = f;} //careful, these 3 are not independent!
  static void SetBFieldFactor(Double_t f) {
    fgBFieldFactor = f;
    fgBField=fgBFieldFactor*fgSolenoidBField*0.1;
  }
  static void SetSolenoidBField(Double_t f){
    fgSolenoidBField = f;
    fgBField=fgBFieldFactor*fgSolenoidBField*0.1;
  }
  static void SetNTimeBins(Int_t i){fgNTimeBins = i; if (fgNTimeBins>0) {fgZWidth = fgZLength / (Double_t)fgNTimeBins;}}
  static void SetNRowLow(Int_t i){fgNRowLow = i;}
  static void SetNRowUp(Int_t i){fgNRowUp = i;}
  static void SetNRowUp1(Int_t i){fgNRowUp1 = i;}
  static void SetNRowUp2(Int_t i){fgNRowUp2 = i;}
  static void SetSlice2Sector(Int_t s[36][2]){
    for(Int_t i=0;i<fgNSlice;i++){
      fgSlice2Sector[i][0] = s[i][0];
      fgSlice2Sector[i][1] = s[i][1];
    }
  }
  static void SetSector2Slice(Int_t s[72]){
    for(Int_t i=0;i<fgNSector;i++) fgSector2Slice[i] = s[i];}
  static void SetSectorLow(Int_t s[72]){
    for(Int_t i=0;i<fgNSector;i++) fgSectorLow[i] = s[i];}
  static void SetNSectorLow(Int_t i){fgNSectorLow = i;}
  static void SetNSectorUp(Int_t i){fgNSectorUp = i;}
  static void SetPadPitchWidthLow(Double_t f){fgPadPitchWidthLow = f;}
  static void SetPadPitchWidthUp(Double_t f){fgPadPitchWidthUp = f;}
  // Matthias 21.09.2007
  // zwidth is given by zlength and no of timebins and should not be set
  // otherwise. Was never used
  //static void SetZWidth(Double_t f){fgZWidth = f;}
  static void SetZSigma(Double_t f){fgZSigma = f;}
  static void SetZLength(Double_t f){fgZLength = f;}
  static void SetZOffset(Double_t f){fgZOffset = f;}
  static void SetNSector(Int_t i){fgNSector = i;}
  static void SetNSlice(Int_t i){fgNSlice = i;}
  static void SetNRow(Int_t i){fgNRow = i;}
  static void SetNRotShift(Double_t f){fgNRotShift = f;}
  static void SetNPads(Int_t pads[159]){
    for(Int_t i=0;i<fgNRow;i++) fgNPads[i] = pads[i];}
  static void SetX(Double_t xs[159]){
    for(Int_t i=0;i<fgNRow;i++) fgX[i] = xs[i];}
  static void SetVersion(Int_t i){fgVersion = i;}
  static void SetDiffT(Double_t f){fgDiffT = f;}
  static void SetDiffL(Double_t f){fgDiffL = f;}
  static void SetOmegaTau(Double_t f){fgOmegaTau = f;}
  static void SetInnerPadLength(Double_t f){fgInnerPadLength = f;}
  static void SetOuter1PadLength(Double_t f){fgOuter1PadLength = f;}
  static void SetOuter2PadLength(Double_t f){fgOuter2PadLength = f;}
  static void SetInnerPRFSigma(Double_t f){fgInnerPRFSigma = f;}
  static void SetOuter1PRFSigma(Double_t f){fgOuter1PRFSigma = f;}
  static void SetOuter2PRFSigma(Double_t f){fgOuter2PRFSigma = f;}
  static void SetTimeSigma(Double_t f){fgTimeSigma = f;}
  static void SetADCSat(Int_t i) {fgADCSat = i;}
  static void SetZeroSup(Int_t i) {fgZeroSup = i;}

  //getters
  static const Char_t* GetParamName() {return "75x40_100x60_150x60";}
  static Double_t Pi()     {return fgkPi;}
  static Double_t PiHalf() {return fgkPi2;}
  static Double_t TwoPi()  {return fgk2Pi;}
  static Double_t GetAnodeWireSpacing() {return fgkAnodeWireSpacing;}
  static Double_t GetBFact() {return fgkBFACT;}
  static Double_t ToRad() {return 1./fgkToDeg;}
  static Double_t ToDeg() {return fgkToDeg;}

  static Int_t GetNumberOfPatches();
  static Int_t GetFirstRow(Int_t patch);
  static Int_t GetLastRow(Int_t patch);
  static Int_t GetFirstRowOnDDL(Int_t patch);
  static Int_t GetLastRowOnDDL(Int_t patch);
  static Int_t GetNRows(Int_t patch);
  static Int_t GetPatch(Int_t padrow);
  static Int_t GetNRows() {return fgNRow;}
  static Int_t GetNRowLow() {return fgNRowLow;}
  static Int_t GetNRowUp1() {return fgNRowUp1;}
  static Int_t GetNRowUp2() {return fgNRowUp2;}
  static Int_t GetPadRow(Float_t x);
  static Int_t GetNPatches() {return fgNPatches;}
  static Int_t GetNPads(Int_t row);
  static Int_t GetNTimeBins(){return fgNTimeBins;}
  static Double_t GetBField() {return fgBField;}
  static Double_t GetSolenoidField() {return fgSolenoidBField;}
  static Double_t GetBFactFactor() {return fgBFieldFactor;}
  static Double_t GetBFieldValue() {return (fgBField*fgkBFACT);}
  static Float_t Deg2Rad(Float_t angle) {return angle/fgkToDeg;}
  static Float_t Rad2Deg(Float_t angle) {return angle*fgkToDeg;}
  static Int_t GetVersion(){return fgVersion;}
  static Double_t GetPadPitchWidthLow() {return fgPadPitchWidthLow;}
  static Double_t GetPadPitchWidthUp() {return fgPadPitchWidthUp;}
  static Double_t GetPadPitchWidth(Int_t patch);
  static Double_t GetZWidth() {return fgZWidth;}
  static Double_t GetZLength() {return fgZLength;}
  static Double_t GetZOffset() {return fgZOffset;}
  static Double_t GetDiffT() {return fgDiffT;}
  static Double_t GetDiffL() {return fgDiffL;}
  static Double_t GetParSigmaY2(Int_t padrow,Float_t z,Float_t angle);
  static Double_t GetParSigmaZ2(Int_t padrow,Float_t z,Float_t tgl);
  static Double_t GetOmegaTau() {return fgOmegaTau;}
  static Double_t GetPadLength(Int_t padrow);
  static Double_t GetPRFSigma(Int_t padrow);
  static Double_t GetTimeSigma() {return fgTimeSigma;}
  static Double_t GetZSigma() {return fgZSigma;}
  static Int_t GetADCSat() {return fgADCSat;}
  static Int_t GetZeroSup() {return fgZeroSup;}
  static Int_t GetNSlice() {return fgNSlice;}
  static Int_t GetNSector() {return fgNSector;}
  static Int_t GetNSectorLow() {return fgNSectorLow;}
  static Int_t GetNSectorUp() {return fgNSectorUp;}
  
  static Bool_t Slice2Sector(Int_t slice, Int_t slicerow, Int_t &sector, Int_t &row);
  static Bool_t Sector2Slice(Int_t &slice, Int_t sector);
  static Bool_t Sector2Slice(Int_t &slice, Int_t &slicerow, Int_t sector, Int_t row);

  static Double_t Row2X(Int_t slicerow);
  static Double_t GetMaxY(Int_t slicerow);
  static Double_t GetEta(Float_t *xyz);
  static Double_t GetEta(Int_t slice,Int_t padrow, Int_t pad, Int_t time);
  static Double_t GetPhi(Float_t *xyz);
  static Double_t GetZFast(Int_t slice, Int_t time, Float_t vertex=0.);

  static void XYZtoRPhiEta(Float_t *rpe, Float_t *xyz);
  static void Local2Global(Float_t *xyz, Int_t slice);
  static void Local2GlobalAngle(Float_t *angle, Int_t slice);
  static void Global2LocalAngle(Float_t *angle, Int_t slice);

  //we have 3 different system: Raw   : row, pad, time
  //                            Local : x,y and global z
  //                            Global: global x,y and global z
  //the methods with HLT in the name differ from the other
  //as you specify slice and slicerow, instead of sector
  //and sector row. In that way we safe "a few ifs"
  static void Raw2Local(Float_t *xyz, Int_t sector, Int_t row, Float_t pad, Float_t time);
  static void RawHLT2Local(Float_t *xyz,Int_t slice,Int_t slicerow,Float_t pad,Float_t time);
  static void Raw2Local(Float_t *xyz, Int_t sector, Int_t row, Int_t pad, Int_t time);
  static void RawHLT2Local(Float_t *xyz,Int_t slice,Int_t slicerow,Int_t pad,Int_t time);
  static void Local2Global(Float_t *xyz, Int_t sector, Int_t row);
  static void LocHLT2Global(Float_t *xyz, Int_t slice, Int_t slicerow);
  static void Global2Local(Float_t *xyz, Int_t sector);
  static void Global2LocHLT(Float_t *xyz, Int_t slice);
  static void Raw2Global(Float_t *xyz, Int_t sector, Int_t row, Float_t pad, Float_t time);
  static void RawHLT2Global(Float_t *xyz, Int_t slice, Int_t slicerow, Float_t pad, Float_t time);
  static void Raw2Global(Float_t *xyz, Int_t sector, Int_t row, Int_t pad, Int_t time);
  static void RawHLT2Global(Float_t *xyz, Int_t slice, 
                            Int_t slicerow, Int_t pad, Int_t time);
  static void Local2Raw(Float_t *xyz, Int_t sector, Int_t row);
  static void LocHLT2Raw(Float_t *xyz, Int_t slice, Int_t slicerow);
  static void Global2Raw(Float_t *xyz, Int_t sector, Int_t row);
  static void Global2HLT(Float_t *xyz, Int_t slice, Int_t slicerow);

  static void PrintCompileOptions();
  
  // cluster indexation

  static UInt_t CreateClusterID( UInt_t Slice, UInt_t Partition, UInt_t ClusterIndex ){
    return ( (Slice & 0x3F)<<25 ) + ( (Partition & 0x7)<<22 ) + ( ClusterIndex & 0x003FFFFF );
  }
  static UInt_t CluID2Slice     ( UInt_t ClusterId )  { return ( ClusterId >> 25 )  & 0x3F; }
  static UInt_t CluID2Partition ( UInt_t ClusterId )  { return ( ClusterId >> 22 )  & 0x7;  }
  static UInt_t CluID2Index     ( UInt_t ClusterId )  { return   ClusterId & 0x003FFFFF;    }
  

  ClassDef(AliHLTTPCGeometry,0)
};
#endif

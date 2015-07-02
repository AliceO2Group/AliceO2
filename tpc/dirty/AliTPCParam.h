#ifndef ALITPCPARAM_H
#define ALITPCPARAM_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

////////////////////////////////////////////////
//  Manager class for TPC parameters          //
////////////////////////////////////////////////

#include "AliDetectorParam.h"
#include "TMath.h"

#include <TGeoMatrix.h>
#include <TVectorD.h>
class TString;
class TGraphErrors;

class AliTPCParam : public AliDetectorParam {
  //////////////////////////////////////////////////////
  //////////////////////////////////////////////////////
  //ALITPCParam object to be possible change 
  //geometry and some other parameters of TPC   
  //used by AliTPC and AliTPCSector 
 
public:
  AliTPCParam(); 
  virtual ~AliTPCParam();
  TGeoHMatrix *  Tracking2LocalMatrix(const TGeoHMatrix * geoMatrix, Int_t sector) const;  
  virtual Bool_t  Transform(Float_t *xyz, Int_t *index, Int_t* oindex);
  //transformation from input coodination system to output coordination system  
  Int_t  Transform0to1(Float_t *xyz, Int_t *index) const;
  //trasforamtion from global to global - adjust index[0] sector 
  //return value is equal to sector corresponding to global position
  void Transform1to2Ideal(Float_t *xyz, Int_t *index) const;
  //transformation to rotated coordinata - ideal frame 
  void Transform1to2(Float_t *xyz, Int_t *index) const;
  //transformation to rotated coordinata   
  void Transform2to1(Float_t *xyz, Int_t *index) const;
  //transformation from rotated coordinata to global coordinata
  void Transform2to2(Float_t *xyz, Int_t *index, Int_t *oindex) const;
  //transform rotated coordinata of one sector to rotated
  //coordinata relative to another sector
  Float_t  Transform2to2NearestWire(Float_t *xyz, Int_t *index) const;
  //round x position to nearest wire
  Int_t   Transform2to3(Float_t *xyz, Int_t *index) const;
  //calulate coresponding index[2] -pad row for straight rows
  //does not change xyz[] 
  //return pad - row 
  void   Transform3to4(Float_t *xyz, Int_t *index) const;
  //valid only for straight rows straight rows
  //calculate xyz[0] position relative to given index
  //return pad - row 
  void   Transform4to3(Float_t *xyz, Int_t *index) const;
  //valid only for straight rows straight rows
  //transform  xyz[0] position relative to given index
  void   Transform2to5( Float_t *xyz, Int_t *index) const;
  //transform [x,y,z] to [r,rphi,z]
  void   Transform5to2(Float_t *xyz, Int_t *index) const;
  //transform [r,rphi,z] coordinata to [x,y,z] 
  void  Transform4to8(Float_t *xyz, Int_t *index) const;
  //transform xyz coordinata to 'digit' coordinata
  void  Transform8to4(Float_t *xyz, Int_t *index) const;
  //transform  'digit' coordinata to xyz coordinata   
  void  Transform6to8(Float_t *xyz, Int_t *index) const;
  //transform dr,f coordinata to 'digit' coordinata
  void  Transform8to6(Float_t *xyz, Int_t *index) const;
  //transform 'digit' coordinata to dr,f coordinata 

  virtual Int_t  Transform2toPadRow(Float_t */*xyz*/, Int_t */*index*/) const{return 0;}
  //transform rotated to

  virtual  Int_t GetPadRow(Float_t *xyz, Int_t *index) const ;
  //return pad row of point xyz - xyz is given in coordinate system -(given by index)
  //output system is 3 for straight row and 7 for cylindrical row
  virtual void XYZtoCRXYZ(Float_t */*xyz*/, 
			  Int_t &/*sector*/, Int_t &/*padrow*/, Int_t /*option*/) const {;}
  //transform global position to the position relative to the sector padrow
  //if option=0  X calculate absolute            calculate sector
  //if option=1  X           absolute            use input sector
  //if option=2  X           relative to pad row calculate sector
  //if option=3  X           relative            use input sector

  virtual void CRXYZtoXYZ(Float_t */*xyz*/,
			  const Int_t &/*sector*/, const Int_t & /*padrow*/, Int_t /*option*/) const {;}  
  //transform relative position  to the gloabal position

  virtual void CRTimePadtoYZ(Float_t &/*y*/, Float_t &/*z*/, 
			     const Float_t &/*time*/, const Float_t &/*pad*/,
			     Int_t /*sector*/, Int_t /*padrow*/ ){;}
  //transform position in digit  units (time slices and pads)  to "normal" 
  //units (cm)   
  virtual void CRYZtoTimePad(const Float_t &/*y*/, const Float_t &/*z*/, 
			     Float_t &/*time*/, Float_t &/*pad*/,
			     Int_t /*sector*/, Int_t /*padrow*/){;}
  //transform position in cm to position in digit unit 
  virtual Int_t   CalcResponse(Float_t* /*x*/, Int_t * /*index*/, Int_t /*row*/){return 0;}
  //calculate bin response as function of the input position -x and the weight 
  //if row -pad row is equal -1 calculate response for each pad row 
  //otherwise it calculate only in given pad row
  //return number of valid response bin
  virtual void SetDefault();          //set defaut TPCparam 
  virtual Bool_t Update();            //recalculate and check geometric parameters 
  virtual Bool_t ReadGeoMatrices();   //read geo matrixes        
  Bool_t GetStatus() const;         //get information about object consistency  
  Int_t GetIndex(Int_t sector, Int_t row) const;  //give index of the given sector and pad row 
  Int_t GetNSegmentsTotal() const {return fNtRows;} 
  Double_t GetLowMaxY(Int_t irow) const {return irow*0.;}
  Double_t GetUpMaxY(Int_t irow) const {return irow*0;}
  //additional geometrical function - for Belikov
 
  Bool_t   AdjustSectorRow(Int_t index, Int_t & sector, Int_t &row) const; //return sector and padrow
  //for given index

  void  AdjustCosSin(Int_t isec, Float_t &cos, Float_t &sin) const;
  //set cosinus and sinus of rotation angles for sector isec 
  Float_t GetAngle(Int_t isec) const;
  //  void GetChamberPos(Int_t isec, Float_t* xyz) const;
  //  void GetChamberRot(Int_t isec, Float_t* angles) const;
  //
  //set sector parameters
  //
  void  SetInnerRadiusLow(Float_t InnerRadiusLow )  { fInnerRadiusLow=InnerRadiusLow;}
  void  SetOuterRadiusLow(Float_t OuterRadiusLow )  { fOuterRadiusLow=OuterRadiusLow;} 
  void  SetInnerRadiusUp(Float_t InnerRadiusUp)  {  fInnerRadiusUp= InnerRadiusUp;} 
  void  SetOuterRadiusUp(Float_t OuterRadiusUp) {  fOuterRadiusUp= OuterRadiusUp;}   
  void  SetSectorAngles(Float_t innerangle, Float_t innershift, Float_t outerangle,
			Float_t outershift);
  void  SetInnerFrameSpace(Float_t frspace) {fInnerFrameSpace = frspace;}
  void  SetOuterFrameSpace(Float_t frspace) {fOuterFrameSpace = frspace;}
  void  SetInnerWireMount(Float_t fmount) {fInnerWireMount = fmount;}
  void  SetOuterWireMount(Float_t fmount) {fOuterWireMount = fmount;}
  void  SetZLength(Float_t zlength) {fZLength = zlength;} 
  void  SetGeometryType(Int_t type) {fGeometryType = type;}
  //
  // pad rows geometry
  //
  void SetRowNLow( Int_t NRowLow){fNRowLow = NRowLow;}
  void SetRowNUp1 (Int_t NRowUp1){fNRowUp1 = NRowUp1 ;} //upper sec short pads 
  void SetRowNUp2 (Int_t NRowUp2){fNRowUp2 = NRowUp2 ;} //upper sec long pads
  void SetRowNUp (Int_t NRowUp){fNRowUp = NRowUp ;}  
  //
  //set wire parameters
  //
  void  SetInnerNWires(Int_t nWires){  fNInnerWiresPerPad=nWires;}
  void  SetInnerDummyWire(Int_t dummy) {fInnerDummyWire  = dummy;}
  void  SetInnerOffWire(Float_t offset) {fInnerOffWire =offset;}    
  void  SetOuter1NWires(Int_t nWires){  fNOuter1WiresPerPad=nWires;}
  void  SetOuter2NWire(Int_t nWires){  fNOuter2WiresPerPad=nWires;}
  void  SetOuterDummyWire(Int_t dummy) {fOuterDummyWire  = dummy;}
  void  SetOuterOffWire(Float_t offset) {fOuterOffWire =offset;} 
  void  SetInnerWWPitch( Float_t wwPitch) {fInnerWWPitch = wwPitch;}
  void  SetRInnerFirstWire(Float_t firstWire){fRInnerFirstWire = firstWire;}
  void  SetRInnerLastWire(Float_t lastWire){fRInnerLastWire = lastWire;}
  void  SetOuterWWPitch(Float_t wwPitch){fOuterWWPitch = wwPitch;}
  void  SetLastWireUp1(Float_t wireUp1){fLastWireUp1 = wireUp1;} 
  void  SetROuterFirstWire(Float_t firstWire){fROuterFirstWire = firstWire;}
  void  SetROuterLastWire(Float_t lastWire){fROuterLastWire = lastWire;}   
  //
  //set pad parameter
  //
  void  SetInnerPadPitchLength(Float_t PadPitchLength){  fInnerPadPitchLength=PadPitchLength;}
  void  SetInnerPadPitchWidth(Float_t PadPitchWidth){  fInnerPadPitchWidth = PadPitchWidth;}
  void  SetInnerPadLength(Float_t PadLength){  fInnerPadLength=PadLength;}
  void  SetInnerPadWidth(Float_t PadWidth) {  fInnerPadWidth=PadWidth;} 
  void  SetOuter1PadPitchLength(Float_t PadPitchLength){  fOuter1PadPitchLength=PadPitchLength;}
  void  SetOuter2PadPitchLength(Float_t PadPitchLength){  fOuter2PadPitchLength=PadPitchLength;}
  void  SetOuterPadPitchWidth(Float_t PadPitchWidth){  fOuterPadPitchWidth = PadPitchWidth;}
  void  SetOuter1PadLength(Float_t PadLength){  fOuter1PadLength=PadLength;}
  void  SetOuter2PadLength(Float_t PadLength){  fOuter2PadLength=PadLength;}
  void  SetOuterPadWidth(Float_t PadWidth) {  fOuterPadWidth=PadWidth;}
  void  SetMWPCReadout(Bool_t type) {fBMWPCReadout = type;}
  void  SetNCrossRows(Int_t rows){fNCrossRows = rows;}
  //
  //set gas paremeters
  //
  void  SetDiffT(Float_t DiffT){  fDiffT= DiffT;}
  void  SetDiffL(Float_t DiffL){  fDiffL=DiffL;}
  void  SetGasGain(Float_t GasGain){  fGasGain=GasGain;}
  void  SetDriftV(Float_t DriftV){  fDriftV= DriftV;}
  void  SetOmegaTau(Float_t OmegaTau){  fOmegaTau=OmegaTau;}
  void  SetAttCoef(Float_t AttCoef){  fAttCoef=AttCoef;}
  void  SetOxyCont(Float_t OxyCont){  fOxyCont=OxyCont;}
  void  SetGainSlopesHV(TGraphErrors * gainSlopesHV){ fGainSlopesHV=gainSlopesHV;}
  void  SetGainSlopesPT(TGraphErrors * gainSlopesPT){ fGainSlopesPT=gainSlopesPT;}
  void  SetNominalGainSlopes();
  void  SetComposition(Float_t c1, Float_t c2, Float_t c3, Float_t c4, Float_t c5, Float_t c6){fComposition[0]=c1;
               fComposition[1]=c2;
               fComposition[2]=c3;
               fComposition[3]=c4;
               fComposition[4]=c5;
               fComposition[5]=c6;}
  void   SetFpot(Float_t fpot){fFpot=fpot;}
  void   SetNprim(Float_t prim){fNprim=prim;}
  void   SetNtot(Float_t ntot){fNtot=ntot;}
  void   SetWmean(Float_t wmean){fWmean=wmean;}
  void   SetExp(Float_t exp){fExp=exp;}
  void   SetEend(Float_t end){fEend=end;}
  void   SetBetheBloch(TVectorD *v){
    if (fBetheBloch) delete fBetheBloch;
    fBetheBloch=0;
    if (v) fBetheBloch=new TVectorD(*v);
  }
  static TVectorD * GetBetheBlochParamNa49();
  static TVectorD * GetBetheBlochParamAlice();
  static void RegisterBBParam(TVectorD* param, Int_t position);
  //
  //set electronivc parameters  
  //
  void  SetPadCoupling(Float_t PadCoupling){  fPadCoupling=PadCoupling;}
  void  SetZeroSup(Int_t ZeroSup)    {  fZeroSup=ZeroSup;}
  void  SetNoise(Float_t Noise )     {  fNoise= Noise;}
  void  SetChipGain(Float_t ChipGain){  fChipGain= ChipGain;}
  void  SetChipNorm(Float_t ChipNorm){  fChipNorm= ChipNorm;}
  void  SetTSample(Float_t TSample)  {  fTSample=TSample;}
  void  SetTFWHM(Float_t fwhm)     {  fTSigma=fwhm/2.35;}
  void  SetMaxTBin(Int_t maxtbin)  {  fMaxTBin = maxtbin;}
  void  SetADCSat(Int_t adcsat)    {  fADCSat  = adcsat;}
  void  SetADCDynRange(Float_t adcdynrange) {fADCDynRange = adcdynrange;}
  //
  // High voltage parameters
  //
  void  SetNominalVoltage(Float_t v, UInt_t i)  {if (i<72) fNominalVoltage[i]=v;}
  void  SetMaxVoltageDeviation(Float_t voltage) { fMaxVoltageDeviation=voltage; }
  void  SetMaxDipVoltage(Float_t voltage)       { fMaxDipVoltage=voltage;       }
  void  SetMaxFractionHVbad(Float_t frac )      { fMaxHVfractionBad=frac;       }
  void  SetVoltageDipScanPeriod(Float_t period) { fVoltageDipScanPeriod=period; }
  //
  //set response  parameters  
  //
  void  SetNResponseMax(Int_t max) { fNResponseMax = max;} 
  void  SetResponseThreshold(Int_t threshold) {fResponseThreshold = threshold;}
  //set L1 parameters
  void  SetGateDelay(Float_t delay) {fGateDelay = delay;}
  void  SetL1Delay(Float_t delay) {fL1Delay = delay;}
  void  SetNTBinsBeforeL1(UShort_t nbins) {fNTBinsBeforeL1 = nbins;}
  //
  //get sector parameters
  //
  Float_t  GetInnerRadiusLow() const {return fInnerRadiusLow;}
  Float_t  GetInnerRadiusUp() const {return fInnerRadiusUp;} 
  Float_t  GetOuterRadiusLow() const {return fOuterRadiusLow;} 
  Float_t  GetOuterRadiusUp() const {return fOuterRadiusUp;} 
  Float_t  GetInnerFrameSpace() const {return fInnerFrameSpace;}
  Float_t  GetOuterFrameSpace() const {return fOuterFrameSpace;}
  Float_t  GetInnerWireMount() const {return fInnerWireMount;}
  Float_t  GetOuterWireMount() const {return fOuterWireMount;}
  Float_t  GetInnerAngle() const ;
  Float_t  GetInnerAngleShift() const ;
  Float_t  GetOuterAngle() const ;
  Float_t  GetOuterAngleShift() const ; 
  Int_t    GetNInnerSector() const {return fNInnerSector;}
  Int_t    GetNOuterSector() const {return fNOuterSector;}
  Int_t    GetNSector() const {return fNSector;}
  Float_t  GetZLength(Int_t sector=0) const;
  Int_t    GetGeometryType() const {return fGeometryType;}

  //
  //get wires parameter
  //
  Int_t    GetInnerNWires() const {return fNInnerWiresPerPad;}
  Float_t  GetInnerWWPitch() const {return fInnerWWPitch;}  
  Int_t    GetInnerDummyWire() const {return fInnerDummyWire;}
  Float_t  GetInnerOffWire() const {return fInnerOffWire;}
  Float_t  GetRInnerFirstWire() const {return fRInnerFirstWire;}
  Float_t  GetRInnerLastWire() const {return fRInnerLastWire;}
  Int_t    GetOuter1NWires() const {return fNOuter1WiresPerPad;}
  Int_t    GetOuter2NWires() const {return fNOuter2WiresPerPad;}
  Float_t  GetOuterWWPitch() const {return fOuterWWPitch;}  
  Int_t    GetOuterDummyWire() const {return fOuterDummyWire;}
  Float_t  GetOuterOffWire() const {return fOuterOffWire;}
  Float_t  GetLastWireUp1()  const {return fLastWireUp1;}
  Float_t  GetROuterFirstWire() const {return fROuterFirstWire;}
  Float_t  GetROuterLastWire() const {return fROuterLastWire;}  
  Float_t  GetWWPitch(Int_t isector = 0) const  {
    return ( (isector < fNInnerSector) ? fInnerWWPitch :fOuterWWPitch);} 
  //
  //get pad  parameters
  //
  Float_t  GetInnerPadPitchLength() const {return fInnerPadPitchLength;}
  Float_t  GetInnerPadPitchWidth() const {return fInnerPadPitchWidth;}
  Float_t  GetInnerPadLength() const {return fInnerPadLength;}
  Float_t  GetInnerPadWidth() const  {return fInnerPadWidth;}
  Float_t  GetOuter1PadPitchLength() const {return fOuter1PadPitchLength;}
  Float_t  GetOuter2PadPitchLength() const {return fOuter2PadPitchLength;}  
  Float_t  GetOuterPadPitchWidth() const {return fOuterPadPitchWidth;}
  Float_t  GetOuter1PadLength() const {return fOuter1PadLength;}
  Float_t  GetOuter2PadLength() const {return fOuter2PadLength;}
  Float_t  GetOuterPadWidth()  const {return fOuterPadWidth;}  
  Bool_t   GetMWPCReadout() const {return fBMWPCReadout;}
  Int_t    GetNCrossRows() const {return fNCrossRows;}
  Float_t  GetPadPitchWidth(Int_t isector = 0) const  {
    return ( (isector < fNInnerSector) ? fInnerPadPitchWidth :fOuterPadPitchWidth);}
  Float_t  GetPadPitchLength(Int_t isector = 0, Int_t padrow=0)  const
  { if (isector < fNInnerSector) return fInnerPadPitchLength; 
    else return ((padrow<fNRowUp1) ? fOuter1PadPitchLength:fOuter2PadPitchLength);}
  Int_t GetNRowLow() const;   //get the number of pad rows in low sector
  Int_t GetNRowUp() const;    //get the number of pad rows in up sector
  Int_t GetNRowUp1() const;  // number of short rows in up sector  
  Int_t GetNRowUp2() const;  // number of long rows in up sector
  Int_t GetNRow(Int_t isec) const {return  ((isec<fNInnerSector) ?  fNRowLow:fNRowUp);}
  Int_t GetNRowsTotal() const {return fNtRows;}  //get total nuber of rows
  Float_t GetPadRowRadiiLow(Int_t irow) const; //get the pad row (irow) radii
  Float_t GetPadRowRadiiUp(Int_t irow) const;  //get the pad row (irow) radii
  Float_t GetPadRowRadii(Int_t isec,Int_t irow) const {
    return ( (isec < fNInnerSector) ?GetPadRowRadiiLow(irow):GetPadRowRadiiUp(irow));}
    //retrun radii of the pad row irow in sector i
  Int_t GetNPadsLow(Int_t irow) const;    //get the number of pads in row irow 
  Int_t GetNPadsUp(Int_t irow) const;     //get the number of pads in row irow
  Int_t GetNPads(Int_t isector,Int_t irow) const{
     return ( (isector < fNInnerSector) ?GetNPadsLow(irow) : GetNPadsUp(irow));} 
  Int_t GetWireSegment(Int_t sector, Int_t row) const ;    // get Anode wire segment index IROC --> [0,4], OROC[0,7]
  Int_t GetNPadsPerSegment(Int_t segmentID) const;         // get number of pads for a given Anode wire segment

  Float_t GetYInner(Int_t irow) const; // wire length in low sec row
  Float_t GetYOuter(Int_t irow) const; // wire length in up sec row  
  Int_t GetSectorIndex(Float_t angle, Int_t row, Float_t z) const; // get sector index
  Float_t GetChamberCenter(Int_t isec, Float_t * center = 0) const; // get readout chamber positions
  TGeoHMatrix *GetTrackingMatrix(Int_t isec) const {
    return fTrackingMatrix[isec];}
  TGeoHMatrix *GetClusterMatrix(Int_t isec) const {
    return fClusterMatrix[isec];}
  TGeoHMatrix *GetGlobalMatrix(Int_t isec) const {
    return fGlobalMatrix[isec];}
  Bool_t   IsGeoRead(){ return fGlobalMatrix!=0;}
  //
  //get GAS parameters 
  //
  Float_t  GetDiffT() const {return fDiffT;}
  Float_t  GetDiffL() const {return fDiffL;}
  Float_t  GetGasGain() const {return fGasGain;}
  Float_t  GetDriftV() const {return fDriftV;}
  Float_t  GetOmegaTau() const {return fOmegaTau;}
  Float_t  GetAttCoef() const {return fAttCoef;}
  Float_t  GetOxyCont() const {return fOxyCont;} 
  TGraphErrors * GetGainSlopesHV() const { return fGainSlopesHV;}
  TGraphErrors * GetGainSlopesPT() const { return fGainSlopesPT;}
  Float_t* GetComposition() {return fComposition;}
  Float_t  GetFpot()const {return fFpot;}
  Float_t  GetNprim() const {return fNprim;}
  Float_t  GetNtot() const {return fNtot;}
  Float_t  GetWmean()const {return fWmean;}
  Float_t  GetExp()const {return fExp;}
  Float_t  GetEend()const {return fEend;}
  TVectorD* GetBetheBlochParameters(){return fBetheBloch;} 
  static Double_t BetheBlochAleph(Double_t bb, Int_t type=0);
  //
  //get Electronic parameters
  //
  Float_t  GetPadCoupling() const {return fPadCoupling;}
  Int_t    GetZeroSup() const {return fZeroSup;}
  Float_t  GetNoise() const {return fNoise;}
  Float_t  GetChipGain() const {return fChipGain;}
  Float_t  GetChipNorm() const {return fChipNorm;}
  Float_t  GetTSample() const {return fTSample;}
  Float_t  GetZWidth() const {return fZWidth;}
  Float_t  GetTFWHM() const {return fTSigma*2.35;}
  Float_t  GetZSigma() const {return fTSigma*fDriftV;}  
  virtual  Float_t  GetZOffset() const {return 3*fTSigma*fDriftV;}
  Int_t    GetMaxTBin() const {return fMaxTBin;}
  Int_t    GetADCSat() const {return fADCSat;}
  Float_t  GetADCDynRange() const {return fADCDynRange;}
  Float_t  GetTotalNormFac() const {return fTotalNormFac;}
  Float_t  GetNoiseNormFac() const {return fNoiseNormFac;}
  //
  // High voltage parameters
  //
  Float_t  GetNominalVoltage(UInt_t i) const {return (i<72)?fNominalVoltage[i]:0;} //0-35:IROC, 36-71:OROC
  Float_t  GetMaxVoltageDeviation()    const { return fMaxVoltageDeviation;      }
  Float_t  GetMaxDipVoltage()          const { return fMaxDipVoltage;            }
  Float_t  GetMaxFractionHVbad()       const { return fMaxHVfractionBad;         }
  Float_t  GetVoltageDipScanPeriod()   const { return fVoltageDipScanPeriod;     }
  
  //
  // get response data
  //  
  Int_t * GetResBin(Int_t i);  
  //return response bin i  - bin given by  padrow [0] pad[1] timebin[2] 
  Float_t & GetResWeight(Int_t i);
  //return  weight of response bin i

  // get L1 data
  Float_t  GetGateDelay() const {return fGateDelay;}
  Float_t  GetL1Delay() const {return fL1Delay;}
  UShort_t GetNTBinsBeforeL1() const {return fNTBinsBeforeL1;}
  Float_t  GetNTBinsL1() const {return fNTBinsL1;}
protected :

  Bool_t fbStatus;  //indicates consistency of the data
  //---------------------------------------------------------------------
  //   ALICE TPC sector geometry
  //--------------------------------------------------------------------  
  Float_t fInnerRadiusLow;    // lower radius of inner sector-IP
  Float_t fInnerRadiusUp;     // upper radius of inner  sector-IP
  Float_t fOuterRadiusUp;     // upper radius of outer  sector-IP
  Float_t fOuterRadiusLow;    // lower radius of outer sector-IP
  Float_t fInnerAngle;        //opening angle of Inner sector
  Float_t fInnerAngleShift;   //shift of first inner sector center to the 0
  Float_t fOuterAngle;        //opening angle of outer sector
  Float_t fOuterAngleShift;   //shift of first sector center to the 0  
  Float_t fInnerFrameSpace;   //space for inner frame in the phi direction 
  Float_t fOuterFrameSpace;   //space for outer frame in the phi direction 
  Float_t fInnerWireMount;    //space for wire mount, inner sector
  Float_t fOuterWireMount;    //space for wire mount, outer sector
  Int_t   fNInnerSector;      //number of inner sectors             -calculated
  Int_t   fNOuterSector;      //number of outer sectors             -calculated
  Int_t   fNSector;           // total number of sectors            -calculated
  Float_t fZLength;           //length of the drift region of the TPC
  Float_t *fRotAngle;         //[fNSector]  sin and cos of rotation angles for 
                              //  diferent sectors -calculated
  Int_t   fGeometryType;      //type of geometry -0 straight rows
  //  Float_t *fChamberPos;       //[fNSector] displacements of the readout chambers 
                              //with respect to the 'idead' geometry
                              //in local corrdinate system
  //  Float_t *fChamberRot;       //[fNSector] rotation angles of the readout chambers 
                              //with respect to the 'idead' geometry
                              //in local corrdinate system
  TGeoHMatrix **fTrackingMatrix;   //![fNSector] transformation matrices of the tracking
                              //coordinate system
  TGeoHMatrix **fClusterMatrix;    //![fNSector] transformation matrices of the cluster
                              //coordinate system
  TGeoHMatrix **fGlobalMatrix;    //![fNSector] fTrackingMatrix * fClusterMatrix

  //1-cylindrical
  //---------------------------------------------------------------------
  //   ALICE TPC wires  geometry - for GEM we can consider that it is gating  
  //--------------------------------------------------------------------
  Int_t   fNInnerWiresPerPad; //Number of wires per pad
  Float_t fInnerWWPitch;      //pitch between wires  in inner sector     - calculated
  Int_t   fInnerDummyWire;    //number of wires without pad readout
  Float_t fInnerOffWire;      //oofset of first wire to the begining of the sector
  Float_t fRInnerFirstWire;   //position of the first wire                -calculated
  Float_t fRInnerLastWire;    //position of the last wire                 -calculated
  Float_t fLastWireUp1;     //position of the last wire in outer1 sector
  Int_t   fNOuter1WiresPerPad; //Number of wires per pad
  Int_t   fNOuter2WiresPerPad; // Number of wires per pad
  Float_t fOuterWWPitch;      //pitch between wires in outer sector      -calculated
  Int_t   fOuterDummyWire;    //number of wires without pad readout
  Float_t fOuterOffWire;      //oofset of first wire to the begining of the sector
  Float_t fROuterFirstWire;   //position of the first wire                -calulated
  Float_t fROuterLastWire;    //position of the last wire                 -calculated 
  //---------------------------------------------------------------------
  //   ALICE TPC pad parameters
  //--------------------------------------------------------------------
  Float_t   fInnerPadPitchLength;    //Inner pad pitch length
  Float_t   fInnerPadPitchWidth;     //Inner pad pitch width
  Float_t   fInnerPadLength;         //Inner pad  length
  Float_t   fInnerPadWidth;          //Inner pad  width
  Float_t   fOuter1PadPitchLength;    //Outer pad pitch length
  Float_t   fOuter2PadPitchLength;    //Outer pad pitch length
  Float_t   fOuterPadPitchWidth;     //Outer pad pitch width
  Float_t   fOuter1PadLength;         //Outer pad  length
  Float_t   fOuter2PadLength;         //Outer pad length
  Float_t   fOuterPadWidth;          //Outer pad  width
  Bool_t    fBMWPCReadout;           //indicate wire readout - kTRUE or GEM readout -kFALSE
  Int_t     fNCrossRows;             //number of rows to crostalk calculation
      
  Int_t fNRowLow;           //number of pad rows per low sector        -set
  Int_t fNRowUp1;            //number of short pad rows per sector up  -set
  Int_t fNRowUp2;            //number of long pad rows per sector up   -set
  Int_t fNRowUp;            //number of pad rows per sector up     -calculated
  Int_t fNtRows;            //total number of rows in TPC          -calculated
  Float_t  fPadRowLow[600]; //Lower sector, pad row radii          -calculated
  Float_t  fPadRowUp[600];  //Upper sector, pad row radii          -calculated 
  Int_t    fNPadsLow[600];  //Lower sector, number of pads per row -calculated
  Int_t    fNPadsUp[600];   //Upper sector, number of pads per row -calculated
  Float_t  fYInner[600];     //Inner sector, wire-length
  Float_t  fYOuter[600];     //Outer sector, wire-length   
  //---------------------------------------------------------------------
  //   ALICE TPC Gas Parameters
  //--------------------------------------------------------------------
  Float_t  fDiffT;          //tangencial diffusion constant
  Float_t  fDiffL;          //longutudinal diffusion constant
  Float_t  fGasGain;        //gas gain constant
  Float_t  fDriftV;         //drift velocity constant
  Float_t  fOmegaTau;       //omega tau ExB coeficient
  Float_t  fAttCoef;        //attachment coefitients
  Float_t  fOxyCont;        //oxygen content
  Float_t  fFpot;            // first ionisation potential
  Float_t  fNprim;           // number of primary electrons/cm
  Float_t  fNtot;            //total number of electrons/c (MIP)
  Float_t  fWmean;           // mean energy for electron/ion pair
  Float_t  fExp;             // de = f(E) - energy loss parametrization
  Float_t  fEend;            // upper cutoff for de generation
  TVectorD*  fBetheBloch;   // Bethe-Bloch parametrization
  // gas mixture composition
  Float_t  fComposition[6]; 
  TGraphErrors * fGainSlopesHV;   // graph with the gain slope as function of HV - per chamber
  TGraphErrors * fGainSlopesPT;   // graph with the gain slope as function of P/T - per chamber
  //---------------------------------------------------------------------
  //   ALICE TPC  Electronics Parameters
  //--------------------------------------------------------------------
  Float_t fPadCoupling;     //coupling factor ration of  anode signal 
                            //and total pads signal  
  Int_t fZeroSup;           //zero suppresion constant
  Float_t fNoise;           //noise sigma constant
  Float_t fChipGain;        //preamp shaper constant
  Float_t fChipNorm;         //preamp shaper normalisation 
  Float_t fTSample;         //sampling time
  Float_t fZWidth;          //derived value calculated using TSample and driftw  -computed
  Float_t fTSigma;          //width of the Preamp/Shaper function
  Int_t   fMaxTBin;         //maximum time bin number
  Int_t   fADCSat;          //saturation value of ADC (10 bits)
  Float_t fADCDynRange;     //input dynamic range (mV)
  Float_t fTotalNormFac;    //full normalisation factor - calculated
  Float_t fNoiseNormFac;    //normalisation factor to transform noise in electron to ADC channel
  //---------------------------------------------------------------------
  // High voltage parameters
  //---------------------------------------------------------------------
  Float_t fNominalVoltage[72];  //nominal voltage in [V] per chamber
  Float_t fMaxVoltageDeviation; // maximum voltage deviation from nominal voltage before a chamber is masked
  Float_t fMaxDipVoltage;       // maximum voltage deviation from median before a dip event is marked
  Float_t fMaxHVfractionBad;    // maximum fraction of bad HV entries (deviation from Median) before a chamber is marked bad
  Float_t fVoltageDipScanPeriod; // scanning period to detect a high volrage dip: event time stamp +- fVoltageDipScanPeriod [sec]
  
  //---------------------------------------------------------------------
  // ALICE TPC response data 
  //---------------------------------------------------------------------
  Int_t   fNResponseMax;   //maximal dimension of response        
  Float_t fResponseThreshold; //threshold for accepted response   
  Int_t   fCurrentMax;     //!current maximal dimension            -calulated 
  Int_t   *fResponseBin;    //!array with bins                     -calulated
  Float_t *fResponseWeight; //!array with response                 -calulated

  //---------------------------------------------------------------------
  //   ALICE TPC L1 Parameters
  //--------------------------------------------------------------------
  Float_t fGateDelay;       //Delay of L1 arrival for the TPC gate signal
  Float_t fL1Delay;         //Delay of L1 arrival for the TPC readout 
  UShort_t fNTBinsBeforeL1; //Number of time bins before L1 arrival which are being read out 
  Float_t fNTBinsL1;        //Overall L1 delay in time bins
 protected:
  static TObjArray *fBBParam; // array of the Bethe-Bloch parameters. 
 private:
  AliTPCParam(const AliTPCParam &);
  AliTPCParam & operator=(const AliTPCParam &);

  void CleanGeoMatrices();

  ClassDef(AliTPCParam,8)  //parameter  object for set:TPC
};

 
inline Int_t * AliTPCParam::GetResBin(Int_t i)
{
  //return response bin i  - bin given by  padrow [0] pad[1] timebin[2] 
  if (i<fCurrentMax) return &fResponseBin[i*3];
  else return 0;
}
  
inline Float_t &AliTPCParam::GetResWeight(Int_t i)
{
  //return  weight of response bin i
  if (i<fCurrentMax) return fResponseWeight[i];
  else return fResponseWeight[i];
}


inline void  AliTPCParam::AdjustCosSin(Int_t isec, Float_t &cos, Float_t &sin) const
{
  //
  //set cosinus and sinus of rotation angles for sector isec
  //
  cos=fRotAngle[isec*4];
  sin=fRotAngle[isec*4+1];
}

inline Float_t   AliTPCParam::GetAngle(Int_t isec) const
{
  //
  //return rotation angle of given sector
  //
  return fRotAngle[isec*4+2];
}


inline void AliTPCParam::Transform1to2Ideal(Float_t *xyz, Int_t *index) const
{
  //transformation to rotated coordinates
  //we must have information about sector!
  //rotate to given sector
  // ideal frame

  Float_t cos,sin;
  AdjustCosSin(index[1],cos,sin);
  Float_t x1=xyz[0]*cos + xyz[1]*sin;
  Float_t y1=-xyz[0]*sin + xyz[1]*cos;
  xyz[0]=x1;
  xyz[1]=y1;
  xyz[2]=fZLength-TMath::Abs(xyz[2]);
  index[0]=2;
}


inline void AliTPCParam::Transform1to2(Float_t *xyz, Int_t *index) const
{
  //transformation to rotated coordinates 
  //we must have information about sector!
  //rotate to given sector
  Double_t xyzmaster[3] = {xyz[0],xyz[1],xyz[2]};
  Double_t xyzlocal[3]={0,0,0};
  if (index[1]>=0 && index[1]<fNSector) 
    fGlobalMatrix[index[1]]->MasterToLocal(xyzmaster,xyzlocal);
  xyz[0] = xyzlocal[0];
  xyz[1] = xyzlocal[1];
  xyz[2] = xyzlocal[2];
  index[0]=2;
}




inline void AliTPCParam::Transform2to1(Float_t *xyz, Int_t *index) const
{
  //
  //transformation from  rotated coordinates to global coordinates
  //
  Float_t cos,sin;
  AdjustCosSin(index[1],cos,sin);   
  Float_t x1=xyz[0]*cos - xyz[1]*sin;
  Float_t y1=xyz[0]*sin + xyz[1]*cos; 
  xyz[0]=x1;
  xyz[1]=y1;
  xyz[2]=fZLength-xyz[2]; 
  if (index[1]<fNInnerSector)
    {if ( index[1]>=(fNInnerSector>>1))	xyz[2]*=-1.;}
  else 
    {if ( (index[1]-fNInnerSector) >= (fNOuterSector>>1) )    xyz[2]*=-1;}
  index[0]=1;
}

inline void AliTPCParam::Transform2to2(Float_t *xyz, Int_t *index, Int_t *oindex) const
{
  //transform rotated coordinats of one sector to rotated
  //coordinates relative to another sector
  Transform2to1(xyz,index);
  Transform1to2(xyz,oindex);
  index[0]=2;
  index[1]=oindex[1];  
}

inline Float_t  AliTPCParam::Transform2to2NearestWire(Float_t *xyz, Int_t *index)  const
{
  //
  // asigns the x-position of the closest wire to xyz[0], return the
  // electron to closest wire distance
  //
  Float_t xnew,dx;
  if (index[1]<fNInnerSector) {
     xnew = fRInnerFirstWire+TMath::Nint((xyz[0]-fRInnerFirstWire)/fInnerWWPitch)*fInnerWWPitch;
    }
    else {
     xnew = fROuterFirstWire+TMath::Nint((xyz[0]-fROuterFirstWire)/fOuterWWPitch)*fOuterWWPitch;
    }
  dx = xnew-xyz[0];
  xyz[0]=xnew;
  return  dx;
}

inline Int_t   AliTPCParam::Transform2to3(Float_t *xyz, Int_t *index)  const
{
  //
  //calulates coresponding pad row number, sets index[2] for straight rows
  //does not change xyz[] information
  //valid only for straight row
  //
  if  (index[1]<fNInnerSector)   
    index[2] =TMath::Nint((xyz[0]-fPadRowLow[0])/fInnerPadPitchLength);
  else
    if (xyz[0] < fLastWireUp1 )
      index[2] = TMath::Nint((xyz[0]-fPadRowUp[0])/fOuter1PadPitchLength);
    else 
      index[2] = TMath::Nint(fNRowUp1+(xyz[0]-fPadRowUp[64])/fOuter2PadPitchLength);
  index[0]=3;
  return index[2];
}

inline void   AliTPCParam::Transform3to4(Float_t *xyz, Int_t *index)  const
{
  //
  //valid only for straight rows straight rows
  //calculate xyz[0] position relative to given index
  //
  if  (index[1]<fNInnerSector)   
    xyz[0] -=index[2]*fInnerPadPitchLength+fPadRowLow[0];
  else
    if (index[2]<fNRowUp1)
      xyz[0] -=index[2]*fOuter1PadPitchLength+fPadRowUp[0];
    else 
      xyz[0] -=(index[2]-fNRowUp1)*fOuter2PadPitchLength+fPadRowUp[64];
  index[0]  =4;
}

inline void   AliTPCParam::Transform4to3(Float_t *xyz, Int_t *index) const
{
  //
  //valid only for straight rows 
  //transforms  relative xyz[0] to the global one within given sector
  //
  if  (index[1]<fNInnerSector)   
    xyz[0] +=index[2]*fInnerPadPitchLength+fPadRowLow[0];
  else
    if(index[2]<fNRowUp1)
      xyz[0] +=index[2]*fOuter1PadPitchLength+fPadRowUp[0];
    else 
      xyz[0] +=index[2]*fOuter2PadPitchLength+fPadRowUp[64];
  index[0]  =3;
}


inline void   AliTPCParam::Transform2to5( Float_t *xyz, Int_t *index) const
{
  //
  //transform [x,y,z] to [r,phi,z]
  //
  Float_t angle;
  Float_t r = TMath::Sqrt(xyz[0]*xyz[0]+xyz[1]*xyz[1]);
  if ((xyz[0]==0)&&(xyz[1]==0)) angle = 0;
  else
    {
      angle =TMath::ASin(xyz[1]/r);
      if   (xyz[0]<0)   angle=TMath::Pi()-angle;
      if ( (xyz[0]>0) && (xyz[1]<0) ) angle=2*TMath::Pi()+angle;
    }
  xyz[0]=r;
  xyz[1]=angle;
  index[0]=5;
}

inline void   AliTPCParam::Transform5to2( Float_t *xyz, Int_t *index)  const
{
  //
  //transform [r,rphi,z] to [x,y,z] 
  //
  Float_t r = xyz[0];
  Float_t angle= xyz[1];
  xyz[0]=r*TMath::Cos(angle);
  xyz[1]=r*TMath::Sin(angle);
  index[0]=2;
}

inline void AliTPCParam::Transform4to8(Float_t *xyz, Int_t *index) const
{
  //
  //transform xyz coordinates to 'digit' coordinates
  //

  if (index[1]<fNInnerSector) {
    if ( index[1]>=(fNInnerSector>>1)) xyz[1]*=-1.;
  }
  else {
    if ( (index[1]-fNInnerSector) >= (fNOuterSector>>1) ) xyz[1]*=-1;      
  }

  xyz[2]/=fZWidth;  
  if  (index[1]<fNInnerSector) {    
    xyz[0]/=fInnerPadPitchLength;
    xyz[1]/=fInnerPadPitchWidth;
  }
  else{  
    xyz[1]/=fOuterPadPitchWidth;  
    if (index[2]<fNRowUp1 ) xyz[0]/=fOuter1PadPitchLength;      
    else xyz[0]/=fOuter2PadPitchLength;     
  }
  xyz[1]-=0.5;
  index[0]=8;
}

inline void AliTPCParam::Transform8to4(Float_t *xyz, Int_t *index) const
{
  //
  //transforms 'digit' coordinates to xyz coordinates
  //
  if (index[1]<fNInnerSector) {
    if ( index[1]>=(fNInnerSector>>1)) xyz[1]*=-1.;
  }
  else {
    if ( (index[1]-fNInnerSector) >= (fNOuterSector>>1) ) xyz[1]*=-1;      
  }

  xyz[2]*=fZWidth;
  if  (index[1]<fNInnerSector) {    
    xyz[0]*=fInnerPadPitchLength;
    xyz[1]*=fInnerPadPitchWidth;    
  }
  else{  
    xyz[1]*=fOuterPadPitchWidth;  
    if (index[2] < fNRowUp1 ) xyz[0]*=fOuter1PadPitchLength;
    else xyz[0]*=fOuter2PadPitchLength;           
  } 
  index[0]=4;
}

inline void  AliTPCParam::Transform6to8(Float_t *xyz, Int_t *index) const
{
  //
  //transforms cylindrical xyz coordinates to 'digit' coordinates
  //
  xyz[2]/=fZWidth;
  if  (index[1]<fNInnerSector) {    
    xyz[0]/=fInnerPadPitchLength;
    xyz[1]*=xyz[0]/fInnerPadPitchWidth;
  }
  else{ 
    xyz[1]*=xyz[0]/fOuterPadPitchWidth;
    if (index[2] < fNRowUp1 ) xyz[0]/=fOuter1PadPitchLength;
    else xyz[0]/=fOuter2PadPitchLength;       
  }
  index[0]=8;
}

inline void  AliTPCParam::Transform8to6(Float_t *xyz, Int_t *index) const
{
  //
  //transforms 'digit' coordinates to cylindrical xyz coordinates 
  //
  xyz[2]*=fZWidth;
  if  (index[1]<fNInnerSector) {    
    xyz[0]*=fInnerPadPitchLength;
    xyz[1]/=xyz[0]/fInnerPadPitchWidth;
  }
  else{ 
    xyz[1]/=xyz[0]/fOuterPadPitchWidth;  
    if (index[2] < fNRowUp1 ) xyz[0]*=fOuter1PadPitchLength;
    else xyz[0]*=fOuter2PadPitchLength;                
  }  
  index[0]=6;
}
inline Float_t AliTPCParam::GetZLength(Int_t sector) const
{ if(sector <18 || (sector>35&&sector<54)) return   fZLength-0.275;
  else return fZLength-0.302;
}
#endif  

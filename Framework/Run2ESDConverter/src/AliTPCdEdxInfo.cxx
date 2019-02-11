
//##################################################################
//
// Simple class to store TPC dE/dx info for different pad regions.
//
// Origin: Marian Ivanov, Alexander Kalweit 
//
//##################################################################

#include "AliTPCdEdxInfo.h"
#include "TObjArray.h"
#include "TGraphErrors.h"
#include "AliExternalTrackParam.h"


TObjArray * AliTPCdEdxInfo::fArraySectorCalibration=0;

ClassImp(AliTPCdEdxInfo)

AliTPCdEdxInfo::AliTPCdEdxInfo():
  TObject(),
  fTPCsignalRegion(),
  fTPCsignalRegionQmax(),
  fTPCsignalNRegion(),
  fTPCsignalNRowRegion()
{
  // Default constructor
  for (Int_t i=0;i<3; i++){
    fTPCsignalRegion[i]=0;
    fTPCsignalRegionQmax[i]=0;
    fTPCsignalNRegion[i]=0;
    fTPCsignalNRowRegion[i]=0;
  }
  fTPCsignalRegion[3]=0;
  fTPCsignalRegionQmax[3]=0;
  
}

//_______________________________________________________________________________________________
AliTPCdEdxInfo::AliTPCdEdxInfo(const AliTPCdEdxInfo& source):
    TObject(),
    fTPCsignalRegion(),
    fTPCsignalRegionQmax(),
    fTPCsignalNRegion(),
    fTPCsignalNRowRegion()
{
    //
    // copy constructor
    //
    for (Int_t i=0;i<3; i++){
      fTPCsignalRegion[i]     = source.fTPCsignalRegion[i];
      fTPCsignalRegionQmax[i] = source.fTPCsignalRegionQmax[i];
      fTPCsignalNRegion[i]    = source.fTPCsignalNRegion[i];
      fTPCsignalNRowRegion[i] = source.fTPCsignalNRowRegion[i];
    }
    fTPCsignalRegion[3]       = source.fTPCsignalRegion[3];
    fTPCsignalRegionQmax[3]   = source.fTPCsignalRegionQmax[3];
    
}

//_______________________________________________________________________________________________
AliTPCdEdxInfo& AliTPCdEdxInfo::operator=(const AliTPCdEdxInfo& source)
{
    //
    // assignment operator
    //

  if (&source == this) return *this;
  TObject::operator=(source);

  for (Int_t i=0;i<3; i++){
    fTPCsignalRegion[i]     = source.fTPCsignalRegion[i];
    fTPCsignalRegionQmax[i] = source.fTPCsignalRegionQmax[i];
    fTPCsignalNRegion[i]    = source.fTPCsignalNRegion[i];
    fTPCsignalNRowRegion[i] = source.fTPCsignalNRowRegion[i];
  }
  fTPCsignalRegion[3]       = source.fTPCsignalRegion[3];
  fTPCsignalRegionQmax[3]   = source.fTPCsignalRegionQmax[3];
  
  return *this;

}

//_______________________________________________________________________________________________
void  AliTPCdEdxInfo::GetTPCSignalRegionInfo(Double_t signal[4], Char_t ncl[3], Char_t nrows[3]) const {
  //
  // Get the TPC dEdx variables per region
  //
  // Double32_t  fTPCsignalRegion[4]; // TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used)  
  // Char_t      fTPCsignalNRegion[3]; // number of clusters above threshold used in the dEdx calculation
  // Char_t      fTPCsignalNRowRegion[3]; // number of crosed rows used in the dEdx calculation - signal below threshold included
  //
  for (Int_t i=0; i<3; i++){
    signal[i]=fTPCsignalRegion[i];
    ncl[i]=fTPCsignalNRegion[i];
    nrows[i]=fTPCsignalNRowRegion[i];
  }
  signal[3]=fTPCsignalRegion[3];
  return; 
}

//_______________________________________________________________________________________________
void  AliTPCdEdxInfo::GetTPCSignals(Double_t signal[4]) const {
  //
  // Set the TPC dEdx variables per region
  //
  // Double32_t  fTPCsignalRegionQmax[4]; // TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used)
  //
  for (Int_t i=0;i<4; i++){
    signal[i]=fTPCsignalRegion[i];
  }
}

//_______________________________________________________________________________________________
void  AliTPCdEdxInfo::SetTPCSignalRegionInfo(const Double_t signal[4], const Char_t ncl[3], const Char_t nrows[3]){
  //
  // Set the TPC dEdx variables per region
  //
  // Double32_t  fTPCsignalRegion[4]; // TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used)  
  // Char_t      fTPCsignalNRegion[3]; // number of clusters above threshold used in the dEdx calculation
  // Char_t      fTPCsignalNRowRegion[3]; // number of crosed rows used in the dEdx calculation - signal below threshold included
  //
  for (Int_t i=0;i<3; i++){
    fTPCsignalRegion[i]=signal[i];
    fTPCsignalNRegion[i]=ncl[i];
    fTPCsignalNRowRegion[i]=nrows[i];
  }
  fTPCsignalRegion[3]=signal[3];
  return;
}

//_______________________________________________________________________________________________
void  AliTPCdEdxInfo::SetTPCSignals(const Double_t signal[4]){
  //
  // Set the TPC dEdx variables per region
  //
  // Double32_t  fTPCsignalRegionQmax[4]; // TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used)
  //
  for (Int_t i=0;i<4; i++){
    fTPCsignalRegion[i]=signal[i];
  }
}

//_______________________________________________________________________________________________
void  AliTPCdEdxInfo::GetTPCSignalRegionInfoQmax(Double_t signal[4], Char_t ncl[3], Char_t nrows[3]) const {
  //
  // Get the TPC dEdx variables per region
  //
  // Double32_t  fTPCsignalRegionQmax[4]; // TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used)
  // Char_t      fTPCsignalNRegion[3]; // number of clusters above threshold used in the dEdx calculation
  // Char_t      fTPCsignalNRowRegion[3]; // number of crosed rows used in the dEdx calculation - signal below threshold included
  //
  for (Int_t i=0; i<3; i++){
    signal[i]=fTPCsignalRegionQmax[i];
    ncl[i]=fTPCsignalNRegion[i];
    nrows[i]=fTPCsignalNRowRegion[i];
  }
  signal[3]=fTPCsignalRegionQmax[3];
  return;
}

//_______________________________________________________________________________________________
void  AliTPCdEdxInfo::GetTPCSignalsQmax(Double_t signal[4]) const {
  //
  // Set the TPC dEdx variables per region
  //
  // Double32_t  fTPCsignalRegionQmax[4]; // TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used)
  //
  for (Int_t i=0;i<4; i++){
    signal[i]=fTPCsignalRegionQmax[i];
  }
}

//_______________________________________________________________________________________________
void  AliTPCdEdxInfo::SetTPCSignalRegionInfoQmax(const Double_t signal[4], const Char_t ncl[3], const Char_t nrows[3]){
  //
  // Set the TPC dEdx variables per region
  //
  // Double32_t  fTPCsignalRegionQmax[4]; // TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used)
  // Char_t      fTPCsignalNRegion[3]; // number of clusters above threshold used in the dEdx calculation
  // Char_t      fTPCsignalNRowRegion[3]; // number of crosed rows used in the dEdx calculation - signal below threshold included
  //
  for (Int_t i=0;i<3; i++){
    fTPCsignalRegionQmax[i]=signal[i];
    fTPCsignalNRegion[i]=ncl[i];
    fTPCsignalNRowRegion[i]=nrows[i];
  }
  fTPCsignalRegionQmax[3]=signal[3];
  return;
}

//_______________________________________________________________________________________________
void  AliTPCdEdxInfo::SetTPCSignalsQmax(const Double_t signal[4]){
  //
  // Set the TPC dEdx variables per region
  //
  // Double32_t  fTPCsignalRegionQmax[4]; // TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used)
  //
  for (Int_t i=0;i<4; i++){
    fTPCsignalRegionQmax[i]=signal[i];
  }
}


Double_t AliTPCdEdxInfo::GetWeightedMean(Int_t qType, Int_t wType, Double_t w0, Double_t w1, Double_t w2) const
{
  //
  // Get weighted mean of the dEdx information
  //
  const Double_t *info = (qType==0)? fTPCsignalRegion :  fTPCsignalRegionQmax;
  const Char_t *ninfo = (wType==0)? fTPCsignalNRegion:  fTPCsignalNRowRegion;
  Double_t weight[3]={w0,w1,w2};
  Double_t sum=0;
  Double_t sumw=0;
  for (Int_t i=0; i<3; i++){
    sum+= info[i]*Double_t(ninfo[i])*weight[i];
    sumw+= ninfo[i]*weight[i];
  }
  Double_t result = (sumw>0) ? sum/sumw:0;
  return result;
}

//
// Apply second order calibration  of the dEdx
//

void  AliTPCdEdxInfo::RegisterSectorCalibration(TGraphErrors* gainSector, Int_t regionID, Int_t calibID){
  //
  // Register sector calibration
  //
  // create if arrray does not exist
  if (!fArraySectorCalibration) fArraySectorCalibration= new TObjArray((calibID+1)*3*10); // boook space for calibration pointer
  // resize if not enough booked
  if (fArraySectorCalibration->GetSize()<(calibID+1)*3) fArraySectorCalibration->Expand((calibID+1)*3);
  //
  fArraySectorCalibration->AddAt(gainSector, 3*calibID+regionID);
}

// Double_t AliTPCdEdxInfo::GetNormalizeddEdx(AliExternalTrackParam *param, Double_t bz,  Int_t regionID, Int_t calibID, Int_t qID){
//   //
//   //
//   // 
//   static AliTPCParamSR paramSR;
//   static Double_t radius[3] ={0.5*(paramSR.GetInnerRadiusLow()+paramSR.GetInnerRadiusUp()),
// 			      0.5*(paramSR.GetPadRowRadii(36,0)+paramSR.GetPadRowRadii(36,paramSR.GetNRowUp1()-1)),
// 			      0.5*(paramSR.GetPadRowRadii(36,0)+paramSR.GetPadRowRadii(36,paramSR.GetNRowUp()-1))};
//   Double_t phi= param->GetParameterAtRadius(radius[regionID],bz,7);

//   TGraphErrors * graphSectorCorection = fArraySectorCalibration->At(regionID+3*calibID);
//   Double_t dEdx = 0;
//   if (qID==0) dEdx = fTPCsignalRegion[regionID];
//   if (qID==1) dEdx = fTPCsignalRegionQmax[regionID];
//   if (graphSectorCorection) dEdx /=graphSectorCorection->EvalAt(sector);
//   return dEdx;
// }



Double_t   AliTPCdEdxInfo::GetdEdxInfo(AliExternalTrackParam *param, Int_t regionID, Int_t calibID, Int_t qID, Int_t valueID){
  //
  //
  //

  return param->GetParameter()[regionID];
}



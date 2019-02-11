#ifndef AliTPCdEdxInfo_H
#define AliTPCdEdxInfo_H

class TGraphErrors;
class TObjArray;
class AliExternalTrackParam;
#include <TObject.h>

class AliTPCdEdxInfo : public TObject 
{
public:
  AliTPCdEdxInfo();
  AliTPCdEdxInfo(const AliTPCdEdxInfo& source);
  AliTPCdEdxInfo& operator=(const AliTPCdEdxInfo& source);
  Double_t GetWeightedMean(Int_t qType, Int_t wType, Double_t w0, Double_t w1, Double_t w2) const;
  Double_t GetFractionOfClusters(Int_t iregion){ return fTPCsignalNRowRegion[iregion]>0 ? Double_t(fTPCsignalNRegion[iregion])/Double_t(fTPCsignalNRowRegion[iregion]):0;}
  //
  // qTot info
  void     GetTPCSignalRegionInfo(Double_t signal[4], Char_t ncl[3], Char_t nrows[3]) const;
  void     GetTPCSignals(Double_t signal[4]) const;

  void     SetTPCSignalRegionInfo(const Double_t signal[4], const Char_t ncl[3], const Char_t nrows[3]);
  void     SetTPCSignals(const Double_t signal[4]);
  
  
  // qMax info
  void     GetTPCSignalRegionInfoQmax(Double_t signal[4], Char_t ncl[3], Char_t nrows[3]) const;
  void     GetTPCSignalsQmax(Double_t signal[4]) const;

  void     SetTPCSignalRegionInfoQmax(const Double_t signal[4], const Char_t ncl[3], const Char_t nrows[3]);
  void     SetTPCSignalsQmax(const Double_t signal[4]);
  
  Double_t GetSignalTot(Int_t index) const { return fTPCsignalRegion[index];}
  Double_t GetSignalMax(Int_t index) const { return fTPCsignalRegionQmax[index];}
  Double_t GetNumberOfClusters(Int_t index) const { return fTPCsignalNRegion[index%3];}
  Double_t GetNumberOfCrossedRows(Int_t index) const { return fTPCsignalNRowRegion[index%3];}
  //
  Double_t GetTPCsignalShortPad()      const {return fTPCsignalRegion[0];}
  Double_t GetTPCsignalMediumPad()     const {return fTPCsignalRegion[1];}
  Double_t GetTPCsignalLongPad()       const {return fTPCsignalRegion[2];}
  Double_t GetTPCsignalOROC()          const {return fTPCsignalRegion[3];}
  
  Double_t GetTPCsignalShortPadQmax()  const {return fTPCsignalRegionQmax[0];}
  Double_t GetTPCsignalMediumPadQmax() const {return fTPCsignalRegionQmax[1];}
  Double_t GetTPCsignalLongPadQmax()   const {return fTPCsignalRegionQmax[2];}
  Double_t GetTPCsignalOROCQmax()      const {return fTPCsignalRegionQmax[3];}
  static void     RegisterSectorCalibration(TGraphErrors* gainSector, Int_t regionID, Int_t calibID);
  Double_t  GetdEdxInfo(AliExternalTrackParam *param, Int_t regionID, Int_t calibID, Int_t qID, Int_t valueID);
private: 

  Double32_t  fTPCsignalRegion[4]; //[0.,0.,10] TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used)  - for qTot
  Double32_t  fTPCsignalRegionQmax[4]; //[0.,0.,10] TPC dEdx signal in 4 different regions - 0 - IROC, 1- OROC medium, 2 - OROC long, 3- OROC all, (default truncation used) - for qMax
  Char_t      fTPCsignalNRegion[3]; // number of clusters above threshold used in the dEdx calculation
  Char_t      fTPCsignalNRowRegion[3]; // number of crosed rows used in the dEdx calculation - signal below threshold included
  //
  static TObjArray *fArraySectorCalibration;
  
  ClassDef(AliTPCdEdxInfo,3)
};

#endif

#ifndef ALIVCLUSTER_H
#define ALIVCLUSTER_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//-------------------------------------------------------------------------
/// \class AliVCluster
/// \brief Virtual class for calorimeter cluster data handling
///
///   Virtual class to access calorimeter 
///   (EMCAL, PHOS, PMD, FMD) cluster data
///
///  \author Gustavo Conesa Balbastre, <Gustavo.Conesa.Balbastre@cern.ch>, LPSC-Grenoble
//
//-------------------------------------------------------------------------

#include <TObject.h>
#include <TLorentzVector.h>

class AliVCluster : public TObject 
{
  
 public:
  
  AliVCluster() { ; }
  virtual ~AliVCluster() { ; }
  AliVCluster(const AliVCluster& clus);
  AliVCluster & operator=(const AliVCluster& source);
  void Clear(const Option_t*) {;}
  
  /// Define the type of clusters for the different calorimeters
  enum VClu_t {
    kUndef = -2, 
    kPHOSNeutral, 
    kPHOSCharged,
    kEMCALClusterv1,		 
    kPMDNeutral, 
    kPMDCharged};
  
  /// Define the PID types
  enum VCluPID_t {
    kElectron = 0,
    kMuon     = 1,
    kPion     = 2,
    kKaon     = 3,
    kProton   = 4,
    kPhoton   = 5,
    kPi0      = 6,
    kNeutron  = 7,
    kKaon0    = 8,
    kEleCon   = 9,
    kUnknown  = 10,
    kCharged  = 11,//For PMD?
    kNeutral  = 12 //For PMD? 
  };

  /// Define the correction types
  enum VCluUserDefEnergy_t {
    kNonLinCorr          = 0,
    kHadCorr             = 1,
    kUserDefEnergy1      = 2,
    kUserDefEnergy2      = 3,
    kLastUserDefEnergy   = 4
  };
  
  // Common EMCAL/PHOS/FMD/PMD
  
  virtual void        SetID(Int_t )                 { ; }
  virtual Int_t       GetID() const                 {return 0 ; }
  
  virtual void        SetType(Char_t )              { ; }  
  virtual Char_t      GetType() const               {return kUndef ; } 
  
  virtual void        SetE(Double_t )               { ; }
  virtual Double_t       E() const                  {return 0. ; }
  
  virtual void        SetChi2(Double_t )            { ; }
  virtual Double_t       Chi2() const               {return 0. ; }
  
  virtual void        SetPositionAt(Float_t,Int_t)  { ; }
  virtual void        SetPosition(Float_t *)        { ; }
  virtual void        GetPosition(Float_t *) const  { ; }	
  
  virtual void        SetPIDAt(Float_t , Int_t)     { ; }
  virtual void        SetPID(const Float_t *)       { ; }
  virtual const Double_t *GetPID() const            { return 0 ; }
  
  // CaloClusters, PHOS/EMCAL
  
  virtual Bool_t      IsEMCAL() const               {return kFALSE ; }
  virtual Bool_t      IsPHOS()  const               {return kFALSE ; }
  
  virtual void        SetDispersion(Double_t )      { ; }
  virtual Double_t    GetDispersion() const         {return 0. ;}
  
  virtual void        SetM20(Double_t)              { ; }
  virtual Double_t    GetM20() const                {return 0. ; }
  
  virtual void        SetM02(Double_t)              { ; }
  virtual Double_t    GetM02() const                {return 0. ; }
  
  virtual void        SetNExMax(UChar_t)            { ; }
  virtual UChar_t     GetNExMax() const             {return 0 ; } 
  
  virtual void        SetTOF(Double_t)              { ; }
  virtual Double_t    GetTOF() const                {return 0. ; }
  
  virtual void        SetEmcCpvDistance(Double_t)   { ; }
  virtual Double_t    GetEmcCpvDistance() const     {return 0. ; }
  virtual void        SetTrackDistance(Double_t, Double_t ){ ; }
  virtual Double_t    GetTrackDx(void)const         {return 0. ; }
  virtual Double_t    GetTrackDz(void)const         {return 0. ; }
  
  virtual void        SetDistanceToBadChannel(Double_t) { ; }
  virtual Double_t    GetDistanceToBadChannel() const   {return 0. ; }
  
  virtual void        SetNCells(Int_t)              { ; }
  virtual Int_t       GetNCells() const             {return 0 ; }
  virtual void        SetCellsAbsId(UShort_t */*array*/) {;}  
  virtual UShort_t   *GetCellsAbsId()               {return 0 ; }
  virtual void        SetCellsAmplitudeFraction(Double32_t */*array*/) {;}
  virtual Double_t   *GetCellsAmplitudeFraction()   {return 0 ; }
  virtual Int_t       GetCellAbsId(Int_t) const     {return 0 ; }  
  virtual Double_t    GetCellAmplitudeFraction(Int_t) const {return 0. ; }
  
  virtual Int_t       GetLabel() const              {return -1 ;}
  virtual Int_t       GetLabelAt(UInt_t) const      {return -1 ;}
  virtual Int_t      *GetLabels() const             {return 0 ; }
  virtual UInt_t      GetNLabels() const            {return 0 ; }
  virtual void        SetLabel(Int_t *, UInt_t )    { ; }

  virtual void        SetCellsMCEdepFractionMap(UInt_t * /*array*/) {;}  
  virtual UInt_t     *GetCellsMCEdepFractionMap()  const   {return 0 ; }
  virtual void        GetCellMCEdepFractionArray(Int_t, Float_t * /*eDepFraction[4]*/) const {;} 
  virtual UInt_t      PackMCEdepFraction(Float_t * /*eDepFraction[4]*/) const {return 0 ; } 
  
  virtual void        SetClusterMCEdepFractionFromEdepArray(Float_t  */*array*/) {;} 
  virtual void        SetClusterMCEdepFraction             (UShort_t */*array*/) {;}   
  virtual UShort_t   *GetClusterMCEdepFraction() const      {return 0 ; }
  virtual Float_t     GetClusterMCEdepFraction(Int_t) const {return 0 ; }
  
  virtual Int_t       GetNTracksMatched() const     {return 0 ; }
  
  /// Only for AODs
  virtual TObject    *GetTrackMatched(Int_t) const  {return 0 ; }
  
  /// Only for ESDs
  virtual Int_t       GetTrackMatchedIndex(Int_t=0) const  {return -1; }
  
  virtual Double_t    GetCoreEnergy() const         {return 0 ; }
  virtual void        SetCoreEnergy(Double_t)       { ; }

  virtual Double_t    GetMCEnergyFraction() const   { return 0 ; }
  virtual void        SetMCEnergyFraction(Double_t) { ; }

  virtual Bool_t      GetIsExotic() const           { return kFALSE; }
  virtual void        SetIsExotic(Bool_t /*b*/)     { ; }
  
  virtual Double_t    GetUserDefEnergy(VCluUserDefEnergy_t) const         { return 0.  ; }
  virtual void        SetUserDefEnergy(VCluUserDefEnergy_t, Double_t)     {  ; }

  virtual Double_t    GetUserDefEnergy(Int_t i) const           { return i >= 0 && i <= kLastUserDefEnergy ? GetUserDefEnergy((VCluUserDefEnergy_t)i) : 0.  ; }
  virtual void        SetUserDefEnergy(Int_t i, Double_t v)     { if (i >= 0 && i <= kLastUserDefEnergy) SetUserDefEnergy((VCluUserDefEnergy_t)i, v)        ; }

  Double_t            GetNonLinCorrEnergy() const      { return GetUserDefEnergy(kNonLinCorr)  ; }
  void                SetNonLinCorrEnergy(Double_t e)  { SetUserDefEnergy(kNonLinCorr, e)      ; }

  Double_t            GetHadCorrEnergy() const      { return GetUserDefEnergy(kHadCorr)     ; }
  void                SetHadCorrEnergy(Double_t e)  { SetUserDefEnergy(kHadCorr, e)         ; }
  
  virtual void GetMomentum(TLorentzVector &/*tl*/, const Double_t * /*v*/) const { ; }
  virtual void GetMomentum(TLorentzVector &/*tl*/, const Double_t * /*v*/, VCluUserDefEnergy_t /*t*/) const { ; }
  
  /// \cond CLASSIMP
  ClassDef(AliVCluster,0) ; //VCluster 
  /// \endcond

};

#endif //ALIVCLUSTER_H


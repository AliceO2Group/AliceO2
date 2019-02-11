#ifndef ALIESDCALOCLUSTER_H
#define ALIESDCALOCLUSTER_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//-------------------------------------------------------------------------
/// \class AliESDCaloCluster
/// \brief Calorimeter cluster data container
///
/// This is the class to deal with during the physics analysis of calorimeters 
/// data. Container for calorimeter clusters, which are the effective 
/// "tracks" for calorimeter detectors.  Can be used by PHOS and EMCAL
///
///  \author J.L. Klay, <Gustavo.Conesa.Balbastre@cern.ch>, LLNL
///  \author Gustavo Conesa Balbastre, <Gustavo.Conesa.Balbastre@cern.ch>, LPSC-Grenoble
///
//-------------------------------------------------------------------------

#include <AliVCluster.h>
#include "AliPID.h"
#include "TArrayS.h"
#include "TArrayI.h"
#include "AliLog.h"

class TLorentzVector;

class AliESDCaloCluster : public AliVCluster 
{
  
 public:
  
  AliESDCaloCluster();
  AliESDCaloCluster(const AliESDCaloCluster& clus);
  AliESDCaloCluster & operator=(const AliESDCaloCluster& source);
  virtual ~AliESDCaloCluster();
  virtual void Copy(TObject &) const;
  void Clear(const Option_t*);

  void  SetID(Int_t id) {fID = id;}
  Int_t GetID() const {return fID;}
  
  void   SetType(Char_t type) { fClusterType = type; }
  Char_t GetType() const {return fClusterType; }
  
  Bool_t IsEMCAL() const {if(fClusterType == kEMCALClusterv1) return kTRUE; else return kFALSE;}
  Bool_t IsPHOS()  const {if(fClusterType == kPHOSNeutral || fClusterType == kPHOSCharged) return kTRUE;
    else return kFALSE;}
  
  void GetPosition  (Float_t *x) const {
    x[0]=fGlobalPos[0]; x[1]=fGlobalPos[1]; x[2]=fGlobalPos[2];}
  void SetPosition  (Float_t *x);
  void SetPositionAt(Float_t pos, Int_t ipos) {if(ipos>=0 && ipos<3) fGlobalPos[ipos] = pos ; 
    else AliInfo(Form("Bad index for position array, i = %d\n",ipos));}
  
  void  SetE(Double_t ene) { fEnergy = ene;}
  Double_t E() const       { return fEnergy;}
  
  void     SetDispersion(Double_t disp)  { fDispersion = disp; }
  Double_t GetDispersion() const         { return fDispersion; }
  
  void  SetChi2(Double_t chi2)  { fChi2 = chi2; }
  Double_t Chi2() const         { return fChi2; }
  
  const Double_t *GetPID() const { return fPID; }
  //for(Int_t i=0; i<AliPID::kSPECIESCN; ++i) pid[i]=fPID[i];}
  void SetPID  (const Float_t *pid) ;
  void SetPIDAt(Float_t p, Int_t i) {if(i>=0 && i<AliPID::kSPECIESCN) fPID[i] = p ; 
    else AliInfo(Form("Bad index for PID array, i = %d \n",i));}
  
  void     SetM20(Double_t m20) { fM20 = m20; }
  Double_t GetM20() const       { return fM20; }
  
  void     SetM02(Double_t m02) { fM02 = m02; }
  Double_t GetM02() const       { return fM02; }
  
  void    SetNExMax(UChar_t nExMax) { fNExMax = nExMax; }
  UChar_t GetNExMax() const         { return fNExMax; }
  
  void SetEmcCpvDistance(Double_t dEmcCpv) { fEmcCpvDistance = dEmcCpv; }
  Double_t GetEmcCpvDistance() const       { return fEmcCpvDistance; }
  void SetTrackDistance(Double_t dx, Double_t dz){fTrackDx=dx; fTrackDz=dz;}
  Double_t GetTrackDx(void)const {return fTrackDx;}
  Double_t GetTrackDz(void)const {return fTrackDz;}
  
  void     SetDistanceToBadChannel(Double_t dist) {fDistToBadChannel=dist;}
  Double_t GetDistanceToBadChannel() const        {return fDistToBadChannel;}
  
  void     SetTOF(Double_t tof) { fTOF = tof; }
  Double_t GetTOF() const       { return fTOF; }
  
  void AddTracksMatched(TArrayI & array)  { 
    if(!fTracksMatched)fTracksMatched   = new TArrayI(array);
    else *fTracksMatched = array;
  }
  
  void AddLabels(TArrayI & array)         { 
    if(!fLabels)fLabels = new TArrayI(array) ;
    else *fLabels = array;
    fNLabel = fLabels->GetSize();
  }
  
  void SetLabel(Int_t *array, UInt_t size){
    if(fLabels) delete fLabels ;
    fLabels = new TArrayI(size,array);
    fNLabel = size;
  }

  TArrayI * GetTracksMatched() const  {return  fTracksMatched;}
  TArrayI * GetLabelsArray() const    {return  fLabels;}
  Int_t   * GetLabels() const         {if (fLabels) return  fLabels->GetArray(); else return 0;}

  Int_t GetTrackMatchedIndex(Int_t i = 0) const;
  
  /// \return Label of MC particle that deposited more energy to the cluster
  Int_t GetLabel() const   {
    if( fLabels &&  fLabels->GetSize() >0)  return  fLabels->At(0); 
    else return -1;} 
  
  Int_t GetLabelAt(UInt_t i) const {
    if (fLabels && i < (UInt_t)fLabels->GetSize()) return fLabels->At(i);
    else return -999; }
  
  Int_t GetNTracksMatched() const { if (fTracksMatched) return  fTracksMatched->GetSize(); 
    else return -1;}
  
  UInt_t GetNLabels() const       { if (fLabels) return  fLabels->GetSize(); 
    else return (0);}
  
  void GetMomentum(TLorentzVector& p, const Double_t * vertexPosition ) const;
  void GetMomentum(TLorentzVector& p, const Double_t * vertexPosition, VCluUserDefEnergy_t t ) const;
  
  void  SetNCells(Int_t n)  { fNCells = n;}
  Int_t GetNCells() const   { return fNCells;}
  
  void      SetCellsAbsId(UShort_t *array) ;
  UShort_t *GetCellsAbsId() {return  fCellsAbsId;}
  
  void        SetCellsAmplitudeFraction(Double32_t *array) ;
  Double32_t *GetCellsAmplitudeFraction() {return  fCellsAmpFraction;}
  
  Int_t GetCellAbsId(Int_t i) const {  
    if (fCellsAbsId && i >=0 && i < fNCells ) return fCellsAbsId[i];    
    else return -1;}
  
  Double_t GetCellAmplitudeFraction(Int_t i) const {  
    if (fCellsAmpFraction && i >=0 && i < fNCells ) return fCellsAmpFraction[i];    
    else return -1;}

  Double_t    GetCoreEnergy() const                { return fCoreEnergy       ; }
  void        SetCoreEnergy(Double_t e)            { fCoreEnergy       = e    ; }

  Double_t    GetMCEnergyFraction() const          { return fMCEnergyFraction ; }
  void        SetMCEnergyFraction(Double_t e)      { fMCEnergyFraction = e    ; }
  
  Bool_t      GetIsExotic() const                  { return fIsExotic         ; }
  void        SetIsExotic(Bool_t b)                { fIsExotic         = b    ; }

  Double_t    GetUserDefEnergy(Int_t t)               const                     { return AliVCluster::GetUserDefEnergy(t); }
  Double_t    GetUserDefEnergy(VCluUserDefEnergy_t t) const                     { return E()*fUserDefEnergy[t]                   ; }
  void        SetUserDefEnergy(Int_t t, Double_t e)                             { AliVCluster::SetUserDefEnergy(t,e);}
  void        SetUserDefEnergy(VCluUserDefEnergy_t t, Double_t e)               { fUserDefEnergy[t] = E() > 1e-6 ? e / E() : 1.  ; }
  void        SetUserDefEnergyCorrFactor(VCluUserDefEnergy_t t, Double_t f)     { fUserDefEnergy[t] = f                          ; }
  
  void        SetCellsMCEdepFractionMap(UInt_t *array) ;
  UInt_t    * GetCellsMCEdepFractionMap() const    { return fCellsMCEdepFractionMap ; }

  void        GetCellMCEdepFractionArray(Int_t cellIndex, Float_t * eDep) const ;
  UInt_t      PackMCEdepFraction(Float_t * eDep) const ; 
  
  void        SetClusterMCEdepFractionFromEdepArray(Float_t  *array) ;
  void        SetClusterMCEdepFraction             (UShort_t *array) ;
  UShort_t  * GetClusterMCEdepFraction() const     { return fClusterMCEdepFraction     ; }
  Float_t     GetClusterMCEdepFraction(Int_t mcIndex) const ;

 protected:
  
  TArrayI    * fTracksMatched;    ///< Index of tracks close to cluster. First entry is the most likely match.
  TArrayI    * fLabels;           ///< List of MC particles that generated the cluster, ordered in deposited energy.
  
  Int_t        fNCells ;          ///< Number of cells in cluster.
  
  /// Array of cell absolute Id numbers.
  UShort_t   * fCellsAbsId;       //[fNCells] 
  
  /// Array with cell amplitudes fraction. Only usable for unfolded clusters, where cell can be shared.
  /// here we store what fraction of the cell energy is assigned to a given cluster.
  Double32_t * fCellsAmpFraction; //[fNCells][0.,1.,16] 
  
  Double32_t   fGlobalPos[3];     ///< Position in global coordinate system (cm).
  Double32_t   fEnergy;           ///< Energy measured by calorimeter in GeV.
  Double32_t   fDispersion;       ///< Cluster shape dispersion.
  Double32_t   fChi2;             ///< Chi2 of cluster fit (unfolded clusters)
  Double32_t   fM20;              ///< 2-nd moment along the second eigen axis.
  Double32_t   fM02;              ///< 2-nd moment along the main eigen axis.
  
  Double32_t   fEmcCpvDistance;   ///< the distance from PHOS EMC rec.point to the closest CPV rec.point.
  
  Double32_t   fTrackDx ;         ///< Distance to closest track in phi.
  Double32_t   fTrackDz ;         ///< Distance to closest track in z.
  
  Double32_t   fDistToBadChannel; ///< Distance to nearest bad channel.
  
  /// Detector response  probabilities for the PID
  Double32_t   fPID[AliPID::kSPECIESCN]; //[0,1,8]
  
  Int_t        fID;               ///< Unique Id of the cluster.
  UChar_t      fNExMax ;          ///< Number of Local (Ex-)maxima before unfolding.  
  Char_t       fClusterType;      ///< Flag for different cluster type/versions. See enum VClu_t in AliVCluster
  
  /// Cluster time-of-flight
  Double_t     fTOF;              //[0,0,12] 
  
  Double32_t   fCoreEnergy;       ///< Energy of the core of cluster. Used by PHOS.
  
  Double_t     fMCEnergyFraction; //!<! MC energy (embedding)
  Bool_t       fIsExotic;         //!<! Cluster marked as "exotic" (high energy deposition concentrated in a single cell)
  Double_t     fUserDefEnergy[kLastUserDefEnergy+1]; //!<!energy of the cluster after other higher level corrections (e.g. non-linearity, hadronic correction, ...)

  UInt_t       fNLabel;           ///< Number of MC particles associated to the cluster
  
  /// Array with fraction of deposited energy per MC particle contributing to the cluster
  UShort_t   * fClusterMCEdepFraction;  //[fNLabel] 
  
  /// Array of maps (4 bits, each bit a different information) with fraction of deposited energy 
  /// per MC particle contributing to the cluster (4 particles maximum) per cell in the cluster
  UInt_t     * fCellsMCEdepFractionMap; //[fNCells] 

  /// \cond CLASSIMP
  ClassDef(AliESDCaloCluster,13) ;  
  /// \endcond

};

#endif 



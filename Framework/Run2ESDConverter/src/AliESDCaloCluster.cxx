/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

#include <TLorentzVector.h>
#include "AliLog.h"
#include "AliESDCaloCluster.h"

/// \cond CLASSIMP
ClassImp(AliESDCaloCluster) ;
/// \endcond

///
/// The default ESD constructor 
///
//_______________________________________________________________________
AliESDCaloCluster::AliESDCaloCluster() : 
  AliVCluster(),
  fTracksMatched(0x0),
  fLabels(0x0),
  fNCells(0),
  fCellsAbsId(0x0),
  fCellsAmpFraction(0x0),
  fEnergy(0),
  fDispersion(0),
  fChi2(0),
  fM20(0),
  fM02(0),
  fEmcCpvDistance(1024),
  fTrackDx(1024),fTrackDz(1024),
  fDistToBadChannel(1024),
  fID(0),
  fNExMax(0),
  fClusterType(kUndef), 
  fTOF(0.),
  fCoreEnergy(0.),
  fMCEnergyFraction(0.),
  fIsExotic(kFALSE),
  fNLabel(0),
  fClusterMCEdepFraction(0x0),
  fCellsMCEdepFractionMap(0x0)
{
  fGlobalPos[0] = fGlobalPos[1] = fGlobalPos[2] = 0.;
  for(Int_t i=0; i<AliPID::kSPECIESCN; i++) fPID[i] = 0.;

  for (Int_t i = 0; i <= kLastUserDefEnergy; i++) {
    fUserDefEnergy[i] = 1.;
  }
}

///
/// The copy constructor 
///
//_______________________________________________________________________
AliESDCaloCluster::AliESDCaloCluster(const AliESDCaloCluster& clus) : 
  AliVCluster(clus),
  fTracksMatched(clus.fTracksMatched?new TArrayI(*clus.fTracksMatched):0x0),
  fLabels(clus.fLabels?new TArrayI(*clus.fLabels):0x0),
  fNCells(clus.fNCells),
  fCellsAbsId(),
  fCellsAmpFraction(),
  fEnergy(clus.fEnergy),
  fDispersion(clus.fDispersion),
  fChi2(clus.fChi2),
  fM20(clus.fM20),
  fM02(clus.fM02),
  fEmcCpvDistance(clus.fEmcCpvDistance),
  fTrackDx(clus.fTrackDx),
  fTrackDz(clus.fTrackDz),
  fDistToBadChannel(clus.fDistToBadChannel),
  fID(clus.fID),
  fNExMax(clus.fNExMax),
  fClusterType(clus.fClusterType),
  fTOF(clus.fTOF),
  fCoreEnergy(clus.fCoreEnergy),
  fMCEnergyFraction(clus.fMCEnergyFraction),
  fIsExotic(clus.fIsExotic),
  fNLabel(clus.fNLabel),
  fClusterMCEdepFraction(),
  fCellsMCEdepFractionMap()
{
  fGlobalPos[0] = clus.fGlobalPos[0];
  fGlobalPos[1] = clus.fGlobalPos[1];
  fGlobalPos[2] = clus.fGlobalPos[2];

  for(Int_t i=0; i<AliPID::kSPECIESCN; i++) fPID[i] = clus.fPID[i];

  if (clus.fNCells > 0) 
  {
    if(clus.fCellsAbsId)
    {
      fCellsAbsId = new UShort_t[clus.fNCells];
      for (Int_t i=0; i<clus.fNCells; i++) fCellsAbsId[i]=clus.fCellsAbsId[i];
    }
    
    if(clus.fCellsAmpFraction)
    {
      fCellsAmpFraction = new Double32_t[clus.fNCells];
      for (Int_t i=0; i<clus.fNCells; i++) fCellsAmpFraction[i]=clus.fCellsAmpFraction[i];
    }

    if(clus.fCellsMCEdepFractionMap)
    {
      fCellsMCEdepFractionMap = new UInt_t[clus.fNCells];
      for (Int_t i=0; i<clus.fNCells; i++) fCellsMCEdepFractionMap[i]=clus.fCellsMCEdepFractionMap[i];
    }
  }
  
  if(clus.fClusterMCEdepFraction && clus.fNLabel > 0)
  {
    fClusterMCEdepFraction = new UShort_t[clus.fNLabel];
    for (UInt_t i=0; i<clus.fNLabel; i++) fClusterMCEdepFraction[i]=clus.fClusterMCEdepFraction[i];
  }

  for (Int_t i = 0; i <= kLastUserDefEnergy; i++) 
    fUserDefEnergy[i] = clus.fUserDefEnergy[i];

}

///
/// The assignment operator.
///
//_______________________________________________________________________
AliESDCaloCluster &AliESDCaloCluster::operator=(const AliESDCaloCluster& source)
{
  if(&source == this) return *this;
  AliVCluster::operator=(source);
  fGlobalPos[0] = source.fGlobalPos[0];
  fGlobalPos[1] = source.fGlobalPos[1];
  fGlobalPos[2] = source.fGlobalPos[2];

  fEnergy = source.fEnergy;
  fDispersion = source.fDispersion;
  fChi2 = source.fChi2;
  fM20 = source.fM20;
  fM02 = source.fM02;
  fEmcCpvDistance = source.fEmcCpvDistance;
  fTrackDx= source.fTrackDx ;
  fTrackDz= source.fTrackDz ;
  fDistToBadChannel = source.fDistToBadChannel ;
  for(Int_t i=0; i<AliPID::kSPECIESCN; i++) fPID[i] = source.fPID[i];
  fID = source.fID;
  
  fNLabel = source.fNLabel;
  fNCells = source.fNCells;

  if (source.fNCells > 0) 
  {
    if(source.fCellsAbsId)
    {
      if(!fCellsAbsId)
      {
        if(fCellsAbsId)delete [] fCellsAbsId;
        fCellsAbsId = new UShort_t[source.fNCells];
      }
      
      for (Int_t i=0; i<source.fNCells; i++)
        fCellsAbsId[i]=source.fCellsAbsId[i];
    }
    
    if(source.fCellsAmpFraction)
    {
      if(!fCellsAmpFraction)
      {
        if(fCellsAmpFraction) delete [] fCellsAmpFraction;
        fCellsAmpFraction = new Double32_t[source.fNCells];
      }
      
      for (Int_t i=0; i<source.fNCells; i++)
        fCellsAmpFraction[i]=source.fCellsAmpFraction[i];
    }  
    
    if(source.fCellsMCEdepFractionMap)
    {
      if (!fCellsMCEdepFractionMap)
      {
        if(fCellsMCEdepFractionMap) delete [] fCellsMCEdepFractionMap;
        fCellsMCEdepFractionMap = new UInt_t[source.fNCells];
      }
      
      for (Int_t i=0; i<source.fNCells; i++) 
        fCellsMCEdepFractionMap[i]=source.fCellsMCEdepFractionMap[i];
    }
  } // fNCells > 0
  
  if(source.fClusterMCEdepFraction && source.fNLabel > 0)
  {
    if (!fClusterMCEdepFraction)
    {
      if(fClusterMCEdepFraction) delete [] fClusterMCEdepFraction;
      fClusterMCEdepFraction = new UShort_t[source.fNLabel];
    }
    
    for (UInt_t i=0; i<source.fNLabel; i++) 
      fClusterMCEdepFraction[i]=source.fClusterMCEdepFraction[i];
  }

  
  fNExMax = source.fNExMax;
  fClusterType = source.fClusterType;
  fTOF = source.fTOF;

  //not in use
  if(source.fTracksMatched){
    // assign or copy construct
    if(fTracksMatched){
      *fTracksMatched = *source.fTracksMatched;
    }
    else fTracksMatched = new TArrayI(*source.fTracksMatched);
  }
  else{
    if(fTracksMatched)delete fTracksMatched;
    fTracksMatched = 0;
  }

  if(source.fLabels){
    // assign or copy construct
    if(fLabels){ 
      *fLabels = *source.fLabels;
    }
    else fLabels = new TArrayI(*source.fLabels);
  }
  else{
    if(fLabels)delete fLabels;
    fLabels = 0;
  }

  fCoreEnergy = source.fCoreEnergy;
  
  fMCEnergyFraction = source.fMCEnergyFraction;
  fIsExotic = source.fIsExotic;

  for (Int_t i = 0; i <= kLastUserDefEnergy; i++) {
    fUserDefEnergy[i] = source.fUserDefEnergy[i];
  }
  
  return *this;

}

///
/// This method overwrites the virtual TObject::Copy()
/// to allow run time copying without casting
/// in AliESDEvent
//_______________________________________________________________________
void AliESDCaloCluster::Copy(TObject &obj) const 
{
  if(this==&obj)return;
 
  AliESDCaloCluster *robj = dynamic_cast<AliESDCaloCluster*>(&obj);
  
  if(!robj)return; // not an AliESDCluster
  
  *robj = *this;
}

///
/// This is destructor according Coding Conventions. 
///
//_______________________________________________________________________
AliESDCaloCluster::~AliESDCaloCluster()
{ 
  if(fTracksMatched) delete fTracksMatched;
  fTracksMatched = 0;
  if(fLabels)        delete fLabels;
  fLabels        = 0;
  
  if(fCellsAmpFraction)       { delete[] fCellsAmpFraction;       fCellsAmpFraction       = 0 ; }
  if(fCellsAbsId)             { delete[] fCellsAbsId;             fCellsAbsId             = 0 ; }
  if(fClusterMCEdepFraction)  { delete[] fClusterMCEdepFraction;  fClusterMCEdepFraction  = 0 ; }
  if(fCellsMCEdepFractionMap) { delete[] fCellsMCEdepFractionMap; fCellsMCEdepFractionMap = 0 ; }
}

//
// Delete pointers 
//
//_______________________________________________________________________
void AliESDCaloCluster::Clear(const Option_t*)
{ 
  if(fTracksMatched) delete fTracksMatched;
  fTracksMatched = 0;
  if(fLabels)        delete fLabels;
  fLabels        = 0;
  
  if(fCellsAmpFraction)       { delete[] fCellsAmpFraction;       fCellsAmpFraction       = 0 ; }
  if(fCellsAbsId)             { delete[] fCellsAbsId;             fCellsAbsId             = 0 ; }
  if(fClusterMCEdepFraction)  { delete[] fClusterMCEdepFraction;  fClusterMCEdepFraction  = 0 ; }
  if(fCellsMCEdepFractionMap) { delete[] fCellsMCEdepFractionMap; fCellsMCEdepFractionMap = 0 ; }
}

///
/// Sets the probability of each particle type
/// Copied from AliESDtrack SetPIDValues
/// This function copies "n" PID weights from "scr" to "dest"
/// and normalizes their sum to 1 thus producing conditional
/// probabilities.
/// The negative weights are set to 0.
/// In case all the weights are non-positive they are replaced by
/// uniform probabilities
//_______________________________________________________________________
void AliESDCaloCluster::SetPID(const Float_t *p) 
{
  Int_t n = AliPID::kSPECIESCN;

  Float_t uniform = 1./(Float_t)n;

  Float_t sum = 0;
  for (Int_t i=0; i<n; i++)
    if (p[i]>=0) {
      sum+=p[i];
      fPID[i] = p[i];
    }
    else {
      fPID[i] = 0;
    }

  if(sum>0)
    for (Int_t i=0; i<n; i++) fPID[i] /= sum;
  else
    for (Int_t i=0; i<n; i++) fPID[i] = uniform;

}

///
/// Returns TLorentzVector with momentum of the cluster. Only valid for clusters 
/// identified as photons or pi0 (overlapped gamma) produced on the vertex
/// Vertex can be recovered with esd pointer doing:  
///    " Double_t vertex[3] ; esd->GetVertex()->GetXYZ(vertex) ; "
///
//_______________________________________________________________________
void AliESDCaloCluster::GetMomentum(TLorentzVector& p, const Double_t *vertex ) const
{
  Double32_t pos[3]={ fGlobalPos[0], fGlobalPos[1], fGlobalPos[2]};
  if(vertex){//calculate direction from vertex
    pos[0]-=vertex[0];
    pos[1]-=vertex[1];
    pos[2]-=vertex[2];
  }
  
  Double_t r = TMath::Sqrt(pos[0]*pos[0]+pos[1]*pos[1]+pos[2]*pos[2]   ) ; 
  
  if ( r > 0 ) 
    p.SetPxPyPzE( fEnergy*pos[0]/r, fEnergy*pos[1]/r, fEnergy*pos[2]/r, fEnergy) ; 
  else
    AliInfo("Null cluster radius, momentum calculation not possible");
}

///
/// Returns TLorentzVector with momentum of the cluster. Only valid for clusters 
/// identified as photons or pi0 (overlapped gamma) produced on the vertex
/// Uses the user defined energy t
/// Vertex can be recovered with esd pointer doing:  
///   " Double_t vertex[3] ; esd->GetVertex()->GetXYZ(vertex) ; "
///
//_______________________________________________________________________
void AliESDCaloCluster::GetMomentum(TLorentzVector& p, const Double_t *vertex, VCluUserDefEnergy_t t ) const
{
  Double32_t energy = GetUserDefEnergy(t);
  Float_t    pos[3];
  GetPosition(pos);
  
  if(vertex){//calculate direction from vertex
    pos[0]-=vertex[0];
    pos[1]-=vertex[1];
    pos[2]-=vertex[2];
  }
  
  Double_t r = TMath::Sqrt(pos[0]*pos[0]+
			   pos[1]*pos[1]+
			   pos[2]*pos[2]   ) ; 
  
  p.SetPxPyPzE( energy*pos[0]/r,  energy*pos[1]/r,  energy*pos[2]/r,  energy) ; 
  
}

///
///  Set the array of cell absId numbers. 
///
//_______________________________________________________________________
void  AliESDCaloCluster::SetCellsAbsId(UShort_t *array)
{
  if (fNCells) {
    fCellsAbsId = new  UShort_t[fNCells];
    for (Int_t i = 0; i < fNCells; i++) fCellsAbsId[i] = array[i];
  }
}

///
///  Set the array of cell amplitude fractions. 
///  Cell can be shared between 2 clusters, here the fraction of energy
///  assigned to each cluster is stored. Only in unfolded clusters.
///
//_______________________________________________________________________
void  AliESDCaloCluster::SetCellsAmplitudeFraction(Double32_t *array)
{
  if (fNCells) {
    fCellsAmpFraction = new  Double32_t[fNCells];
    for (Int_t i = 0; i < fNCells; i++) fCellsAmpFraction[i] = array[i];
  }
}

///
/// Set the cluster global position.
///
//______________________________________________________________________________
void AliESDCaloCluster::SetPosition(Float_t *x) 
{  
  if (x) {
    fGlobalPos[0] = x[0];
    fGlobalPos[1] = x[1];
    fGlobalPos[2] = x[2];
  } else {
    
    fGlobalPos[0] = -999.;
    fGlobalPos[1] = -999.;
    fGlobalPos[2] = -999.;
  }
}

///
/// \return Index of track matched to cluster. Several matches are possible.
///
/// \param i: matched track index in array of matches
///
//______________________________________________________________________________
Int_t AliESDCaloCluster::GetTrackMatchedIndex(Int_t i) const
{
  if (fTracksMatched && i >= 0 && i < fTracksMatched->GetSize())  {
    return fTracksMatched->At(i);
  }
  else {
    return -1;
  }
}

///
/// \param cellIndex: position of cell in array fCellsAbsId
/// \param eDep: Filled float array with 4 entries, each is the fraction of deposited 
///              energy by 4 most significant MC particles (GetLabels()) in a cell of the cluster.
/// In this method, the 4 fractions  stored in % values (0 to 100) 
/// in each bit of the integer fCellsMCEdepFractionMap[cellIndex] are unpacked. 
//______________________________________________________________________________
void  AliESDCaloCluster::GetCellMCEdepFractionArray(Int_t cellIndex, Float_t * eDep) const
{ 
  if ( cellIndex >= fNCells || fNCells < 0 || !fCellsMCEdepFractionMap)
  {
    eDep[0] = eDep[1] = eDep[2] = eDep[3] = 0. ;
    return;
  }
  
  eDep[0] =  (fCellsMCEdepFractionMap[cellIndex]&0x000000ff)        / 100.;
  eDep[1] = ((fCellsMCEdepFractionMap[cellIndex]&0x0000ff00) >>  8) / 100.;
  eDep[2] = ((fCellsMCEdepFractionMap[cellIndex]&0x00ff0000) >> 16) / 100.;
  eDep[3] = ((fCellsMCEdepFractionMap[cellIndex]&0xff000000) >> 24) / 100.;  
}

///
/// \param eDep: Float array with 4 entries, each is the fraction of deposited 
///              energy by an MC particle in a cell of the cluster.
/// 
/// The MC particle must correspond one of the 4 first labels in GetLabels(). This method
/// packs the 4 floats into an integer, assigning each bit a value between 0 and 100
//______________________________________________________________________________
UInt_t  AliESDCaloCluster::PackMCEdepFraction(Float_t * eDep) const
{ 
  UInt_t intEDep[4];
  
  for(Int_t i = 0; i < 4; i++)
    intEDep[i] = TMath::Nint(eDep[i]*100) ;
  
  UInt_t map = intEDep[0]|(intEDep[1]<<8)|(intEDep[2]<<16)|(intEDep[3]<<24);

  return map;
}

///
/// \return Fraction of deposited energy by one of the particles in array fLable
/// 
/// \param mcIndex: position of MC particle in array fLabel
///
/// The parameter is stored as %, return the corresponding float.
//______________________________________________________________________________
Float_t  AliESDCaloCluster::GetClusterMCEdepFraction(Int_t mcIndex) const
{ 
  if ( mcIndex < 0 ||  mcIndex >= (Int_t) GetNLabels() || !fClusterMCEdepFraction) return 0. ;

  return  fClusterMCEdepFraction[mcIndex]/100. ; 
}

///
/// Set the array with the fraction of deposited energy in a cell belonging to 
/// the cluster by a given primary  particle. Each entry of the array corresponds 
/// to the same entry in fCellsAbsId. Each entry is an integer where a maximum 
/// of 4 energy deposition fractions are encoded, each corresponding to the 
/// first 4 entries in fLabels
//______________________________________________________________________________
void  AliESDCaloCluster::SetCellsMCEdepFractionMap(UInt_t *array)
{
  if ( fNCells <= 0 || !array) return; 
  
  fCellsMCEdepFractionMap = new  UInt_t[fNCells];
  
  for (Int_t i = 0; i < fNCells; i++) 
    fCellsMCEdepFractionMap[i] = array[i];
}

///
/// Set the array with the fraction of deposited energy in cluster by a given primary 
/// particle. Each entry of the array corresponds to the same entry in GetLabels().
/// Set the fraction in % with respect the cluster energy, store a value between 0 and 100
///
/// \param array: energy deposition array
//______________________________________________________________________________
void  AliESDCaloCluster::SetClusterMCEdepFractionFromEdepArray(Float_t *array)
{
  if ( fLabels->GetSize() <= 0 || !array) return ; 

  fClusterMCEdepFraction = new  UShort_t[fLabels->GetSize()];
  
  // Get total deposited energy (can be different from reconstructed energy)
  Float_t totalE = 0;
  for (Int_t i = 0; i < fLabels->GetSize(); i++) totalE+=array[i];

  // Set the fraction of energy per MC contributor in %
  for (Int_t i = 0; i < fLabels->GetSize(); i++)     
    fClusterMCEdepFraction[i] = TMath::Nint(array[i]/totalE*100.);
}


///
/// Set the array with the fraction of deposited energy in cluster by a given primary 
/// particle. Each entry of the array corresponds to the same entry in GetLabels().
///
/// The fraction must already be in % with respect the cluster energy, store a value between 0 and 100
/// \param array: array of fraction of energy deposition / cluster energy 
//______________________________________________________________________________
void  AliESDCaloCluster::SetClusterMCEdepFraction(UShort_t *array)
{
  if ( fLabels->GetSize() <= 0 || !array ) return ; 
  
  fClusterMCEdepFraction = new  UShort_t[fLabels->GetSize()];
  
  for (Int_t i = 0; i < fLabels->GetSize(); i++) 
    fClusterMCEdepFraction[i] = array[i];
}

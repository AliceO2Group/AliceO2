#ifndef ALIESDMUONCLUSTER_H
#define ALIESDMUONCLUSTER_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
* See cxx source for full Copyright notice                               */

// $Id$

/// \class AliESDMuonCluster
/// \brief Class to describe the MUON clusters in the Event Summary Data
//  Author Philippe Pillot, Subatech


#include <TObject.h>
#include <TArrayI.h>

class AliESDEvent;
class TClonesArray;

class AliESDMuonCluster : public TObject {
public:
  AliESDMuonCluster(); // Constructor
  virtual ~AliESDMuonCluster(); //< Destructor
  AliESDMuonCluster(const AliESDMuonCluster& cluster);
  AliESDMuonCluster& operator=(const AliESDMuonCluster& cluster);
  virtual void Copy(TObject &obj) const;
  
  virtual void Clear(Option_t* opt = "");
  
  /// Set coordinates (cm)
  void     SetXYZ(Double_t x, Double_t y, Double_t z) {fXYZ[0] = x; fXYZ[1] = y; fXYZ[2] = z;}
  /// Return X-position (cm)
  Double_t GetX() const {return fXYZ[0];}
  /// Return Y-position (cm)
  Double_t GetY() const {return fXYZ[1];}
  /// Return Z-position (cm)
  Double_t GetZ() const {return fXYZ[2];}
  
  /// Set (X,Y) resolution (cm)
  void     SetErrXY(Double_t errX, Double_t errY) {fErrXY[0] = errX; fErrXY[1] = errY;}
  /// Return X-resolution (cm)
  Double_t GetErrX() const  {return fErrXY[0];}
  /// Return X-resolution**2 (cm**2)
  Double_t GetErrX2() const {return fErrXY[0]*fErrXY[0];}
  /// Return Y-resolution (cm)
  Double_t GetErrY() const  {return fErrXY[1];}
  /// Return Y-resolution**2 (cm**2)
  Double_t GetErrY2() const {return fErrXY[1]*fErrXY[1];}
  
  /// Set the total charge
  void     SetCharge(Double_t charge) {fCharge = charge;}
  /// Return the total charge
  Double_t GetCharge() const {return fCharge;}
  
  /// Set the chi2 value
  void     SetChi2(Double_t chi2) {fChi2 = chi2;}
  /// Return the chi2 value
  Double_t GetChi2() const {return fChi2;}
  
  /// Return chamber id (0..), part of the uniqueID
  Int_t    GetChamberId() const    {return (GetUniqueID() & 0xF0000000) >> 28;}
  /// Return detection element id, part of the uniqueID
  Int_t    GetDetElemId() const    {return (GetUniqueID() & 0x0FFE0000) >> 17;}
  /// Return the index of this cluster (0..), part of the uniqueID
  Int_t    GetClusterIndex() const {return (GetUniqueID() & 0x0001FFFF);}
  
  // Add the given pad Id to the list associated to the cluster
  void     AddPadId(UInt_t padId);
  // Fill the list pads'Id associated to the cluster with the given list
  void     SetPadsId(Int_t nPads, const UInt_t *padsId);
  /// Return the number of pads associated to this cluster
  Int_t    GetNPads() const {return fNPads;}
  /// Return the Id of pad i
  UInt_t   GetPadId(Int_t i) const {return (fPadsId && i >= 0 && i < fNPads) ? static_cast<UInt_t>(fPadsId->At(i)) : 0;}
  /// Return the array of pads'Id
  const UInt_t* GetPadsId() const {return fPadsId ? reinterpret_cast<UInt_t*>(fPadsId->GetArray()) : 0x0;}
  /// Return kTrue if the pads'Id are stored
  Bool_t   PadsStored() const {return (fNPads > 0);}
  
  // Transfer pads to the new ESD structure
  void     MovePadsToESD(AliESDEvent &esd);
  
  /// Set the corresponding MC track number
  void  SetLabel(Int_t label) {fLabel = label;}
  /// Return the corresponding MC track number
  Int_t GetLabel() const {return fLabel;}
  
  void     Print(Option_t */*option*/ = "") const;
  
  
protected:
  Double32_t fXYZ[3];   ///< cluster position
  Double32_t fErrXY[2]; ///< transverse position errors
  Double32_t fCharge;   ///< cluster charge
  Double32_t fChi2;     ///< cluster chi2
  
  mutable TClonesArray* fPads;  ///< Array of pads attached to the cluster -- deprecated
  
  Int_t    fNPads;  ///< number of pads attached to the cluster
  TArrayI* fPadsId; ///< array of Ids of pads attached to the cluster
  
  Int_t fLabel; ///< point to the corresponding MC track
  
  
  ClassDef(AliESDMuonCluster, 4) // MUON ESD cluster class
};

#endif

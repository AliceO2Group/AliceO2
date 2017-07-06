// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_GEOMETRY_H_
#define ALICEO2_EMCAL_GEOMETRY_H_

#include <TArrayD.h>
#include <TGeoMatrix.h>
#include <TList.h>
#include <TNamed.h>
#include <TParticle.h>
#include <TVector3.h>

#include "EMCALBase/Constants.h"
#include "EMCALBase/EMCGeometry.h"

namespace o2
{
namespace EMCAL
{
class ShishKebabTrd1Module;

class Geometry : public TNamed
{
 public:
  enum fEMCSMType {
    EMCAL_STANDARD = 0,
    EMCAL_HALF = 1,
    EMCAL_THIRD = 2,
    DCAL_STANDARD = 3,
    DCAL_EXT = 4
  }; // possible SM Type

  ///
  /// Default constructor.
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world
  Geometry() = default;

  ///
  /// Constructor for normal use.
  ///
  /// \param name: geometry name, EMCAL_COMPLETEV1, EMCAL_COMPLETE12SMV1, EMCAL_COMPLETE12SMV1_DCAL,
  /// EMCAL_COMPLETE12SMV1_DCAL_8SM, EMCAL_COMPLETE12SMV1_DCAL_DEV (see main class description for definition) \param
  /// title \param mcname: Geant3/4, Flukla, needed for settings of transport (check) \param mctitle: Geant4 physics
  /// list (check)
  ///
  Geometry(const Text_t* name, const Text_t* title = "", const Text_t* mcname = "", const Text_t* mctitle = "");

  ///
  /// Copy constructor.
  ///
  Geometry(const Geometry& geom);

  ///
  /// Destructor.
  ///
  ~Geometry() override;

  ///
  /// Assign operator.
  ///
  Geometry& operator=(const Geometry& rvalue);

  ///
  /// \return the pointer of the unique instance of the geometry
  ///
  /// It should have been set before.
  ///
  static Geometry* GetInstance();

  ///
  /// \return the pointer of the unique instance of the geometry
  ///
  /// \param name: geometry name, EMCAL_COMPLETEV1, EMCAL_COMPLETE12SMV1, EMCAL_COMPLETE12SMV1_DCAL,
  /// EMCAL_COMPLETE12SMV1_DCAL_8SM, EMCAL_COMPLETE12SMV1_DCAL_DEV (see main class description for definition) \param
  /// title \param mcname: Geant3/4, Fluka, needed for settings of transport (check) \param mctitle: Geant4 physics list
  /// (check)
  ///
  static Geometry* GetInstance(const Text_t* name, const Text_t* title = "", const Text_t* mcname = "TGeant3",
                               const Text_t* mctitle = "");

  ///
  /// Instanciate geometry depending on the run number. Mostly used in analysis and MC anchors.
  ///
  /// \return the pointer of the unique instance
  ///
  /// \param runNumber: as indicated
  /// \param geoName: geometry name, EMCAL_COMPLETEV1, etc. Not really needed to be specified.
  /// \param mcname: Geant3/4, Fluka, needed for settings of transport (check). Not really needed to be specified.
  /// \param mctitle:  Geant4 physics list (check). Not really needed to be specified.
  ///
  static Geometry* GetInstanceFromRunNumber(Int_t runNumber, TString geoName = "", const Text_t* mcname = "TGeant3",
                                            const Text_t* mctitle = "");

  //////////
  // General
  //
  static Bool_t IsInitialized() { return Geometry::sInit; }
  // static const Char_t* GetDefaultGeometryName() {return EMCGeometry::fgkDefaultGeometryName;}

  ///
  /// Generate the list of Trd1 modules
  /// which will make up the EMCAL geometry
  /// key: look to the AliEMCALShishKebabTrd1Module::
  ///
  void CreateListOfTrd1Modules();

  TList* GetShishKebabTrd1Modules() const { return mShishKebabTrd1Modules; }

  ///
  /// \return  the shishkebabmodule at a given eta index point.
  ///
  ShishKebabTrd1Module* GetShishKebabModule(Int_t neta) const;

  ///
  /// Given a TParticle, check if it falls in the EMCal/DCal geometry
  /// Call ImpactOnEmcal.
  ///
  /// \param particle: TParticle
  /// \return true in EMCal/DCa;
  ///
  virtual Bool_t Impact(const TParticle* particle) const;

  ///
  /// Calculates the impact coordinates on EMCAL (centre of a tower/not on EMCAL surface)
  /// of a neutral particle
  /// emitted in the vertex vtx[3] with direction theta and phi in the ALICE global coordinate system
  ///
  /// \param vtx: TVector3 with vertex?, input
  /// \param theta: theta location, input
  /// \param phi: azimuthal angle, input
  /// \param absId: absolute ID number
  /// \param vimpact: TVector3 of impact coordinates?
  ///
  void ImpactOnEmcal(TVector3 vtx, Double_t theta, Double_t phi, Int_t& absId, TVector3& vimpact) const;

  ///
  /// Checks whether point is inside the EMCal volume
  ///
  Bool_t IsInEMCAL(Double_t x, Double_t y, Double_t z) const;

  ///
  /// Checks whether point is inside the DCal volume
  ///
  Bool_t IsInDCAL(Double_t x, Double_t y, Double_t z) const;

  ///
  /// Checks whether point is inside the EMCal volume (included DCal), used in AliEMCALv*.cxx
  /// Code uses cylindrical approximation made of inner radius (for speed)
  ///
  /// Points behind EMCAl/DCal, i.e. R > outer radius, but eta, phi in acceptance
  /// are considered to inside
  ///
  /// \return calo type, 1 EMCal, 2 DCal
  ///
  Int_t IsInEMCALOrDCAL(Double_t x, Double_t y, Double_t z) const;

  //////////////////////////////////////
  // Return EMCAL geometrical parameters
  //

  EMCGeometry* GetEMCGeometry() const { return mEMCGeometry; }

  const Char_t* GetNameOfEMCALEnvelope() const { return mEMCGeometry->GetNameOfEMCALEnvelope(); }
  Float_t GetArm1PhiMin() const { return mEMCGeometry->GetArm1PhiMin(); }
  Float_t GetArm1PhiMax() const { return mEMCGeometry->GetArm1PhiMax(); }
  Float_t GetArm1EtaMin() const { return mEMCGeometry->GetArm1EtaMin(); }
  Float_t GetArm1EtaMax() const { return mEMCGeometry->GetArm1EtaMax(); }
  Float_t GetIPDistance() const { return mEMCGeometry->GetIPDistance(); }
  Float_t GetEnvelop(Int_t index) const { return mEMCGeometry->GetEnvelop(index); }
  Float_t GetShellThickness() const { return mEMCGeometry->GetShellThickness(); }
  Float_t GetZLength() const { return mEMCGeometry->GetZLength(); }
  Float_t GetDCALInnerEdge() const { return mEMCGeometry->GetDCALInnerEdge(); }
  Float_t GetDCALPhiMin() const { return mEMCGeometry->GetDCALPhiMin(); }
  Float_t GetDCALPhiMax() const { return mEMCGeometry->GetDCALPhiMax(); }
  Float_t GetEMCALPhiMax() const { return mEMCGeometry->GetEMCALPhiMax(); }
  Int_t GetNECLayers() const { return mEMCGeometry->GetNECLayers(); }
  Float_t GetDCALInnerExtandedEta() const { return mEMCGeometry->GetDCALInnerExtandedEta(); }
  Int_t GetNZ() const { return mEMCGeometry->GetNZ(); }
  Int_t GetNEta() const { return mEMCGeometry->GetNEta(); }
  Int_t GetNPhi() const { return mEMCGeometry->GetNPhi(); }
  Float_t GetECPbRadThick() const { return mEMCGeometry->GetECPbRadThick(); }
  Float_t GetECScintThick() const { return mEMCGeometry->GetECScintThick(); }
  Float_t GetSampling() const { return mEMCGeometry->GetSampling(); }
  Int_t GetNumberOfSuperModules() const { return mEMCGeometry->GetNumberOfSuperModules(); }
  Float_t GetPhiGapForSuperModules() const { return mEMCGeometry->GetPhiGapForSuperModules(); }
  Float_t GetPhiModuleSize() const { return mEMCGeometry->GetPhiModuleSize(); }
  Float_t GetEtaModuleSize() const { return mEMCGeometry->GetEtaModuleSize(); }
  Float_t GetFrontSteelStrip() const { return mEMCGeometry->GetFrontSteelStrip(); }
  Float_t GetLateralSteelStrip() const { return mEMCGeometry->GetLateralSteelStrip(); }
  Float_t GetPassiveScintThick() const { return mEMCGeometry->GetPassiveScintThick(); }
  Float_t GetPhiTileSize() const { return mEMCGeometry->GetPhiTileSize(); }
  Float_t GetEtaTileSize() const { return mEMCGeometry->GetEtaTileSize(); }
  Float_t GetPhiSuperModule() const { return mEMCGeometry->GetPhiSuperModule(); }
  Int_t GetNPhiSuperModule() const { return mEMCGeometry->GetNPhiSuperModule(); }
  Int_t GetNPHIdiv() const { return mEMCGeometry->GetNPHIdiv(); }
  Int_t GetNETAdiv() const { return mEMCGeometry->GetNETAdiv(); }
  Int_t GetNCells() const { return mEMCGeometry->GetNCells(); }
  Float_t GetLongModuleSize() const { return mEMCGeometry->GetLongModuleSize(); }
  Float_t GetTrd1Angle() const { return mEMCGeometry->GetTrd1Angle(); }
  Float_t Get2Trd1Dx2() const { return mEMCGeometry->Get2Trd1Dx2(); }
  Float_t GetTrd1AlFrontThick() const { return mEMCGeometry->GetTrd1AlFrontThick(); }
  Float_t GetTrd1BondPaperThick() const { return mEMCGeometry->GetTrd1BondPaperThick(); }
  // --
  Int_t GetNCellsInSupMod() const { return mEMCGeometry->GetNCellsInSupMod(); }
  Int_t GetNCellsInModule() const { return mEMCGeometry->GetNCellsInModule(); }
  Int_t GetKey110DEG() const { return mEMCGeometry->GetKey110DEG(); }
  Int_t GetnSupModInDCAL() const { return mEMCGeometry->GetnSupModInDCAL(); }
  Int_t GetILOSS() const { return mEMCGeometry->GetILOSS(); }
  Int_t GetIHADR() const { return mEMCGeometry->GetIHADR(); }
  // --
  Float_t GetDeltaEta() const { return mEMCGeometry->GetDeltaEta(); }
  Float_t GetDeltaPhi() const { return mEMCGeometry->GetDeltaPhi(); }
  Int_t GetNTowers() const { return mEMCGeometry->GetNTowers(); }
  //
  Double_t GetPhiCenterOfSM(Int_t nsupmod) const { return mEMCGeometry->GetPhiCenterOfSM(nsupmod); }
  Double_t GetPhiCenterOfSMSec(Int_t nsupmod) const { return mEMCGeometry->GetPhiCenterOfSMSec(nsupmod); }
  Float_t GetSuperModulesPar(Int_t ipar) const { return mEMCGeometry->GetSuperModulesPar(ipar); }
  //
  Int_t GetSMType(Int_t nSupMod) const
  {
    if (nSupMod > mEMCGeometry->GetNumberOfSuperModules())
      return -1;
    return mEMCGeometry->GetEMCSystem()[nSupMod];
  }

  ///
  /// Method to check if iSupMod is a valid DCal SM
  ///
  Bool_t IsDCALSM(Int_t nSupMod) const;

  ///
  /// Method to check if iSupMod is a valid DCal SM from 1/3rd
  ///
  Bool_t IsDCALExtSM(Int_t nSupMod) const;

  // Methods needed for SM in extension, where center of SM != center of the SM-section.
  // Used in AliEMCALv0 to calculate position.
  Bool_t GetPhiBoundariesOfSM(Int_t nSupMod, Double_t& phiMin, Double_t& phiMax) const
  {
    return mEMCGeometry->GetPhiBoundariesOfSM(nSupMod, phiMin, phiMax);
  }
  Bool_t GetPhiBoundariesOfSMGap(Int_t nPhiSec, Double_t& phiMin, Double_t& phiMax) const
  {
    return mEMCGeometry->GetPhiBoundariesOfSMGap(nPhiSec, phiMin, phiMax);
  }

  // Obsolete?
  Float_t GetSteelFrontThickness() const { return mEMCGeometry->GetSteelFrontThickness(); }

  ///////////////////////////////
  // Geometry data member setters
  //
  void SetNZ(Int_t nz) { mEMCGeometry->SetNZ(nz); }
  void SetNPhi(Int_t nphi) { mEMCGeometry->SetNPhi(nphi); }
  //
  void SetSampling(Float_t samp) { mEMCGeometry->SetSampling(samp); }

  //////////////////////////
  // Global geometry methods
  //

  ///
  /// Figure out the global coordinates from local coordinates on a supermodule.
  /// Use the supermodule alignment. Use double[3]
  ///
  /// \param loc: double[3] local coordinates, input
  /// \param glob: double[3] global coordinates, output
  /// \param iSM: super module number
  ///
  void GetGlobal(const Double_t* loc, Double_t* glob, int ind) const;

  ///
  /// Figure out the global coordinates from local coordinates on a supermodule.
  /// Use the supermodule alignment. Use TVector3.
  ///
  /// \param vloc: 3-vector local coordinates, input (remove & ?)
  /// \param vglob: 3-vector global coordinates, output
  /// \param iSM: super module number
  ///

  void GetGlobal(const TVector3& vloc, TVector3& vglob, int ind) const;

  ///
  /// Figure out the global coordinates of a cell.
  /// Use the supermodule alignment. Use double[3].
  ///
  /// \param absId: cell absolute id. number.
  /// \param glob: 3-double coordinates, output
  ///
  void GetGlobal(Int_t absId, Double_t glob[3]) const;

  ///
  /// Figure out the global coordinates of a cell.
  /// Use the supermodule alignment. Use TVector3.
  ///
  /// \param absId: cell absolute id. number.
  /// \param vglob: TVector3 coordinates, output
  ///
  void GetGlobal(Int_t absId, TVector3& vglob) const;

  ////////////////////////////////////////
  // May 31, 2006; ALICE numbering scheme:
  // see ALICE-INT-2003-038: ALICE Coordinate System and Software Numbering Convention
  // All indexes are stared from zero now.
  //
  // abs id <-> indexes; Shish-kebab case, only TRD1 now.
  // EMCAL -> Super Module -> module -> tower(or cell) - logic tree of EMCAL
  //
  //**  Usual name of variable - Dec 18,2006 **
  //  nSupMod - index of super module (SM)
  //  nModule - index of module in SM
  //  nIphi   - phi index of tower(cell) in module
  //  nIeta   - eta index of tower(cell) in module
  //
  //  Inside SM
  //  iphim   - phi index of module in SM
  //  ietam   - eta index of module in SM
  //
  //  iphi    - phi index of tower(cell) in SM
  //  ieta    - eta index of tower(cell) in SM
  //
  // for a given tower index absId returns eta and phi of gravity center of tower.

  ///
  /// Figure out the eta/phi coordinates of a cell.
  /// Call to GetGlobal().
  ///
  /// \param absId: cell absolute id. number.
  /// \param eta: pseudo-rapidity, double
  /// \param phi: azimuthal angle, double
  ///
  void EtaPhiFromIndex(Int_t absId, Double_t& eta, Double_t& phi) const;

  ///
  /// Figure out the eta/phi coordinates of a cell.
  /// Call to GetGlobal(). Discard? Keep upper one?
  ///
  /// \param absId: cell absolute id. number.
  /// \param eta: pseudo-rapidity, float
  /// \param phi: azimuthal angle, float
  ///
  void EtaPhiFromIndex(Int_t absId, Float_t& eta, Float_t& phi) const;

  ///
  /// Get cell absolute ID number from eta and phi location.
  ///
  /// \param eta: pseudorapidity location
  /// \param phi: azimutal location
  /// \param absId: cell absolute ID number
  ///
  /// \return true if cell connexion found
  ///
  Bool_t GetAbsCellIdFromEtaPhi(Double_t eta, Double_t phi, Int_t& absId) const;

  ///
  /// Given a global eta/phi point check if it belongs to a supermodule covered region.
  /// \return false if phi belongs a phi cracks between SM or far from SM
  ///
  /// \param eta: pseudorapidity location
  /// \param phi: azimutal location
  /// \param nSupMod: super module number, output
  ///
  Bool_t SuperModuleNumberFromEtaPhi(Double_t eta, Double_t phi, Int_t& nSupMod) const;

  ///
  /// Get cell absolute ID number from location module (2 times 2 cells) of a super module
  ///
  /// \param nSupMod: super module number
  /// \param nModule: module number
  /// \param nIphi: index of cell in module in phi direction 0 or 1
  /// \param nIeta: index of cell in module in eta direction 0 or 1
  ///
  /// \return cell absolute ID number
  ///
  Int_t GetAbsCellId(Int_t nSupMod, Int_t nModule, Int_t nIphi, Int_t nIeta) const;

  ///
  /// \return true if cell ID number exists
  ///
  /// \param absId: input absolute cell ID number to check
  ///
  Bool_t CheckAbsCellId(Int_t absId) const;

  ///
  /// Get cell SM, module numbers from absolute ID number
  ///
  /// \param absId: cell absolute id. number
  /// \param nSupMod: super module number
  /// \param nModule: module number
  /// \param nIphi: index of cell in module in phi direction 0 or 1
  /// \param nIeta: index of cell in module in eta direction 0 or 1
  ///
  /// \return true if absolute ID number exists
  ///
  Bool_t GetCellIndex(Int_t absId, Int_t& nSupMod, Int_t& nModule, Int_t& nIphi, Int_t& nIeta) const;

  ///
  /// Get eta-phi indexes of module in SM
  ///
  /// \param nSupMod: super module number, input
  /// \param nModule: module number, input
  /// \param iphim: index in phi direction of module, output
  /// \param ietam: index in eta direction of module, output
  ///
  void GetModulePhiEtaIndexInSModule(Int_t nSupMod, Int_t nModule, Int_t& iphim, Int_t& ietam) const;

  ///
  /// Get eta-phi indexes of cell in SM
  ///
  /// \param nSupMod: super module number, input
  /// \param nModule: module number, input
  /// \param nIphi: index in phi direction in module, input
  /// \param nIeta: index in phi direction in module, input
  /// \param iphi: index in phi direction in super module, output
  /// \param ieta: index in eta direction in super module, output
  ///
  void GetCellPhiEtaIndexInSModule(Int_t nSupMod, Int_t nModule, Int_t nIphi, Int_t nIeta, Int_t& iphi,
                                   Int_t& ieta) const;

  ///
  /// Get cell SM,  from absolute ID number
  ///
  /// \param absId: cell absolute id. number
  /// \return super module number
  ///
  Int_t GetSuperModuleNumber(Int_t absId) const;
  Int_t GetNumberOfModuleInPhiDirection(Int_t nSupMod) const
  {
    if (GetSMType(nSupMod) == EMCAL_HALF)
      return mNPhi / 2;
    else if (GetSMType(nSupMod) == EMCAL_THIRD)
      return mNPhi / 3;
    else if (GetSMType(nSupMod) == DCAL_EXT)
      return mNPhi / 3;
    else
      return mNPhi;
  }

  ///
  /// Transition from cell indexes (ieta,iphi) to module indexes (ietam, iphim, nModule)
  //
  /// \param nSupMod: super module number
  /// \param iphi: index of cell in phi direction inside super module
  /// \param ieta: index of cell in eta direction inside super module
  /// \param iphim: index of cell in module in phi direction 0 or 1
  /// \param ietam: index of cell in module in eta direction 0 or 1
  /// \param nModule: module number
  ///
  void GetModuleIndexesFromCellIndexesInSModule(Int_t nSupMod, Int_t iphi, Int_t ieta, Int_t& iphim, Int_t& ietam,
                                                Int_t& nModule) const;

  ///
  /// Transition from super module number (nSupMod) and cell indexes (ieta,iphi) to cell absolute ID number.
  ///
  /// \param nSupMod: super module number
  /// \param iphi: index of cell in phi direction inside super module
  /// \param ieta: index of cell in eta direction inside super module
  ///
  /// \return cell absolute ID number
  Int_t GetAbsCellIdFromCellIndexes(Int_t nSupMod, Int_t iphi, Int_t ieta) const;

  ///
  /// Online mapping and numbering is the same for EMCal and DCal SMs but:
  ///  - DCal odd SM (13,15,17) has online cols: 16-47; offline cols 0-31.
  ///  - Even DCal SMs have the same numbering online and offline 0-31.
  ///  - DCal 1/3 SM (18,19), online rows 16-23; offline rows 0-7
  ///
  /// Here shift the online cols or rows depending on the
  /// super-module number to match the offline mapping.
  ///
  /// \param sm: super module number of the channel/cell
  /// \param iphi: row/phi cell index, modified for DCal
  /// \param ieta: column/eta index, modified for DCal
  ///
  void ShiftOnlineToOfflineCellIndexes(Int_t sm, Int_t& iphi, Int_t& ieta) const;

  ///
  /// Here shift the DCal online cols or rows depending on the
  /// super-module number to match the online mapping.
  ///
  /// Reverse procedure to the one in the method above
  /// ShiftOnlineToOfflineCellIndexes().
  ///
  /// \param sm: super module number of the channel/cell
  /// \param iphi: row/phi cell index, modified for DCal
  /// \param ieta: column/eta index, modified for DCal
  ///
  void ShiftOfflineToOnlineCellIndexes(Int_t sm, Int_t& iphi, Int_t& ieta) const;

  ///
  /// Methods for AliEMCALRecPoint: Look to see what the relative position inside a given cell is for a recpoint.
  ///
  /// \param absId: cell absolute id. number, input
  /// \param xr,yr,zr - x,y,z coordinates of cell with absId inside SM, output
  ///
  /// \return false if cell absId does not exist
  Bool_t RelPosCellInSModule(Int_t absId, Double_t& xr, Double_t& yr, Double_t& zr) const;

  ///
  /// Methods for AliEMCALRecPoint: Look to see what the relative position inside a given cell is for a recpoint.
  /// Same as RelPosCellInSModule(Int_t absId, Double_t &xr, Double_t &yr, Double_t &zr)
  /// but taking into account position of shower max.
  ///
  /// \param absId: cell absolute id. number, input
  /// \param distEff: shower max position? check call in AliEMCALRecPoint!, input
  /// \param xr,yr,zr - x,y,z coordinates of cell with absId inside SM, output
  ///
  /// \return false if cell absId does not exist=
  Bool_t RelPosCellInSModule(Int_t absId, Double_t distEff, Double_t& xr, Double_t& yr, Double_t& zr) const;

  ///
  /// Methods for AliEMCALRecPoint: Look to see what the relative position inside a given cell is for a recpoint.
  ///
  /// \param absId: cell absolute id. number, input
  /// \param loc: Double[3] with x,y,z coordinates of cell with absId inside SM, output
  ///
  /// \return false if cell absId does not exist
  Bool_t RelPosCellInSModule(Int_t absId, Double_t loc[3]) const;

  ///
  /// Methods for AliEMCALRecPoint: Look to see what the relative position inside a given cell is for a recpoint.
  ///
  /// \param absId: cell absolute id. number, input
  /// \param vloc: TVector3 with x,y,z coordinates of cell with absId inside SM, output
  ///
  /// \return false if cell absId does not exist
  Bool_t RelPosCellInSModule(Int_t absId, TVector3& vloc) const;

  Int_t* GetEMCSystem() const { return mEMCGeometry->GetEMCSystem(); } // EMC System, SM type list
  // Local Coordinates of SM
  TArrayD GetCentersOfCellsEtaDir() const
  {
    return mCentersOfCellsEtaDir;
  } // size fNEta*fNETAdiv (for TRD1 only) (eta or z in SM, in cm)
  TArrayD GetCentersOfCellsXDir() const
  {
    return mCentersOfCellsXDir;
  } // size fNEta*fNETAdiv (for TRD1 only) (       x in SM, in cm)
  TArrayD GetCentersOfCellsPhiDir() const
  {
    return mCentersOfCellsPhiDir;
  } // size fNPhi*fNPHIdiv (for TRD1 only) (phi or y in SM, in cm)
  //
  TArrayD GetEtaCentersOfCells() const
  {
    return mEtaCentersOfCells;
  } // [fNEta*fNETAdiv*fNPhi*fNPHIdiv], positive direction (eta>0); eta depend from phi position;
  TArrayD GetPhiCentersOfCells() const
  {
    return mPhiCentersOfCells;
  } // [fNPhi*fNPHIdiv] from center of SM (-10. < phi < +10.)

  ///////////////////
  // useful utilities
  //
  Float_t AngleFromEta(Float_t eta) const
  { // returns theta in radians for a given pseudorapidity
    return 2.0 * TMath::ATan(TMath::Exp(-eta));
  }
  Float_t ZFromEtaR(Float_t r, Float_t eta) const
  { // returns z in for a given
    // pseudorapidity and r=sqrt(x*x+y*y).
    return r / TMath::Tan(AngleFromEta(eta));
  }

  ///
  /// Method to set shift-rotational matrixes from ESDHeader
  /// Move from header due to coding violations : Dec 2,2011 by PAI
  ///
  void SetMisalMatrix(const TGeoHMatrix* m, Int_t smod) const;

  ///
  /// Transform clusters cell position into global with alternative method, taking into account the depth calculation.
  /// Input are:
  ///    * the tower indeces,
  ///    * supermodule,
  ///    * particle type (photon 0, electron 1, hadron 2 )
  ///    * misalignment shifts to global position in case of need.
  ///
  ///  Federico.Ronchetti@cern.ch
  void RecalculateTowerPosition(Float_t drow, Float_t dcol, const Int_t sm, const Float_t depth,
                                const Float_t misaligTransShifts[15], const Float_t misaligRotShifts[15],
                                Float_t global[3]) const;

  ///
  /// Provides shift-rotation matrix for EMCAL from externally set matrix or
  /// from TGeoManager
  /// \return alignment matrix for a super module number
  /// \param smod: super module number
  ///
  const TGeoHMatrix* GetMatrixForSuperModule(Int_t smod) const;

  ///
  /// Provides shift-rotation matrix for EMCAL from the TGeoManager.
  /// \return alignment matrix for a super module number
  /// \param smod: super module number
  ///
  const TGeoHMatrix* GetMatrixForSuperModuleFromGeoManager(Int_t smod) const;

  ///
  /// Provides shift-rotation matrix for EMCAL from fkSModuleMatrix[smod]
  /// Unsafe method, not to be used in reconstruction, just check there is
  /// something in the array of matrices without crashing, for EVE checks.
  ///
  /// \return alignment matrix for a super module number
  /// \param smod: super module number
  ///
  const TGeoHMatrix* GetMatrixForSuperModuleFromArray(Int_t smod) const;

 protected:
  /// initializes the parameters of EMCAL
  void Init();

  EMCGeometry* mEMCGeometry; ///< Geometry object for Electromagnetic calorimeter

  TString mGeoName; ///< Geometry name string
  // Int_t    *fEMCSMSystem;	         ///< [fEMCGeometry.fNumberOfSuperModules] geometry structure
  Int_t mKey110DEG;           ///< For calculation abs cell id; 19-oct-05
  Int_t mnSupModInDCAL;       ///< For calculation abs cell id; 06-nov-12
  Int_t mNCellsInSupMod;      ///< Number cell in super module
  Int_t mNETAdiv;             ///< Number eta division of module
  Int_t mNPHIdiv;             ///< Number phi division of module
  Int_t mNCellsInModule;      ///< Number cell in module
  TArrayD mPhiBoundariesOfSM; ///< Phi boundaries of SM in rad; size is fNumberOfSuperModules;
  TArrayD mPhiCentersOfSM;    ///< Phi of centers of SM; size is fNumberOfSuperModules/2
  TArrayD mPhiCentersOfSMSec; ///< Phi of centers of section where SM lies; size is fNumberOfSuperModules/2

  // Local Coordinates of SM
  TArrayD mPhiCentersOfCells;    ///< [fNPhi*fNPHIdiv] from center of SM (-10. < phi < +10.)
  TArrayD mCentersOfCellsEtaDir; ///< Size fNEta*fNETAdiv (for TRD1 only) (eta or z in SM, in cm)
  TArrayD mCentersOfCellsPhiDir; ///< Size fNPhi*fNPHIdiv (for TRD1 only) (phi or y in SM, in cm)
  TArrayD
    mEtaCentersOfCells; ///< [fNEta*fNETAdiv*fNPhi*fNPHIdiv], positive direction (eta>0); eta depend from phi position;
  Int_t mNCells;        ///< Number of cells in calo
  Int_t mNPhi;          ///< Number of Towers in the PHI direction
  TArrayD mCentersOfCellsXDir;   ///< Size fNEta*fNETAdiv (for TRD1 only) (       x in SM, in cm)
  Float_t mEnvelop[3];           ///< The GEANT TUB for the detector
  Float_t mArm1EtaMin;           ///< Minimum pseudorapidity position of EMCAL in Eta
  Float_t mArm1EtaMax;           ///< Maximum pseudorapidity position of EMCAL in Eta
  Float_t mArm1PhiMin;           ///< Minimum angular position of EMCAL in Phi (degrees)
  Float_t mArm1PhiMax;           ///< Maximum angular position of EMCAL in Phi (degrees)
  Float_t mEtaMaxOfTRD1;         ///< Max eta in case of TRD1 geometry (see AliEMCALShishKebabTrd1Module)
  Float_t mDCALPhiMin;           ///< Minimum angular position of DCAL in Phi (degrees)
  Float_t mDCALPhiMax;           ///< Maximum angular position of DCAL in Phi (degrees)
  Float_t mEMCALPhiMax;          ///< Maximum angular position of EMCAL in Phi (degrees)
  Float_t mDCALStandardPhiMax;   ///< Special edge for the case that DCAL contian extension
  Float_t mDCALInnerExtandedEta; ///< DCAL inner edge in Eta (with some extension)
  TList* mShishKebabTrd1Modules; ///< List of modules
  Float_t mParSM[3];             ///< SM sizes as in GEANT (TRD1)
  Float_t mPhiModuleSize;        ///< Phi -> X
  Float_t mEtaModuleSize;        ///< Eta -> Y
  Float_t mPhiTileSize;          ///< Size of phi tile
  Float_t mEtaTileSize;          ///< Size of eta tile
  Int_t mNZ;                     ///< Number of Towers in the Z direction
  Float_t mIPDistance;           ///< Radial Distance of the inner surface of the EMCAL
  Float_t mLongModuleSize;       ///< Size of long module

  // Geometry Parameters
  Float_t mShellThickness; ///< Total thickness in (x,y) direction
  Float_t mZLength;        ///< Total length in z direction
  Float_t mSampling;       ///< Sampling factor

  mutable const TGeoHMatrix* SMODULEMATRIX[EMCAL_MODULES]; ///< Orientations of EMCAL super modules

 private:
  static Geometry* sGeom;                    ///< Pointer to the unique instance of the singleton
  static Bool_t sInit;                       ///< Tells if geometry has been succesfully set up.
  static const Char_t* sDefaultGeometryName; ///< Default name of geometry

  ClassDef(Geometry, 1);
};
}
}
#endif

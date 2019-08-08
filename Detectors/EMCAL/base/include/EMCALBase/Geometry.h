// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_GEOMETRY_H_
#define ALICEO2_EMCAL_GEOMETRY_H_

#include <exception>
#include <string>
#include <tuple>
#include <vector>

#include <RStringView.h>
#include <TGeoMatrix.h>
#include <TNamed.h>
#include <TParticle.h>
#include <TVector3.h>

#include "DataFormatsEMCAL/Constants.h"
#include "EMCALBase/GeometryBase.h"
#include "MathUtils/Cartesian3D.h"

namespace o2
{
namespace emcal
{
class ShishKebabTrd1Module;

class Geometry
{
 public:
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
  Geometry(const std::string_view name, const std::string_view mcname = "", const std::string_view mctitle = "");

  ///
  /// Copy constructor.
  ///
  Geometry(const Geometry& geom);

  ///
  /// Destructor.
  ///
  ~Geometry();

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
  static Geometry* GetInstance(const std::string_view name, const std::string_view mcname = "TGeant3",
                               const std::string_view mctitle = "");

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
  static Geometry* GetInstanceFromRunNumber(Int_t runNumber, const std::string_view = "",
                                            const std::string_view mcname = "TGeant3",
                                            const std::string_view mctitle = "");

  /// Set the value of the Sampling used to calibrate the MC hits energy (check)
  /// Called in Detector::ConstructGeometry and not anymore here in Init() in order to be able to work with Geant4
  ///
  /// \param mcname: Geant3/4, Flukla, ...
  /// \param mctitle: Geant4 physics list tag name
  void DefineSamplingFraction(const std::string_view mcname = "", const std::string_view mctitle = "");

  //////////
  // General
  //

  const std::string& GetName() const { return mGeoName; }

  static const std::string& GetDefaultGeometryName() { return DEFAULT_GEOMETRY; }

  static Bool_t IsInitialized() { return Geometry::sGeom != nullptr; }

  ///
  /// Generate the list of Trd1 modules
  /// which will make up the EMCAL geometry
  /// key: look to the AliEMCALShishKebabTrd1Module::
  ///
  void CreateListOfTrd1Modules();

  const std::vector<ShishKebabTrd1Module>& GetShishKebabTrd1Modules() const { return mShishKebabTrd1Modules; }

  ///
  /// \return  the shishkebabmodule at a given eta index point.
  ///
  const ShishKebabTrd1Module& GetShishKebabModule(Int_t neta) const;

  ///
  /// Given a TParticle, check if it falls in the EMCal/DCal geometry
  /// Call ImpactOnEmcal.
  ///
  /// \param particle TParticle
  /// \return true in EMCal/DCa;
  ///
  Bool_t Impact(const TParticle* particle) const;

  ///
  /// Calculates the impact coordinates on EMCAL (centre of a tower/not on EMCAL surface)
  /// of a neutral particle emitted in the vertex vtx[3] with direction theta and phi in
  /// the global coordinate system
  ///
  /// \param[in] vtx TVector3 with vertex
  /// \param[in] theta theta location
  /// \param[in] phi azimuthal angle
  /// \param[out] absId absolute ID number
  /// \param[out] vimpact TVector3 of impact coordinates?
  ///
  void ImpactOnEmcal(const Point3D<double>& vtx, Double_t theta, Double_t phi, Int_t& absId, Point3D<double>& vimpact) const;

  ///
  /// Checks whether point is inside the EMCal volume
  ///
  Bool_t IsInEMCAL(const Point3D<double>& pnt) const;

  ///
  /// Checks whether point is inside the DCal volume
  ///
  Bool_t IsInDCAL(const Point3D<double>& pnt) const;

  ///
  /// Checks whether point is inside the EMCal volume (included DCal), used in AliEMCALv*.cxx
  /// Code uses cylindrical approximation made of inner radius (for speed)
  ///
  /// Points behind EMCAl/DCal, i.e. R > outer radius, but eta, phi in acceptance
  /// are considered to inside
  ///
  /// \return calo acceptance type
  ///
  AcceptanceType_t IsInEMCALOrDCAL(const Point3D<double>& pnt) const;

  //////////////////////////////////////
  // Return EMCAL geometrical parameters
  //

  const Char_t* GetNameOfEMCALEnvelope() const { return "XEN1"; }
  Float_t GetArm1PhiMin() const { return mArm1PhiMin; }
  Float_t GetArm1PhiMax() const { return mArm1PhiMax; }
  Float_t GetArm1EtaMin() const { return mArm1EtaMin; }
  Float_t GetArm1EtaMax() const { return mArm1EtaMax; }
  Float_t GetIPDistance() const { return mIPDistance; }
  Float_t GetEnvelop(Int_t index) const { return mEnvelop[index]; }
  Float_t GetShellThickness() const { return mShellThickness; }
  Float_t GetZLength() const { return mZLength; }
  Float_t GetDCALInnerEdge() const { return mDCALInnerEdge; }
  Float_t GetDCALPhiMin() const { return mDCALPhiMin; }
  Float_t GetDCALPhiMax() const { return mDCALPhiMax; }
  Float_t GetEMCALPhiMax() const { return mEMCALPhiMax; }
  Float_t GetDCALStandardPhiMax() const { return mDCALStandardPhiMax; }
  Int_t GetNECLayers() const { return mNECLayers; }
  Float_t GetDCALInnerExtandedEta() const { return mDCALInnerExtandedEta; }
  Int_t GetNZ() const { return mNZ; }
  Int_t GetNEta() const { return mNZ; }
  Int_t GetNPhi() const { return mNPhi; }
  Float_t GetECPbRadThick() const { return mECPbRadThickness; }
  Float_t GetECScintThick() const { return mECScintThick; }
  Float_t GetSampling() const { return mSampling; }
  Int_t GetNumberOfSuperModules() const { return mNumberOfSuperModules; }
  Float_t GetPhiGapForSuperModules() const { return mPhiGapForSM; }
  Float_t GetPhiModuleSize() const { return mPhiModuleSize; }
  Float_t GetEtaModuleSize() const { return mEtaModuleSize; }
  Float_t GetFrontSteelStrip() const { return mFrontSteelStrip; }
  Float_t GetLateralSteelStrip() const { return mLateralSteelStrip; }
  Float_t GetPassiveScintThick() const { return mPassiveScintThick; }
  Float_t GetPhiTileSize() const { return mPhiTileSize; }
  Float_t GetEtaTileSize() const { return mEtaTileSize; }
  Float_t GetPhiSuperModule() const { return mPhiSuperModule; }
  Int_t GetNPhiSuperModule() const { return mNPhiSuperModule; }
  Int_t GetNPHIdiv() const { return mNPHIdiv; }
  Int_t GetNETAdiv() const { return mNETAdiv; }
  Int_t GetNCells() const { return mNCells; }
  Float_t GetLongModuleSize() const { return mLongModuleSize; }
  Float_t GetTrd1Angle() const { return mTrd1Angle; }
  Float_t Get2Trd1Dx2() const { return m2Trd1Dx2; }
  Float_t GetTrd1AlFrontThick() const { return mTrd1AlFrontThick; }
  Float_t GetTrd1BondPaperThick() const { return mTrd1BondPaperThick; }
  // --
  Int_t GetNCellsInSupMod() const { return mNCellsInSupMod; }
  Int_t GetNCellsInModule() const { return mNCellsInModule; }
  Int_t GetKey110DEG() const { return mKey110DEG; }
  Int_t GetnSupModInDCAL() const { return mnSupModInDCAL; }
  Int_t GetILOSS() const { return mILOSS; }
  Int_t GetIHADR() const { return mIHADR; }
  // --
  Float_t GetDeltaEta() const { return (mArm1EtaMax - mArm1EtaMin) / ((Float_t)mNZ); }
  Float_t GetDeltaPhi() const { return (mArm1PhiMax - mArm1PhiMin) / ((Float_t)mNPhi); }
  Int_t GetNTowers() const { return mNPhi * mNZ; }
  //
  Double_t GetPhiCenterOfSM(Int_t nsupmod) const;
  Double_t GetPhiCenterOfSMSec(Int_t nsupmod) const;
  Float_t GetSuperModulesPar(Int_t ipar) const { return mParSM[ipar]; }
  //
  EMCALSMType GetSMType(Int_t nSupMod) const
  {
    if (nSupMod > mNumberOfSuperModules)
      return NOT_EXISTENT;
    return mEMCSMSystem[nSupMod];
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
  std::tuple<double, double> GetPhiBoundariesOfSM(Int_t nSupMod) const;
  std::tuple<double, double> GetPhiBoundariesOfSMGap(Int_t nPhiSec) const;

  // Obsolete?
  Float_t GetSteelFrontThickness() const { return mSteelFrontThick; }

  ///////////////////////////////
  // Geometry data member setters
  //
  void SetNZ(Int_t nz) { mNZ = nz; }
  void SetNPhi(Int_t nphi) { mNPhi = nphi; }
  //
  void SetSampling(Float_t samp) { mSampling = samp; }

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
  /// \param absId cell absolute id. number.
  /// \return tuple with (pseudorapidity, polar angle)
  ///
  std::tuple<double, double> EtaPhiFromIndex(Int_t absId) const;

  ///
  /// Get cell absolute ID number from eta and phi location.
  ///
  /// \param eta pseudorapidity location
  /// \param phi azimutal location
  /// \return cell absolute ID number
  /// \thow InvalidPositionException
  ///
  int GetAbsCellIdFromEtaPhi(Double_t eta, Double_t phi) const;

  /// \brief get (Column,Row) pair of cell in global numbering scheme
  /// \param cellID Absolute cell ID
  /// \return tuple with position in global numbering scheme (0 - row, 1 - column)
  std::tuple<int, int> GlobalRowColFromIndex(int cellID) const;

  /// \brief Get column number of cell in global numbering scheme
  /// \param cellID Absolute cell ID
  /// \return Column number in global numbering scheme
  int GlobalCol(int cellID) const;

  /// \brief Get row number of cell in global numbering scheme
  /// \param cellID Absolute cell ID
  /// \return Row number in global numbering scheme
  int GlobalRow(int cellID) const;

  ///
  /// Given a global eta/phi point check if it belongs to a supermodule covered region.
  ///
  /// \param eta pseudorapidity location
  /// \param phi azimutal location
  /// \return super module number
  /// \throw InvalidPositionException
  ///
  int SuperModuleNumberFromEtaPhi(Double_t eta, Double_t phi) const;

  ///
  /// Get cell absolute ID number from location module (2 times 2 cells) of a super module
  ///
  /// \param nSupMod super module number
  /// \param nModule module number
  /// \param nIphi index of cell in module in phi direction 0 or 1
  /// \param nIeta index of cell in module in eta direction 0 or 1
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
  /// \param absId cell absolute id. number
  /// \param nSupMod super module number
  /// \param nModule module number
  ///
  /// \return tuple(supermodule ID, module number, index of cell in module in phi, index of cell in module in eta)
  /// \throw InvalidCellIDException
  ///
  std::tuple<int, int, int, int> GetCellIndex(Int_t absId) const;

  ///
  /// \brief Get eta-phi indexes of module in SM
  ///
  /// \param[in] nSupMod: super module number, input
  /// \param[in] nModule: module number, input
  /// \return tuple (index in phi direction of module, index in eta direction of module)
  ///
  std::tuple<int, int> GetModulePhiEtaIndexInSModule(Int_t nSupMod, Int_t nModule) const;

  ///
  /// \brief Get eta-phi indexes of cell in SM
  ///
  /// \param nSupMod[in] super module number
  /// \param nModule[in] module number
  /// \param nIphi[in] index in phi direction in module
  /// \param nIeta[in] index in phi direction in module
  /// \return tuple (index in phi direction in super module, index in eta direction in super module)
  ///
  std::tuple<int, int> GetCellPhiEtaIndexInSModule(Int_t nSupMod, Int_t nModule, Int_t nIphi, Int_t nIeta) const;

  ///
  /// \brief Get cell SM,  from absolute ID number
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
  /// Transition from cell indexes (iphi, ieta) to module indexes (iphim, ietam, nModule)
  //
  /// \param nSupMod: super module number
  /// \param iphi: index of cell in phi direction inside super module
  /// \param ieta: index of cell in eta direction inside super module
  /// \return tuple:
  ///               iphim: index of cell in module in phi direction: 0 or 1
  ///               ietam: index of cell in module in eta direction: 0 or 1
  ///               nModule: module number
  ///
  std::tuple<Int_t, Int_t, Int_t> GetModuleIndexesFromCellIndexesInSModule(Int_t nSupMod, Int_t iphi, Int_t ieta) const;

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
  /// \brief Look to see what the relative position inside a given cell is for a recpoint.
  ///
  /// Same as RelPosCellInSModule(Int_t absId, Double_t &xr, Double_t &yr, Double_t &zr)
  /// but taking into account position of shower max.
  ///
  /// \param[in] absId cell absolute id. number, input
  /// \param[in] distEff shower max position? check call in RecPoint!
  /// \return Point3D with x,y,z coordinates of cell with absId inside SM
  /// \throw InvalidCellIDException if cell ID does not exist
  ///
  Point3D<double> RelPosCellInSModule(Int_t absId, Double_t distEf) const;

  ///
  /// \brief Look to see what the relative position inside a given cell is for a recpoint.
  ///
  /// \param absId cell absolute id. number, input
  /// \return Point3D with x,y,z coordinates of cell with absId inside SM
  /// \throw InvalidCellIDException if cell ID does not exist
  ///
  Point3D<double> RelPosCellInSModule(Int_t absId) const;

  std::vector<EMCALSMType> GetEMCSystem() const { return mEMCSMSystem; } // EMC System, SM type list
  // Local Coordinates of SM
  std::vector<Double_t> GetCentersOfCellsEtaDir() const
  {
    return mCentersOfCellsEtaDir;
  } // size fNEta*fNETAdiv (for TRD1 only) (eta or z in SM, in cm)
  std::vector<Double_t> GetCentersOfCellsXDir() const
  {
    return mCentersOfCellsXDir;
  } // size fNEta*fNETAdiv (for TRD1 only) (       x in SM, in cm)
  std::vector<Double_t> GetCentersOfCellsPhiDir() const
  {
    return mCentersOfCellsPhiDir;
  } // size fNPhi*fNPHIdiv (for TRD1 only) (phi or y in SM, in cm)
  //
  std::vector<Double_t> GetEtaCentersOfCells() const
  {
    return mEtaCentersOfCells;
  } // [fNEta*fNETAdiv*fNPhi*fNPHIdiv], positive direction (eta>0); eta depend from phi position;
  std::vector<Double_t> GetPhiCentersOfCells() const
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
  /// \brief Provides shift-rotation matrix for EMCAL from externally set matrix or
  /// from TGeoManager
  /// \return alignment matrix for a super module number
  /// \param smod: super module number
  ///
  const TGeoHMatrix* GetMatrixForSuperModule(Int_t smod) const;

  ///
  /// \brief Provides shift-rotation matrix for EMCAL from the TGeoManager.
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
  /// \brief initializes the parameters of EMCAL
  void Init();

  /// Init function of previous class EMCGeometry
  void DefineEMC(std::string_view mcname, std::string_view mctitle);

  std::string mGeoName;                     ///< Geometry name string
  Int_t mKey110DEG;                         ///< For calculation abs cell id; 19-oct-05
  Int_t mnSupModInDCAL;                     ///< For calculation abs cell id; 06-nov-12
  Int_t mNCellsInSupMod;                    ///< Number cell in super module
  Int_t mNETAdiv;                           ///< Number eta division of module
  Int_t mNPHIdiv;                           ///< Number phi division of module
  Int_t mNCellsInModule;                    ///< Number cell in module
  std::vector<Double_t> mPhiBoundariesOfSM; ///< Phi boundaries of SM in rad; size is fNumberOfSuperModules;
  std::vector<Double_t> mPhiCentersOfSM;    ///< Phi of centers of SM; size is fNumberOfSuperModules/2
  std::vector<Double_t> mPhiCentersOfSMSec; ///< Phi of centers of section where SM lies; size is fNumberOfSuperModules/2

  // Local Coordinates of SM
  std::vector<Double_t> mPhiCentersOfCells;    ///< [fNPhi*fNPHIdiv] from center of SM (-10. < phi < +10.)
  std::vector<Double_t> mCentersOfCellsEtaDir; ///< Size fNEta*fNETAdiv (for TRD1 only) (eta or z in SM, in cm)
  std::vector<Double_t> mCentersOfCellsPhiDir; ///< Size fNPhi*fNPHIdiv (for TRD1 only) (phi or y in SM, in cm)
  std::vector<Double_t>
    mEtaCentersOfCells;                                     ///< [fNEta*fNETAdiv*fNPhi*fNPHIdiv], positive direction (eta>0); eta depend from phi position;
  Int_t mNCells;                                            ///< Number of cells in calo
  Int_t mNPhi;                                              ///< Number of Towers in the PHI direction
  std::vector<Double_t> mCentersOfCellsXDir;                ///< Size fNEta*fNETAdiv (for TRD1 only) (       x in SM, in cm)
  Float_t mEnvelop[3];                                      ///< The GEANT TUB for the detector
  Float_t mArm1EtaMin;                                      ///< Minimum pseudorapidity position of EMCAL in Eta
  Float_t mArm1EtaMax;                                      ///< Maximum pseudorapidity position of EMCAL in Eta
  Float_t mArm1PhiMin;                                      ///< Minimum angular position of EMCAL in Phi (degrees)
  Float_t mArm1PhiMax;                                      ///< Maximum angular position of EMCAL in Phi (degrees)
  Float_t mEtaMaxOfTRD1;                                    ///< Max eta in case of TRD1 geometry (see AliEMCALShishKebabTrd1Module)
  Float_t mDCALPhiMin;                                      ///< Minimum angular position of DCAL in Phi (degrees)
  Float_t mDCALPhiMax;                                      ///< Maximum angular position of DCAL in Phi (degrees)
  Float_t mEMCALPhiMax;                                     ///< Maximum angular position of EMCAL in Phi (degrees)
  Float_t mDCALStandardPhiMax;                              ///< Special edge for the case that DCAL contian extension
  Float_t mDCALInnerExtandedEta;                            ///< DCAL inner edge in Eta (with some extension)
  Float_t mDCALInnerEdge;                                   ///< Inner edge for DCAL
  std::vector<ShishKebabTrd1Module> mShishKebabTrd1Modules; ///< List of modules
  Float_t mParSM[3];                                        ///< SM sizes as in GEANT (TRD1)
  Float_t mPhiModuleSize;                                   ///< Phi -> X
  Float_t mEtaModuleSize;                                   ///< Eta -> Y
  Float_t mPhiTileSize;                                     ///< Size of phi tile
  Float_t mEtaTileSize;                                     ///< Size of eta tile
  Int_t mNZ;                                                ///< Number of Towers in the Z direction
  Float_t mIPDistance;                                      ///< Radial Distance of the inner surface of the EMCAL
  Float_t mLongModuleSize;                                  ///< Size of long module

  // Geometry Parameters
  Float_t mShellThickness; ///< Total thickness in (x,y) direction
  Float_t mZLength;        ///< Total length in z direction
  Float_t mSampling;       ///< Sampling factor

  // Members from the EMCGeometry class
  Float_t mECPbRadThickness; ///< cm, Thickness of the Pb radiators
  Float_t mECScintThick;     ///< cm, Thickness of the scintillators
  Int_t mNECLayers;          ///< number of scintillator layers

  // Shish-kebab option - 23-aug-04 by PAI; COMPACT, TWIST, TRD1 and TRD2
  Int_t mNumberOfSuperModules; ///< default is 12 = 6 * 2

  /// geometry structure
  std::vector<EMCALSMType> mEMCSMSystem; ///< Type of the supermodule (size number of supermodules

  Float_t mFrontSteelStrip;   ///< 13-may-05
  Float_t mLateralSteelStrip; ///< 13-may-05
  Float_t mPassiveScintThick; ///< 13-may-05

  Float_t mPhiSuperModule; ///< Phi of normal supermodule (20, in degree)
  Int_t mNPhiSuperModule;  ///< 9 - number supermodule in phi direction

  // TRD1 options - 30-sep-04
  Float_t mTrd1Angle;   ///< angle in x-z plane (in degree)
  Float_t m2Trd1Dx2;    ///< 2*dx2 for TRD1
  Float_t mPhiGapForSM; ///< Gap betweeen supermodules in phi direction

  // Oct 26,2010
  Float_t mTrd1AlFrontThick;   ///< Thickness of the Al front plate
  Float_t mTrd1BondPaperThick; ///< Thickness of the Bond Paper sheet

  Int_t mILOSS; ///< Options for Geant (MIP business) - will call in AliEMCAL
  Int_t mIHADR; ///< Options for Geant (MIP business) - will call in AliEMCAL

  Float_t mSteelFrontThick; ///< Thickness of the front stell face of the support box - 9-sep-04; obsolete?

  mutable const TGeoHMatrix* SMODULEMATRIX[EMCAL_MODULES]; ///< Orientations of EMCAL super modules

 private:
  static Geometry* sGeom; ///< Pointer to the unique instance of the singleton
};

inline Bool_t Geometry::CheckAbsCellId(Int_t absId) const
{
  if (absId < 0 || absId >= mNCells)
    return kFALSE;
  else
    return kTRUE;
}
} // namespace emcal
} // namespace o2
#endif

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_EMCGEOMETRY_H_
#define ALICEO2_EMCAL_EMCGEOMETRY_H_

#include <array>
#include <iosfwd>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include <RStringView.h>
#include <TArrayD.h>
#include <TList.h>
#include <TMath.h>

#include <FairLogger.h>

#include <EMCALBase/GeometryBase.h>

class TObjArray;

namespace o2
{
namespace EMCAL
{
class EMCGeometry
{
 public:
  ///
  /// Default ctor only for internal usage (singleton).
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world.
  ///
  EMCGeometry() = default;

  ///
  /// Copy constructor.
  ///
  EMCGeometry(const EMCGeometry& geom);

  ///
  /// Constructor only for internal usage (singleton).
  ///
  /// \param name: geometry name, EMCAL_COMPLETEV1, EMCAL_COMPLETE12SMV1, EMCAL_COMPLETE12SMV1_DCAL,
  /// EMCAL_COMPLETE12SMV1_DCAL_8SM, EMCAL_COMPLETE12SMV1_DCAL_DEV (see main class description for definition) \param
  /// title \param mcname: Geant3/4, Flukla, ... \param mctitle: Geant4 physics list tag name
  ///
  EMCGeometry(const std::string_view name, const std::string_view mcname = "", const std::string_view mctitle = "");

  ///
  /// Destructor
  ///
  ~EMCGeometry();

  /// Assignement operator requested by coding convention but not needed
  EMCGeometry& operator=(const EMCGeometry& /*rvalue*/)
  {
    LOG(FATAL) << "operator = not implemented\n";
    return *this;
  };

  //////////
  // General
  //

  const std::string& GetName() const { return mGeoName; }
  Bool_t IsInitialized() const { return sInit; }
  static const std::string& GetDefaultGeometryName() { return sDefaultGeometryName; }

  ///
  /// Print EMCal parameters
  ///
  void PrintStream(std::ostream& stream) const; //*MENU*

  ///
  /// It initializes the EMCAL parameters based on the name.
  /// Only Shashlyk geometry is available, but various combinations of
  /// layers and number of supermodules can be selected with additional
  /// options or geometry name
  ///
  /// \param mcname: Geant3/4, Fluka, needed for settings of transport (not needed since 15/03/16)
  /// \param mctitle: Geant4 physics list ((not needed since 15/03/16))
  ///
  void Init(const std::string_view mcname = "", const std::string_view mctitle = "");

  ///
  /// Additional options that can be used to select
  /// the specific geometry of EMCAL to run
  ///
  void CheckAdditionalOptions();

  ///
  /// Set the value of fSampling used to calibrate the MC hits energy (check)
  /// Called in AliEMCALv0 and not anymore here in Init() in order to be able to work with Geant4
  ///
  /// \param mcname: Geant3/4, Flukla, ...
  /// \param mctitle: Geant4 physics list tag name
  ///
  void DefineSamplingFraction(const std::string_view mcname = "", const std::string_view mctitle = "");

  //////////////////////////////////////
  // Return EMCAL geometrical parameters
  //

  const std::string& GetGeoName() const { return mGeoName; }

  const Int_t* GetEMCSystem() const { return mEMCSMSystem; }
  Int_t* GetEMCSystem() { return mEMCSMSystem; } // Why? GCB

  const Char_t* GetNameOfEMCALEnvelope() const
  {
    const Char_t* env = "XEN1";
    return env;
  }

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
  Float_t GetDCALInnerExtandedEta() const { return mDCALInnerExtandedEta; }
  Float_t GetEMCALPhiMax() const { return mEMCALPhiMax; }
  Float_t GetDCALStandardPhiMax() const { return mDCALStandardPhiMax; }

  Int_t GetNECLayers() const { return mNECLayers; }
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
  Float_t GetEtaMaxOfTRD1() const { return mEtaMaxOfTRD1; }
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

  ///
  /// \return center of supermodule in phi
  ///
  Double_t GetPhiCenterOfSM(Int_t nsupmod) const;
  ///
  /// \return center of supermodule in phi sector
  ///
  Double_t GetPhiCenterOfSMSec(Int_t nsupmod) const;
  Float_t GetSuperModulesPar(Int_t ipar) const { return mParSM[ipar]; }
  Int_t GetSMType(Int_t nSupMod) const
  {
    if (nSupMod > GetNumberOfSuperModules())
      return NOT_EXISTENT;
    return mEMCSMSystem[nSupMod];
  }

  ///
  /// SM boundaries
  ///
  /// \param[in] nSupMod: super module index
  /// \return tuple with (min, max) phi value in radians
  ///
  std::tuple<double, double> GetPhiBoundariesOfSM(Int_t nSupMod) const;

  ///
  /// SM boundaries between gaps
  ///
  /// \param[in] nPhiSec: super module sector index
  /// \return tuple with (min, max) phi value in radians
  ///
  /// * 0;  gap boundaries between  0th&2th  | 1th&3th SM
  /// * 1;  gap boundaries between  2th&4th  | 3th&5th SM
  /// * 2;  gap boundaries between  4th&6th  | 5th&7th SM
  /// * 3;  gap boundaries between  6th&8th  | 7th&9th SM
  /// * 4;  gap boundaries between  8th&10th | 9th&11th SM
  /// * 5;  gap boundaries between 10th&12th | 11h&13th SM
  ///
  std::tuple<double, double> GetPhiBoundariesOfSMGap(Int_t nPhiSec) const;

  ///
  /// Play with strings names and modify them for better handling (?)
  ///
  static int ParseString(const TString& topt, TObjArray& Opt);

  ///////////////////////////////
  // Geometry data member setters
  //
  void SetNZ(Int_t nz)
  {
    mNZ = nz;
    LOG(INFO) << "SetNZ: Number of modules in Z set to " << mNZ << FairLogger::endl;
  }
  void SetNPhi(Int_t nphi)
  {
    mNPhi = nphi;
    LOG(INFO) << "SetNPhi: Number of modules in Phi set to " << mNPhi << FairLogger::endl;
  }
  void SetSampling(Float_t samp)
  {
    mSampling = samp;
    LOG(INFO) << "SetSampling: Sampling factor set to " << mSampling << FairLogger::endl;
  }

  ///////////////////
  // useful utilities
  //
  /// \return theta in radians for a given pseudorapidity
  Float_t AngleFromEta(Float_t eta) const { return 2.0 * TMath::ATan(TMath::Exp(-eta)); }

  /// \return z in for a given pseudorapidity and r=sqrt(x*x+y*y).
  Float_t ZFromEtaR(Float_t r, Float_t eta) const { return r / TMath::Tan(AngleFromEta(eta)); }

  //////////////////////////////////////////////////
  // Obsolete?
  Float_t GetSteelFrontThickness() const { return mSteelFrontThick; }
  //////////////////////////////////////////////////

  static std::string sDefaultGeometryName; ///< Default name of geometry
  static Bool_t sInit;                     ///< Tells if geometry has been succesfully set up.

 private:
  // Member data

  std::string mGeoName; ///< geometry name

  TObjArray* mArrayOpts;                      //!<! array of geometry options
  std::array<std::string, 6> mAdditionalOpts; //!<! some additional options for the geometry type and name
  Int_t mNAdditionalOpts;                     //!<! size of additional options parameter

  Float_t mECPbRadThickness; ///< cm, Thickness of the Pb radiators
  Float_t mECScintThick;     ///< cm, Thickness of the scintillators
  Int_t mNECLayers;          ///< number of scintillator layers

  Float_t mArm1PhiMin; ///< Minimum angular position of EMCAL in Phi (degrees)
  Float_t mArm1PhiMax; ///< Maximum angular position of EMCAL in Phi (degrees)
  Float_t mArm1EtaMin; ///< Minimum pseudorapidity position of EMCAL in Eta
  Float_t mArm1EtaMax; ///< Maximum pseudorapidity position of EMCAL in Eta

  // Geometry Parameters
  Float_t mEnvelop[3];           ///< the GEANT TUB for the detector
  Float_t mIPDistance;           ///< Radial Distance of the inner surface of the EMCAL
  Float_t mShellThickness;       ///< Total thickness in (x,y) direction
  Float_t mZLength;              ///< Total length in z direction
  Float_t mDCALInnerEdge;        ///< Inner edge for DCAL
  Float_t mDCALPhiMin;           ///< Minimum angular position of DCAL in Phi (degrees)
  Float_t mDCALPhiMax;           ///< Maximum angular position of DCAL in Phi (degrees)
  Float_t mEMCALPhiMax;          ///< Maximum angular position of EMCAL in Phi (degrees)
  Float_t mDCALStandardPhiMax;   ///< special edge for the case that DCAL contian extension
  Float_t mDCALInnerExtandedEta; ///< DCAL inner edge in Eta (with some extension)
  Int_t mNZ;                     ///< Number of Towers in the Z direction
  Int_t mNPhi;                   ///< Number of Towers in the PHI direction
  Float_t mSampling;             ///< Sampling factor

  // Shish-kebab option - 23-aug-04 by PAI; COMPACT, TWIST, TRD1 and TRD2
  Int_t mNumberOfSuperModules; ///< default is 12 = 6 * 2

  /// geometry structure
  Int_t* mEMCSMSystem; //[mNumberOfSuperModules]

  Float_t mFrontSteelStrip;   ///< 13-may-05
  Float_t mLateralSteelStrip; ///< 13-may-05
  Float_t mPassiveScintThick; ///< 13-may-05

  Float_t mPhiModuleSize; ///< Phi -> X
  Float_t mEtaModuleSize; ///< Eta -> Y
  Float_t mPhiTileSize;   ///< Size of phi tile
  Float_t mEtaTileSize;   ///< Size of eta tile

  Float_t mLongModuleSize; ///< Size of long module
  Float_t mPhiSuperModule; ///< Phi of normal supermodule (20, in degree)
  Int_t mNPhiSuperModule;  ///< 9 - number supermodule in phi direction

  Int_t mNPHIdiv; ///< number phi divizion of module
  Int_t mNETAdiv; ///< number eta divizion of module
  //
  Int_t mNCells;         ///< number of cells in calo
  Int_t mNCellsInSupMod; ///< number cell in super module
  Int_t mNCellsInModule; ///< number cell in module)

  // TRD1 options - 30-sep-04
  Float_t mTrd1Angle;         ///< angle in x-z plane (in degree)
  Float_t m2Trd1Dx2;          ///< 2*dx2 for TRD1
  Float_t mPhiGapForSM;       ///< Gap betweeen supermodules in phi direction
  Int_t mKey110DEG;           ///< for calculation abs cell id; 19-oct-05
  Int_t mnSupModInDCAL;       ///< for calculation abs cell id;
  TArrayD mPhiBoundariesOfSM; ///< phi boundaries of SM in rad; size is fNumberOfSuperModules;
  TArrayD mPhiCentersOfSM;    ///< phi of centers of SM; size is fNumberOfSuperModules/2
  TArrayD mPhiCentersOfSMSec; ///< phi of centers of section where SM lies; size is fNumberOfSuperModules/2
  Float_t mEtaMaxOfTRD1;      ///< max eta in case of TRD1 geometry (see AliEMCALShishKebabTrd1Module)

  // Oct 26,2010
  Float_t mTrd1AlFrontThick;   ///< Thickness of the Al front plate
  Float_t mTrd1BondPaperThick; ///< Thickness of the Bond Paper sheet

  // Local Coordinates of SM
  TArrayD mCentersOfCellsEtaDir; ///< size fNEta*fNETAdiv (for TRD1 only) (eta or z in SM, in cm)
  TArrayD mCentersOfCellsXDir;   ///< size fNEta*fNETAdiv (for TRD1 only) (       x in SM, in cm)
  TArrayD mCentersOfCellsPhiDir; ///< size fNPhi*fNPHIdiv (for TRD1 only) (phi or y in SM, in cm)
  //
  TArrayD
    mEtaCentersOfCells; ///< [fNEta*fNETAdiv*fNPhi*fNPHIdiv], positive direction (eta>0); eta depend from phi position;
  TArrayD mPhiCentersOfCells; ///< [fNPhi*fNPHIdiv] from center of SM (-10. < phi < +10.)

  // Local coordinates of SM for TRD1
  Float_t mParSM[3]; ///< SM sizes as in GEANT (TRD1)

  Int_t mILOSS; ///< Options for Geant (MIP business) - will call in AliEMCAL
  Int_t mIHADR; ///< Options for Geant (MIP business) - will call in AliEMCAL

  Float_t mSteelFrontThick; ///< Thickness of the front stell face of the support box - 9-sep-04; obsolete?
};

std::ostream& operator<<(std::ostream& stream, const EMCGeometry& geo);
}
}

#endif

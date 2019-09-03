// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file MagF.cxx
/// \brief Implementation of the MagF class
/// \author ruben.shahoyan@cern.ch

#include "Field/MagneticField.h"
#include <TFile.h>      // for TFile
#include <TPRegexp.h>   // for TPRegexp
#include <TSystem.h>    // for TSystem, gSystem
#include "FairLogger.h" // for FairLogger
#include "FairParamList.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"

using namespace o2::field;

ClassImp(MagneticField);

const Double_t MagneticField::sSolenoidToDipoleZ = -700.;

/// Explanation for polarity conventions: these are the mapping between the
/// current signs and main field components in L3 (Bz) and Dipole (Bx) (in Alice frame)
/// 1) kConvMap2005: used for the field mapping in 2005
/// positive L3  current -> negative Bz
/// positive Dip current -> positive Bx
/// 2) kConvMapDCS2008: defined by the microswitches/cabling of power converters as of 2008 - 1st half 2009
/// positive L3  current -> positive Bz
/// positive Dip current -> positive Bx
/// 3) kConvLHC : defined by LHC
/// positive L3  current -> positive Bz
/// positive Dip current -> negative Bx
///
/// Note: only "negative Bz(L3) with postive Bx(Dipole)" and its inverse was mapped in 2005. Hence
/// the GRP Manager will reject the runs with the current combinations (in the convention defined by the
/// static Int_t MagneticField::getPolarityConvention()) which do not lead to such field polarities.
///
/// Explanation on integrals in the TPC region
/// getTPCInt(xyz,b) and getTPCRatInt(xyz,b) give integrals from point (x,y,z) to point (x,y,0)
/// (irrespectively of the z sign) of the following:
/// TPCInt:    b contains int{bx}, int{by}, int{bz}
/// TPCRatInt: b contains int{bx/bz}, int{by/bz}, int{(bx/bz)^2+(by/bz)^2}
///
/// The same applies to integral in cylindrical coordinates:
/// getTPCIntCyl(rphiz,b)
/// getTPCIntRatCyl(rphiz,b)
/// They accept the R,Phi,Z coordinate (-pi<phi<pi) and return the field
/// integrals in cyl. coordinates.
///
/// Thus, to compute the integral from arbitrary xy_z1 to xy_z2, one should take
/// b = b1-b2 with b1 and b2 coming from getTPCInt(xy_z1,b1) and getTPCInt(xy_z2,b2)
///
/// Note: the integrals are defined for the range -300<Z<300 and 0<R<300
const UShort_t MagneticField::sPolarityConvention = MagneticField::kConvLHC;

MagneticField::MagneticField()
  : FairField(),
    mMeasuredMap(nullptr),
    mFastField(nullptr),
    mMapType(MagFieldParam::k5kG),
    mSolenoid(0),
    mBeamType(MagFieldParam::kNoBeamField),
    mBeamEnergy(0),
    mDefaultIntegration(0),
    mPrecisionInteg(0),
    mMultipicativeFactorSolenoid(1.),
    mMultipicativeFactorDipole(1.),
    mMaxField(15),
    mDipoleOnOffFlag(kFALSE),
    mQuadrupoleGradient(0),
    mDipoleField(0),
    mCompensatorField2C(0),
    mCompensatorField1A(0),
    mCompensatorField2A(0),
    mParameterNames("", "")
{
  /*
   * Default constructor
   */
  fType = 2; // flag non-constant field
}

MagneticField::MagneticField(const char* name, const char* title, Double_t factorSol, Double_t factorDip,
                             MagFieldParam::BMap_t maptype, MagFieldParam::BeamType_t bt, Double_t be, Int_t integ,
                             Double_t fmax, const std::string path)
  : FairField(name, title),
    mMeasuredMap(nullptr),
    mFastField(nullptr),
    mMapType(maptype),
    mSolenoid(0),
    mBeamType(bt),
    mBeamEnergy(be),
    mDefaultIntegration(integ),
    mPrecisionInteg(1),
    mMultipicativeFactorSolenoid(factorSol),
    mMultipicativeFactorDipole(factorDip),
    mMaxField(fmax),
    mDipoleOnOffFlag(factorDip == 0.),
    mQuadrupoleGradient(0),
    mDipoleField(0),
    mCompensatorField2C(0),
    mCompensatorField1A(0),
    mCompensatorField2A(0),
    mParameterNames("", "")
{
  /*
   * Constructor for human readable params
   */

  setDataFileName(path.c_str());
  CreateField();
}

MagneticField::MagneticField(const MagFieldParam& param)
  : FairField(param.GetName(), param.GetTitle()),
    mMeasuredMap(nullptr),
    mFastField(nullptr),
    mMapType(param.GetMapType()),
    mSolenoid(0),
    mBeamType(param.GetBeamType()),
    mBeamEnergy(param.GetBeamEnergy()),
    mDefaultIntegration(param.GetDefInt()),
    mPrecisionInteg(1),
    mMultipicativeFactorSolenoid(param.GetFactorSol()), // temporary
    mMultipicativeFactorDipole(param.GetFactorDip()),   // temporary
    mMaxField(param.GetMaxField()),
    mDipoleOnOffFlag(param.GetFactorDip() == 0.),
    mQuadrupoleGradient(0),
    mDipoleField(0),
    mCompensatorField2C(0),
    mCompensatorField1A(0),
    mCompensatorField2A(0),
    mParameterNames("", "")
{
  /*
   * Constructor for FairParam derived params
   */

  setDataFileName(param.GetMapPath());
  CreateField();
}

void MagneticField::CreateField()
{
  /*
   * field initialization
   */

  fType = 2; // flag non-constant field

  // does real creation of the field
  if (mDefaultIntegration < 0 || mDefaultIntegration > 2) {
    LOG(WARNING) << "MagneticField::CreateField: Invalid magnetic field flag: " << mDefaultIntegration
                 << "; Helix tracking chosen instead";
    mDefaultIntegration = 2;
  }
  if (mDefaultIntegration == 0)
    mPrecisionInteg = 0;

  if (mBeamEnergy <= 0 && mBeamType != MagFieldParam::kNoBeamField) {
    if (mBeamType == MagFieldParam::kBeamTypepp)
      mBeamEnergy = 7000.; // max proton energy
    else if (mBeamType == MagFieldParam::kBeamTypeAA)
      mBeamEnergy = 2760; // max PbPb energy
    else if (mBeamType == MagFieldParam::kBeamTypepA || mBeamType == MagFieldParam::kBeamTypeAp)
      mBeamEnergy = 2760; // same rigitiy max PbPb energy
    //
    LOG(INFO) << "MagneticField::CreateField: Maximim possible beam energy for requested beam is assumed";
  }

  const char* parname = nullptr;

  if (mMapType == MagFieldParam::k2kG) {
    parname = mDipoleOnOffFlag ? "Sol12_Dip0_Hole" : "Sol12_Dip6_Hole";
  } else if (mMapType == MagFieldParam::k5kG) {
    parname = mDipoleOnOffFlag ? "Sol30_Dip0_Hole" : "Sol30_Dip6_Hole";
  } else if (mMapType == MagFieldParam::k5kGUniform) {
    parname = "Sol30_Dip6_Uniform";
  } else {
    LOG(FATAL) << "MagneticField::CreateField: Unknown field identifier " << mMapType << " is requested\n";
  }

  setParameterName(parname);

  loadParameterization();
  initializeMachineField(mBeamType, mBeamEnergy);
  setFactorSolenoid(mMultipicativeFactorSolenoid);
  setFactorDipole(mMultipicativeFactorDipole);
  double xyz[3] = {0., 0., 0.};
  mSolenoid = getBz(xyz);
  Print("a");
  //
}

Bool_t MagneticField::loadParameterization()
{
  /*
   * load parametrization for measured field
   */

  if (mMeasuredMap) {
    LOG(FATAL) << "MagneticField::loadParameterization: Field data " << getParameterName()
               << " are already loaded from " << getDataFileName() << "\n";
  }
  const char* fname = gSystem->ExpandPathName(getDataFileName());
  TFile* file = TFile::Open(fname);
  if (!file) {
    LOG(FATAL) << "MagneticField::loadParameterization: Failed to open magnetic field data file " << fname << "\n";
  }

  mMeasuredMap =
    std::unique_ptr<MagneticWrapperChebyshev>(dynamic_cast<MagneticWrapperChebyshev*>(file->Get(getParameterName())));
  if (!mMeasuredMap) {
    LOG(FATAL) << "MagneticField::loadParameterization: Did not find field " << getParameterName() << " in " << fname
               << "%s\n";
  }
  file->Close();
  delete file;
  return kTRUE;
}

void MagneticField::Field(const Double_t* __restrict__ xyz, Double_t* __restrict__ b)
{
  /*
   * query field value at point
   */

  //  b[0]=b[1]=b[2]=0.0;
  if (mFastField && mFastField->Field(xyz, b))
    return;

  if (mMeasuredMap && xyz[2] > mMeasuredMap->getMinZ() && xyz[2] < mMeasuredMap->getMaxZ()) {
    mMeasuredMap->Field(xyz, b);
    if (xyz[2] > sSolenoidToDipoleZ || mDipoleOnOffFlag) {
      for (int i = 3; i--;) {
        b[i] *= mMultipicativeFactorSolenoid;
      }
    } else {
      for (int i = 3; i--;) {
        b[i] *= mMultipicativeFactorDipole;
      }
    }
  } else {
    MachineField(xyz, b);
  }
}

Double_t MagneticField::getBz(const Double_t* xyz) const
{
  /*
   * query field Bz component at point
   */

  if (mFastField) {
    double bz = 0;
    if (mFastField->GetBz(xyz, bz))
      return bz;
  }
  if (mMeasuredMap && xyz[2] > mMeasuredMap->getMinZ() && xyz[2] < mMeasuredMap->getMaxZ()) {
    double bz = mMeasuredMap->getBz(xyz);
    return (xyz[2] > sSolenoidToDipoleZ || mDipoleOnOffFlag) ? bz * mMultipicativeFactorSolenoid
                                                             : bz * mMultipicativeFactorDipole;
  } else {
    return 0.;
  }
}

MagneticField& MagneticField::operator=(const MagneticField& src)
{
  /*
   * assignment operator
   */

  if (this != &src) {
    if (src.mMeasuredMap) {
      mMeasuredMap.reset(new MagneticWrapperChebyshev(*src.getMeasuredMap()));
    }
    SetName(src.GetName());
    mSolenoid = src.mSolenoid;
    mBeamType = src.mBeamType;
    mBeamEnergy = src.mBeamEnergy;
    mDefaultIntegration = src.mDefaultIntegration;
    mPrecisionInteg = src.mPrecisionInteg;
    mMultipicativeFactorSolenoid = src.mMultipicativeFactorSolenoid;
    mMultipicativeFactorDipole = src.mMultipicativeFactorDipole;
    mMaxField = src.mMaxField;
    mDipoleOnOffFlag = src.mDipoleOnOffFlag;
    mParameterNames = src.mParameterNames;
    mFastField.reset(src.mFastField ? new MagFieldFast(*src.getFastField()) : nullptr);
  }
  return *this;
}

void MagneticField::initializeMachineField(MagFieldParam::BeamType_t btype, Double_t benergy)
{
  if (btype == MagFieldParam::kNoBeamField) {
    mQuadrupoleGradient = mDipoleField = mCompensatorField2C = mCompensatorField1A = mCompensatorField2A = 0.;
    return;
  }

  double rigScale = benergy / 7000.; // scale according to ratio of E/Enominal
  // for ions assume PbPb (with energy provided per nucleon) and account for A/Z
  if (btype == MagFieldParam::kBeamTypeAA /* || btype==kBeamTypepA || btype==kBeamTypeAp */) {
    rigScale *= 208. / 82.;
  }
  // Attention: in p-Pb the energy recorded in the GRP is the PROTON energy, no rigidity
  // rescaling is needed

  mQuadrupoleGradient = 22.0002 * rigScale;
  mDipoleField = 37.8781 * rigScale;

  // SIDE C
  mCompensatorField2C = -9.6980;
  // SIDE A
  mCompensatorField1A = -13.2247;
  mCompensatorField2A = 11.7905;
}

void MagneticField::MachineField(const Double_t* __restrict__ x, Double_t* __restrict__ b) const
{
  // ---- This is the ZDC part
  // Compansators for Alice Muon Arm Dipole
  const Double_t kBComp1CZ = 1075., kBComp1hDZ = 260. / 2., kBComp1SqR = 4.0 * 4.0;
  const Double_t kBComp2CZ = 2049., kBComp2hDZ = 153. / 2., kBComp2SqR = 4.5 * 4.5;

  const Double_t kTripQ1CZ = 2615., kTripQ1hDZ = 637. / 2., kTripQ1SqR = 3.5 * 3.5;
  const Double_t kTripQ2CZ = 3480., kTripQ2hDZ = 550. / 2., kTripQ2SqR = 3.5 * 3.5;
  const Double_t kTripQ3CZ = 4130., kTripQ3hDZ = 550. / 2., kTripQ3SqR = 3.5 * 3.5;
  const Double_t kTripQ4CZ = 5015., kTripQ4hDZ = 637. / 2., kTripQ4SqR = 3.5 * 3.5;

  const Double_t kDip1CZ = 6310.8, kDip1hDZ = 945. / 2., kDip1SqRC = 4.5 * 4.5, kDip1SqRA = 3.375 * 3.375;
  const Double_t kDip2CZ = 12640.3, kDip2hDZ = 945. / 2., kDip2SqRC = 4.5 * 4.5, kDip2SqRA = 3.75 * 3.75;
  const Double_t kDip2DXC = 9.7, kDip2DXA = 9.4;

  double rad2 = x[0] * x[0] + x[1] * x[1];

  b[0] = b[1] = b[2] = 0;

  // SIDE C
  if (x[2] < 0.) {
    if (TMath::Abs(x[2] + kBComp2CZ) < kBComp2hDZ && rad2 < kBComp2SqR) {
      b[0] = mCompensatorField2C * mMultipicativeFactorDipole;
    } else if (TMath::Abs(x[2] + kTripQ1CZ) < kTripQ1hDZ && rad2 < kTripQ1SqR) {
      b[0] = mQuadrupoleGradient * x[1];
      b[1] = mQuadrupoleGradient * x[0];
    } else if (TMath::Abs(x[2] + kTripQ2CZ) < kTripQ2hDZ && rad2 < kTripQ2SqR) {
      b[0] = -mQuadrupoleGradient * x[1];
      b[1] = -mQuadrupoleGradient * x[0];
    } else if (TMath::Abs(x[2] + kTripQ3CZ) < kTripQ3hDZ && rad2 < kTripQ3SqR) {
      b[0] = -mQuadrupoleGradient * x[1];
      b[1] = -mQuadrupoleGradient * x[0];
    } else if (TMath::Abs(x[2] + kTripQ4CZ) < kTripQ4hDZ && rad2 < kTripQ4SqR) {
      b[0] = mQuadrupoleGradient * x[1];
      b[1] = mQuadrupoleGradient * x[0];
    } else if (TMath::Abs(x[2] + kDip1CZ) < kDip1hDZ && rad2 < kDip1SqRC) {
      b[1] = mDipoleField;
    } else if (TMath::Abs(x[2] + kDip2CZ) < kDip2hDZ && rad2 < kDip2SqRC) {
      double dxabs = TMath::Abs(x[0]) - kDip2DXC;
      if ((dxabs * dxabs + x[1] * x[1]) < kDip2SqRC) {
        b[1] = -mDipoleField;
      }
    }
  }

  // SIDE A
  else {
    if (TMath::Abs(x[2] - kBComp1CZ) < kBComp1hDZ && rad2 < kBComp1SqR) {
      // Compensator magnet at z = 1075 m
      b[0] = mCompensatorField1A * mMultipicativeFactorDipole;
    }

    if (TMath::Abs(x[2] - kBComp2CZ) < kBComp2hDZ && rad2 < kBComp2SqR) {
      b[0] = mCompensatorField2A * mMultipicativeFactorDipole;
    } else if (TMath::Abs(x[2] - kTripQ1CZ) < kTripQ1hDZ && rad2 < kTripQ1SqR) {
      b[0] = -mQuadrupoleGradient * x[1];
      b[1] = -mQuadrupoleGradient * x[0];
    } else if (TMath::Abs(x[2] - kTripQ2CZ) < kTripQ2hDZ && rad2 < kTripQ2SqR) {
      b[0] = mQuadrupoleGradient * x[1];
      b[1] = mQuadrupoleGradient * x[0];
    } else if (TMath::Abs(x[2] - kTripQ3CZ) < kTripQ3hDZ && rad2 < kTripQ3SqR) {
      b[0] = mQuadrupoleGradient * x[1];
      b[1] = mQuadrupoleGradient * x[0];
    } else if (TMath::Abs(x[2] - kTripQ4CZ) < kTripQ4hDZ && rad2 < kTripQ4SqR) {
      b[0] = -mQuadrupoleGradient * x[1];
      b[1] = -mQuadrupoleGradient * x[0];
    } else if (TMath::Abs(x[2] - kDip1CZ) < kDip1hDZ && rad2 < kDip1SqRA) {
      b[1] = -mDipoleField;
    } else if (TMath::Abs(x[2] - kDip2CZ) < kDip2hDZ && rad2 < kDip2SqRA) {
      double dxabs = TMath::Abs(x[0]) - kDip2DXA;
      if ((dxabs * dxabs + x[1] * x[1]) < kDip2SqRA) {
        b[1] = mDipoleField;
      }
    }
  }
}

void MagneticField::getTPCIntegral(const Double_t* xyz, Double_t* b) const
{
  b[0] = b[1] = b[2] = 0.0;
  if (mMeasuredMap) {
    mMeasuredMap->getTPCIntegral(xyz, b);
    for (int i = 3; i--;) {
      b[i] *= mMultipicativeFactorSolenoid;
    }
  }
}

void MagneticField::getTPCRatIntegral(const Double_t* xyz, Double_t* b) const
{
  b[0] = b[1] = b[2] = 0.0;
  if (mMeasuredMap) {
    mMeasuredMap->getTPCRatIntegral(xyz, b);
    b[2] /= 100;
  }
}

void MagneticField::getTPCIntegralCylindrical(const Double_t* rphiz, Double_t* b) const
{
  b[0] = b[1] = b[2] = 0.0;
  if (mMeasuredMap) {
    mMeasuredMap->getTPCIntegralCylindrical(rphiz, b);
    for (int i = 3; i--;) {
      b[i] *= mMultipicativeFactorSolenoid;
    }
  }
}

void MagneticField::getTPCRatIntegralCylindrical(const Double_t* rphiz, Double_t* b) const
{
  b[0] = b[1] = b[2] = 0.0;
  if (mMeasuredMap) {
    mMeasuredMap->getTPCRatIntegralCylindrical(rphiz, b);
    b[2] /= 100;
  }
}

void MagneticField::setFactorSolenoid(Float_t fc)
{
  switch (sPolarityConvention) {
    case kConvDCS2008:
      mMultipicativeFactorSolenoid = -fc;
      break;
    case kConvLHC:
      mMultipicativeFactorSolenoid = -fc;
      break;
    default:
      mMultipicativeFactorSolenoid = fc;
      break; // case kConvMap2005: mMultipicativeFactorSolenoid =  fc; break;
  }
  if (mFastField)
    mFastField->setFactorSol(getFactorSolenoid());
}

void MagneticField::setFactorDipole(Float_t fc)
{
  switch (sPolarityConvention) {
    case kConvDCS2008:
      mMultipicativeFactorDipole = fc;
      break;
    case kConvLHC:
      mMultipicativeFactorDipole = -fc;
      break;
    default:
      mMultipicativeFactorDipole = fc;
      break; // case kConvMap2005: mMultipicativeFactorDipole =  fc; break;
  }
}

Double_t MagneticField::getFactorSolenoid() const
{
  switch (sPolarityConvention) {
    case kConvDCS2008:
      return -mMultipicativeFactorSolenoid;
    case kConvLHC:
      return -mMultipicativeFactorSolenoid;
    default:
      return mMultipicativeFactorSolenoid; //  case kConvMap2005: return  mMultipicativeFactorSolenoid;
  }
}

Double_t MagneticField::getFactorDipole() const
{
  switch (sPolarityConvention) {
    case kConvDCS2008:
      return mMultipicativeFactorDipole;
    case kConvLHC:
      return -mMultipicativeFactorDipole;
    default:
      return mMultipicativeFactorDipole; //  case kConvMap2005: return  mMultipicativeFactorDipole;
  }
}

MagneticField* MagneticField::createFieldMap(Float_t l3Cur, Float_t diCur, Int_t convention, Bool_t uniform,
                                             Float_t beamenergy, const Char_t* beamtype, const std::string path)
{
  const Float_t l3NominalCurrent1 = 30000.f; // (A)
  const Float_t l3NominalCurrent2 = 12000.f; // (A)
  const Float_t diNominalCurrent = 6000.f;   // (A)

  const Float_t tolerance = 0.03; // relative current tolerance
  const Float_t zero = 77.f;      // "zero" current (A)

  MagFieldParam::BMap_t map = MagFieldParam::k5kG;
  double sclL3, sclDip;

  Float_t l3Pol = l3Cur > 0 ? 1 : -1;
  Float_t diPol = diCur > 0 ? 1 : -1;

  l3Cur = TMath::Abs(l3Cur);
  diCur = TMath::Abs(diCur);

  if (TMath::Abs((sclDip = diCur / diNominalCurrent) - 1.) > tolerance && !uniform) {
    if (diCur <= zero) {
      sclDip = 0.; // some small current.. -> Dipole OFF
    } else {
      LOG(FATAL) << "MagneticField::createFieldMap: Wrong dipole current (" << diCur << " A)!";
    }
  }

  if (uniform) {
    // special treatment of special MC with uniform mag field (normalized to 0.5 T)
    // no check for scaling/polarities are done
    map = MagFieldParam::k5kGUniform;
    sclL3 = l3Cur / l3NominalCurrent1;
  } else {
    if (TMath::Abs((sclL3 = l3Cur / l3NominalCurrent1) - 1.) < tolerance) {
      map = MagFieldParam::k5kG;
    } else if (TMath::Abs((sclL3 = l3Cur / l3NominalCurrent2) - 1.) < tolerance) {
      map = MagFieldParam::k2kG;
    } else if (l3Cur <= zero && diCur <= zero) {
      sclL3 = 0;
      sclDip = 0;
      map = MagFieldParam::k5kGUniform;
    } else {
      LOG(FATAL) << "MagneticField::createFieldMap: Wrong L3 current (" << l3Cur << "  A)!";
    }
  }

  if (sclDip != 0 && map != MagFieldParam::k5kGUniform) {
    if ((l3Cur <= zero) ||
        ((convention == kConvLHC && l3Pol != diPol) || (convention == kConvDCS2008 && l3Pol == diPol))) {
      LOG(FATAL) << "MagneticField::createFieldMap: Wrong combination for L3/Dipole polarities ("
                 << (l3Pol > 0 ? '+' : '-') << "/" << (diPol > 0 ? '+' : '-') << ") for convention "
                 << getPolarityConvention();
    }
  }

  if (l3Pol < 0) {
    sclL3 = -sclL3;
  }
  if (diPol < 0) {
    sclDip = -sclDip;
  }

  MagFieldParam::BeamType_t btype = MagFieldParam::kNoBeamField;
  TString btypestr = beamtype;
  btypestr.ToLower();
  TPRegexp protonBeam(R"((proton|p)\s*-?\s*\1)");
  TPRegexp ionBeam(R"((lead|pb|ion|a|A)\s*-?\s*\1)");
  TPRegexp protonionBeam(R"((proton|p)\s*-?\s*(lead|pb|ion|a|A))");
  TPRegexp ionprotonBeam(R"((lead|pb|ion|a|A)\s*-?\s*(proton|p))");
  if (btypestr.Contains(ionBeam)) {
    btype = MagFieldParam::kBeamTypeAA;
  } else if (btypestr.Contains(protonBeam)) {
    btype = MagFieldParam::kBeamTypepp;
  } else if (btypestr.Contains(protonionBeam)) {
    btype = MagFieldParam::kBeamTypepA;
  } else if (btypestr.Contains(ionprotonBeam)) {
    btype = MagFieldParam::kBeamTypeAp;
  } else {
    LOG(INFO) << "Assume no LHC magnet field for the beam type " << beamtype;
  }
  char ttl[80];
  snprintf(ttl, 79, "L3: %+5d Dip: %+4d kA; %s | Polarities in %s convention", (int)TMath::Sign(l3Cur, float(sclL3)),
           (int)TMath::Sign(diCur, float(sclDip)), uniform ? " Constant" : "",
           convention == kConvLHC ? "LHC" : "DCS2008");
  // LHC and DCS08 conventions have opposite dipole polarities
  if (getPolarityConvention() != convention) {
    sclDip = -sclDip;
  }

  return new MagneticField("MagneticFieldMap", ttl, sclL3, sclDip, map, btype, beamenergy, 2, 10., path);
}

const char* MagneticField::getBeamTypeText() const
{
  const char* beamNA = "No Beam";
  const char* beamPP = "p-p";
  const char* beamPbPb = "A-A";
  const char* beamPPb = "p-A";
  const char* beamPbP = "A-p";
  switch (mBeamType) {
    case MagFieldParam::kBeamTypepp:
      return beamPP;
    case MagFieldParam::kBeamTypeAA:
      return beamPbPb;
    case MagFieldParam::kBeamTypepA:
      return beamPPb;
    case MagFieldParam::kBeamTypeAp:
      return beamPbP;
    case MagFieldParam::kNoBeamField:
    default:
      return beamNA;
  }
}

void MagneticField::Print(Option_t* opt) const
{
  TString opts = opt;
  opts.ToLower();
  LOG(INFO) << "MagneticField::Print: " << GetName() << ":" << GetTitle();
  LOG(INFO) << "MagneticField::Print: Solenoid (" << getFactorSolenoid() << "*)"
            << ((mMapType == MagFieldParam::k5kG || mMapType == MagFieldParam::k5kGUniform) ? 5. : 2) << " kG, Dipole "
            << (mDipoleOnOffFlag ? "OFF" : "ON") << " (" << getFactorDipole() << ") "
            << (mMapType == MagFieldParam::k5kGUniform ? " |Constant Field!" : "");
  if (opts.Contains("a")) {
    LOG(INFO) << "MagneticField::Print: Machine B fields for " << getBeamTypeText() << "  beam (" << mBeamEnergy
              << " GeV): QGrad: " << mQuadrupoleGradient << " Dipole: " << mDipoleField;
    LOG(INFO) << "MagneticField::Print: Uses " << getParameterName() << "  of " << getDataFileName();
  }
}

void MagneticField::FillParContainer()
{
  // fill field parameters
  FairRun* fRun = FairRun::Instance();
  FairRuntimeDb* rtdb = fRun->GetRuntimeDb();
  MagFieldParam* par = static_cast<MagFieldParam*>(rtdb->getContainer("MagFieldParam"));
  par->SetParam(this);
  par->setChanged();
}

//_____________________________________________________________________________
void MagneticField::AllowFastField(bool v)
{
  if (v) {
    if (!mFastField)
      mFastField = std::make_unique<MagFieldFast>(getFactorSolenoid(), mMapType == MagFieldParam::k2kG ? 2 : 5);
  } else {
    mFastField.reset(nullptr);
  }
}

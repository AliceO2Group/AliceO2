// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cassert>

#include "EMCALBase/EMCGeometry.h"
#include "EMCALBase/ShishKebabTrd1Module.h"

ClassImp(o2::EMCAL::ShishKebabTrd1Module);

using namespace o2::EMCAL;

/*
//AliEMCALEMCGeometry *AliEMCALShishKebabTrd1Module::fgGeometry=0;
Double_t AliEMCALShishKebabTrd1Module::fga=0.;
Double_t AliEMCALShishKebabTrd1Module::fga2=0.;
Double_t AliEMCALShishKebabTrd1Module::fgb=0.;
Double_t AliEMCALShishKebabTrd1Module::fgr=0.;
Double_t AliEMCALShishKebabTrd1Module::fgangle=0.;   // around one degree
Double_t AliEMCALShishKebabTrd1Module::fgtanBetta=0; //
*/

ShishKebabTrd1Module::ShishKebabTrd1Module(Double_t theta, EMCGeometry* g)
  : TNamed(),
    mGeometry(g),
    mOK(),
    mA(0.),
    mB(0.),
    mThetaA(0.),
    mTheta(theta),
    mOK1(),
    mOK2(),
    mOB(),
    mOB1(),
    mOB2(),
    mThetaOB1(0.),
    mThetaOB2(0.),
    mOK3X3(),
    mORB(),
    mORT()
{
  TString snam(g->GetName());
  Int_t key = 0;
  if (snam.Contains("v1", TString::kIgnoreCase))
    key = 1; // EMCAL_COMPLETEV1 vs EMCAL_COMPLETEv1 (or other)

  if (GetParameters())
    DefineFirstModule(key);

  DefineName(mTheta);
  LOG(DEBUG4) << "o2::EMCAL::ShishKebabTrd1Module - first module key=" << key << ":  theta " << std::setw(1)
              << std::setprecision(4) << mTheta << " geometry " << g << FairLogger::endl;
}

ShishKebabTrd1Module::ShishKebabTrd1Module(ShishKebabTrd1Module& leftNeighbor)
  : TNamed(),
    mGeometry(leftNeighbor.mGeometry),
    mOK(),
    mA(0.),
    mB(0.),
    mThetaA(0.),
    mTheta(0.),
    mOK1(),
    mOK2(),
    mOB(),
    mOB1(),
    mOB2(),
    mThetaOB1(0.),
    mThetaOB2(0.),
    mOK3X3(),
    mORB(),
    mORT()
{
  //  printf("** Left Neighbor : %s **\n", leftNeighbor.GetName());
  mTheta = leftNeighbor.GetTheta() - mgangle;

  TObject::SetUniqueID(leftNeighbor.GetUniqueID() + 1);

  Init(leftNeighbor.GetA(), leftNeighbor.GetB());
}

ShishKebabTrd1Module::ShishKebabTrd1Module(const ShishKebabTrd1Module& mod)
  : TNamed(mod.GetName(), mod.GetTitle()),
    mGeometry(mod.mGeometry),
    mOK(mod.mOK),
    mA(mod.mA),
    mB(mod.mB),
    mThetaA(mod.mThetaA),
    mTheta(mod.mTheta),
    mOK1(mod.mOK1),
    mOK2(mod.mOK2),
    mOB(mod.mOB),
    mOB1(mod.mOB1),
    mOB2(mod.mOB2),
    mThetaOB1(mod.mThetaOB1),
    mThetaOB2(mod.mThetaOB2),
    mORB(mod.mORB),
    mORT(mod.mORT)
{
  for (Int_t i = 0; i < 3; i++)
    mOK3X3[i] = mod.mOK3X3[i];
}

void ShishKebabTrd1Module::Init(Double_t A, Double_t B)
{
  // Define parameter module from parameters A,B from previous.
  Double_t yl = (mgb / 2) * TMath::Sin(mTheta) + (mga / 2) * TMath::Cos(mTheta) + mgr, y = yl;
  Double_t xl = (yl - B) / A; // y=A*x+B

  //  Double_t xp1 = (fga/2. + fgb/2.*fgtanBetta)/(TMath::Sin(fTheta) + fgtanBetta*TMath::Cos(fTheta));
  //  printf(" xp1 %9.3f \n ", xp1);
  // xp1 == xp => both methods give the same results - 3-feb-05
  Double_t alpha = TMath::Pi() / 2. + mgangle / 2;
  Double_t xt =
    (mga + mga2) * TMath::Tan(mTheta) * TMath::Tan(alpha) / (4. * (1. - TMath::Tan(mTheta) * TMath::Tan(alpha)));
  Double_t yt = xt / TMath::Tan(mTheta), xp = TMath::Sqrt(xt * xt + yt * yt);
  Double_t x = xl + xp;
  mOK.Set(x, y);
  //  printf(" yl %9.3f | xl %9.3f | xp %9.3f \n", yl, xl, xp);

  // have to define A and B;
  Double_t yCprev = mgr + mga * TMath::Cos(mTheta);
  Double_t xCprev = (yCprev - B) / A;
  Double_t xA = xCprev + mga * TMath::Sin(mTheta), yA = mgr;

  mThetaA = mTheta - mgangle / 2.;
  mA = TMath::Tan(mThetaA); // !!
  mB = yA - mA * xA;

  DefineAllStuff();
}

void ShishKebabTrd1Module::DefineAllStuff()
{
  // Define some parameters
  DefineName(mTheta);
  // Centers of cells - 2X2 case
  Double_t kk1 = (mga + mga2) / (2. * 4.); // kk1=kk2

  Double_t xk1 = mOK.X() - kk1 * TMath::Sin(mTheta);
  Double_t yk1 = mOK.Y() + kk1 * TMath::Cos(mTheta) - mgr;
  mOK1.Set(xk1, yk1);

  Double_t xk2 = mOK.X() + kk1 * TMath::Sin(mTheta);
  Double_t yk2 = mOK.Y() - kk1 * TMath::Cos(mTheta) - mgr;
  mOK2.Set(xk2, yk2);

  // Centers of cells - 3X3 case; Nov 9,2006
  mOK3X3[1].Set(mOK.X(), mOK.Y() - mgr); // coincide with module center

  kk1 = ((mga + mga2) / 4. + mga / 6.) / 2.;

  xk1 = mOK.X() - kk1 * TMath::Sin(mTheta);
  yk1 = mOK.Y() + kk1 * TMath::Cos(mTheta) - mgr;
  mOK3X3[0].Set(xk1, yk1);

  xk2 = mOK.X() + kk1 * TMath::Sin(mTheta);
  yk2 = mOK.Y() - kk1 * TMath::Cos(mTheta) - mgr;
  mOK3X3[2].Set(xk2, yk2);

  // May 15, 2006; position of module(cells) center face
  mOB.Set(mOK.X() - mgb / 2. * TMath::Cos(mTheta), mOK.Y() - mgb / 2. * TMath::Sin(mTheta) - mgr);
  mOB1.Set(mOB.X() - mga / 4. * TMath::Sin(mTheta), mOB.Y() + mga / 4. * TMath::Cos(mTheta));
  mOB2.Set(mOB.X() + mga / 4. * TMath::Sin(mTheta), mOB.Y() - mga / 4. * TMath::Cos(mTheta));
  // Jul 30, 2007 - for taking into account a position of shower maximum
  mThetaOB1 = mTheta - mgangle / 4.; // ??
  mThetaOB2 = mTheta + mgangle / 4.;

  // Position of right/top point of module
  // Gives the posibility to estimate SM size in z direction
  Double_t xBottom = (mgr - mB) / mA;
  Double_t yBottom = mgr;
  mORB.Set(xBottom, yBottom);

  Double_t l = mgb / TMath::Cos(mgangle / 2.); // length of lateral module side
  Double_t xTop = xBottom + l * TMath::Cos(TMath::ATan(mA));
  Double_t yTop = mA * xTop + mB;
  mORT.Set(xTop, yTop);
}

void ShishKebabTrd1Module::DefineFirstModule(const Int_t key)
{
  // Define first module
  if (key == 0) {
    // theta in radians ; first object theta=pi/2.
    mTheta = TMath::PiOver2();
    mOK.Set(mga2 / 2., mgr + mgb / 2.); // position the center of module vs o

    // parameters of right line : y = A*z + B in system where zero point is IP.
    mThetaA = mTheta - mgangle / 2.;
    mA = TMath::Tan(mThetaA);
    Double_t xA = mga / 2. + mga2 / 2.;
    Double_t yA = mgr;
    mB = yA - mA * xA;
  } else if (key == 1) {
    // theta in radians ; first object theta = 90-0.75 = 89.25 degree
    mTheta = 89.25 * TMath::DegToRad();
    Double_t al1 = mgangle / 2.;
    Double_t x = 0.5 * (mga * TMath::Cos(al1) + mgb * TMath::Sin(al1));
    Double_t y = 0.5 * (mgb + mga * TMath::Sin(al1)) * TMath::Cos(al1);
    mOK.Set(x, mgr + y);
    // parameters of right line : y = A*z + B in system where zero point is IP.
    mThetaA = mTheta - mgangle / 2.;
    mA = TMath::Tan(mThetaA);
    Double_t xA = mga * TMath::Cos(al1);
    Double_t yA = mgr;
    mB = yA - mA * xA;
  } else {
    LOG(ERROR) << "key=" << key << " : wrong case \n";
    assert(0);
  }

  TObject::SetUniqueID(1); //

  DefineAllStuff();
}

void ShishKebabTrd1Module::DefineName(Double_t theta)
{
  SetName(Form("%2i(%5.2f)", TObject::GetUniqueID(), theta * TMath::RadToDeg()));
}

Bool_t ShishKebabTrd1Module::GetParameters()
{
  if (!mGeometry) {
    LOG(WARNING) << "GetParameters(): << No geometry\n";
    return kFALSE;
  }

  TString sn(mGeometry->GetName()); // 2-Feb-05
  sn.ToUpper();

  mga = (Double_t)mGeometry->GetEtaModuleSize();
  mgb = (Double_t)mGeometry->GetLongModuleSize();
  mgangle = Double_t(mGeometry->GetTrd1Angle()) * TMath::DegToRad();
  mgtanBetta = TMath::Tan(mgangle / 2.);
  mgr = (Double_t)mGeometry->GetIPDistance();

  mgr += mGeometry->GetSteelFrontThickness();

  mga2 = Double_t(mGeometry->Get2Trd1Dx2());
  // PH  PrintShish(0);
  return kTRUE;
}

//
// Service methods
//

/// Add comment
//_____________________________________________________________________________
void ShishKebabTrd1Module::PrintShish(int pri) const
{
  if (pri >= 0) {
    if (pri >= 1) {
      printf("PrintShish() \n a %7.3f:%7.3f | b %7.2f | r %7.2f \n TRD1 angle %7.6f(%5.2f) | tanBetta %7.6f", mga, mga2,
             mgb, mgr, mgangle, mgangle * TMath::RadToDeg(), mgtanBetta);
      printf(" fTheta %f : %5.2f : cos(theta) %f\n", mTheta, GetThetaInDegree(), TMath::Cos(mTheta));
      printf(" OK : %i |%s| theta %f :  phi = %f(%5.2f) \n", GetUniqueID(), GetName(), mTheta, mOK.Phi(),
             mOK.Phi() * TMath::RadToDeg());
    }

    printf(" y %9.3f x %9.3f xrb %9.3f (right bottom on r=%9.3f ) \n", mOK.X(), mOK.Y(), mORB.X(), mORB.Y());

    if (pri >= 2) {
      printf(" A %f B %f | fThetaA %7.6f(%5.2f)\n", mA, mB, mThetaA, mThetaA * TMath::RadToDeg());
      printf(" fOK  : X %9.4f: Y %9.4f : eta  %5.3f\n", mOK.X(), mOK.Y(), GetEtaOfCenterOfModule());
      printf(" fOK1 : X %9.4f: Y %9.4f :   (local, ieta=2)\n", mOK1.X(), mOK1.Y());
      printf(" fOK2 : X %9.4f: Y %9.4f :   (local, ieta=1)\n\n", mOK2.X(), mOK2.Y());
      printf(" fOB  : X %9.4f: Y %9.4f \n", mOB.X(), mOB.Y());
      printf(" fOB1 : X %9.4f: Y %9.4f (local, ieta=2)\n", mOB1.X(), mOB1.Y());
      printf(" fOB2 : X %9.4f: Y %9.4f (local, ieta=1)\n", mOB2.X(), mOB2.Y());
      // 3X3
      printf(" 3X3 \n");
      for (int ieta = 0; ieta < 3; ieta++) {
        printf(" fOK3X3[%i] : X %9.4f: Y %9.4f (local) \n", ieta, mOK3X3[ieta].X(), mOK3X3[ieta].Y());
      }
      //      fOK.Dump();
      GetMaxEtaOfModule();
    }
  }
}

Double_t ShishKebabTrd1Module::GetThetaInDegree() const { return mTheta * TMath::RadToDeg(); }

Double_t ShishKebabTrd1Module::GetEtaOfCenterOfModule() const { return -TMath::Log(TMath::Tan(mOK.Phi() / 2.)); }

void ShishKebabTrd1Module::GetPositionAtCenterCellLine(Int_t ieta, Double_t dist, TVector2& v)
{
  // Jul 30, 2007
  static Double_t theta = 0., x = 0., y = 0.;
  if (ieta == 0) {
    v = mOB2;
    theta = mTheta;
  } else if (ieta == 1) {
    v = mOB1;
    theta = mTheta;
  } else {
    assert(0);
  }

  x = v.X() + TMath::Cos(theta) * dist;
  y = v.Y() + TMath::Sin(theta) * dist;
  //  printf(" GetPositionAtCenterCellLine() %s : dist %f : ieta %i : x %f %f v.X() | y %f %f v.Y() : cos %f sin %f \n",
  // GetName(), dist, ieta, v.X(),x, y,v.Y(),TMath::Cos(theta),TMath::Sin(theta));
  v.Set(x, y);
}

Double_t ShishKebabTrd1Module::GetMaxEtaOfModule() const
{
  // Right bottom point of module
  Double_t thetaBottom = TMath::ATan2(mORB.Y(), mORB.X());
  Double_t etaBottom = ThetaToEta(thetaBottom);

  // Right top point of module
  Double_t thetaTop = TMath::ATan2(mORT.Y(), mORT.X());
  Double_t etaTop = ThetaToEta(thetaTop);

  LOG(DEBUG) << " Right bottom point of module : eta " << std::setw(5) << std::setprecision(4) << etaBottom
             << " : theta " << std::setw(6) << std::setprecision(4) << thetaBottom << " (" << std::setw(6)
             << std::setprecision(2) << thetaBottom * TMath::RadToDeg() << " ) : x(zglob) " << std::setw(7)
             << std::setprecision(2) << mORB.X() << " y(phi) " << std::setw(5) << std::setprecision(2) << mORB.Y()
             << FairLogger::endl;
  LOG(DEBUG) << " Right    top point of module : eta " << std::setw(5) << std::setprecision(4) << etaTop << ": theta "
             << std::setw(6) << std::setprecision(4) << thetaTop << " (" << std::setw(6) << std::setprecision(2)
             << thetaTop * TMath::RadToDeg() << ") : x(zglob) " << std::setw(7) << std::setprecision(2) << mORT.X()
             << "  y(phi) " << std::setw(5) << std::setprecision(2) << mORT.Y() << FairLogger::endl;
  return etaBottom > etaTop ? etaBottom : etaTop;
}

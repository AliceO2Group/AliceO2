/// \file MagWrapCheb.cxx
/// \brief Implementation of the MagWrapCheb class
/// \author ruben.shahoyan@cern.ch 20/03/2007

#include "Field/MagneticWrapperChebyshev.h"
#include <TArrayF.h>     // for TArrayF
#include <TArrayI.h>     // for TArrayI
#include <TSystem.h>     // for TSystem, gSystem
#include <cstdio>       // for printf, fprintf, fclose, fopen, FILE
#include <cstring>      // for memcpy
#include "FairLogger.h"  // for FairLogger, MESSAGE_ORIGIN
#include "TMath.h"       // for BinarySearch, Sort
#include "TMathBase.h"   // for Abs
#include "TNamed.h"      // for TNamed
#include "TObjArray.h"   // for TObjArray
#include "TString.h"     // for TString

using namespace o2::field;
using namespace o2::mathUtils;

ClassImp(MagneticWrapperChebyshev)

MagneticWrapperChebyshev::MagneticWrapperChebyshev()
  : mNumberOfParameterizationSolenoid(0),
    mNumberOfDistinctZSegmentsSolenoid(0),
    mNumberOfDistinctPSegmentsSolenoid(0),
    mNumberOfDistinctRSegmentsSolenoid(0),
    mCoordinatesSegmentsZSolenoid(nullptr),
    mCoordinatesSegmentsPSolenoid(nullptr),
    mCoordinatesSegmentsRSolenoid(nullptr),
    mBeginningOfSegmentsPSolenoid(nullptr),
    mNumberOfSegmentsPSolenoid(nullptr),
    mBeginningOfSegmentsRSolenoid(nullptr),
    mNumberOfRSegmentsSolenoid(nullptr),
    mSegmentIdSolenoid(nullptr),
    mMinZSolenoid(1.e6),
    mMaxZSolenoid(-1.e6),
    mParameterizationSolenoid(nullptr),
    mMaxRadiusSolenoid(0),
    mNumberOfParameterizationTPC(0),
    mNumberOfDistinctZSegmentsTPC(0),
    mNumberOfDistinctPSegmentsTPC(0),
    mNumberOfDistinctRSegmentsTPC(0),
    mCoordinatesSegmentsZTPC(nullptr),
    mCoordinatesSegmentsPTPC(nullptr),
    mCoordinatesSegmentsRTPC(nullptr),
    mBeginningOfSegmentsPTPC(nullptr),
    mNumberOfSegmentsPTPC(nullptr),
    mBeginningOfSegmentsRTPC(nullptr),
    mNumberOfRSegmentsTPC(nullptr),
    mSegmentIdTPC(nullptr),
    mMinZTPC(1.e6),
    mMaxZTPC(-1.e6),
    mParameterizationTPC(nullptr),
    mMaxRadiusTPC(0),
    mNumberOfParameterizationTPCRat(0),
    mNumberOfDistinctZSegmentsTPCRat(0),
    mNumberOfDistinctPSegmentsTPCRat(0),
    mNumberOfDistinctRSegmentsTPCRat(0),
    mCoordinatesSegmentsZTPCRat(nullptr),
    mCoordinatesSegmentsPTPCRat(nullptr),
    mCoordinatesSegmentsRTPCRat(nullptr),
    mBeginningOfSegmentsPTPCRat(nullptr),
    mNumberOfSegmentsPTPCRat(nullptr),
    mBeginningOfSegmentsRTPCRat(nullptr),
    mNumberOfRSegmentsTPCRat(nullptr),
    mSegmentIdTPCRat(nullptr),
    mMinZTPCRat(1.e6),
    mMaxZTPCRat(-1.e6),
    mParameterizationTPCRat(nullptr),
    mMaxRadiusTPCRat(0),
    mNumberOfParameterizationDipole(0),
    mNumberOfDistinctZSegmentsDipole(0),
    mNumberOfDistinctYSegmentsDipole(0),
    mNumberOfDistinctXSegmentsDipole(0),
    mCoordinatesSegmentsZDipole(nullptr),
    mCoordinatesSegmentsYDipole(nullptr),
    mCoordinatesSegmentsXDipole(nullptr),
    mBeginningOfSegmentsYDipole(nullptr),
    mNumberOfSegmentsYDipole(nullptr),
    mBeginningOfSegmentsXDipole(nullptr),
    mNumberOfSegmentsXDipole(nullptr),
    mSegmentIdDipole(nullptr),
    mMinDipoleZ(1.e6),
    mMaxDipoleZ(-1.e6),
    mParameterizationDipole(nullptr),
    mLogger(FairLogger::GetLogger())
{
}

MagneticWrapperChebyshev::MagneticWrapperChebyshev(const MagneticWrapperChebyshev &src)
  : TNamed(src),
    mNumberOfParameterizationSolenoid(0),
    mNumberOfDistinctZSegmentsSolenoid(0),
    mNumberOfDistinctPSegmentsSolenoid(0),
    mNumberOfDistinctRSegmentsSolenoid(0),
    mCoordinatesSegmentsZSolenoid(nullptr),
    mCoordinatesSegmentsPSolenoid(nullptr),
    mCoordinatesSegmentsRSolenoid(nullptr),
    mBeginningOfSegmentsPSolenoid(nullptr),
    mNumberOfSegmentsPSolenoid(nullptr),
    mBeginningOfSegmentsRSolenoid(nullptr),
    mNumberOfRSegmentsSolenoid(nullptr),
    mSegmentIdSolenoid(nullptr),
    mMinZSolenoid(1.e6),
    mMaxZSolenoid(-1.e6),
    mParameterizationSolenoid(nullptr),
    mMaxRadiusSolenoid(0),
    mNumberOfParameterizationTPC(0),
    mNumberOfDistinctZSegmentsTPC(0),
    mNumberOfDistinctPSegmentsTPC(0),
    mNumberOfDistinctRSegmentsTPC(0),
    mCoordinatesSegmentsZTPC(nullptr),
    mCoordinatesSegmentsPTPC(nullptr),
    mCoordinatesSegmentsRTPC(nullptr),
    mBeginningOfSegmentsPTPC(nullptr),
    mNumberOfSegmentsPTPC(nullptr),
    mBeginningOfSegmentsRTPC(nullptr),
    mNumberOfRSegmentsTPC(nullptr),
    mSegmentIdTPC(nullptr),
    mMinZTPC(1.e6),
    mMaxZTPC(-1.e6),
    mParameterizationTPC(nullptr),
    mMaxRadiusTPC(0),
    mNumberOfParameterizationTPCRat(0),
    mNumberOfDistinctZSegmentsTPCRat(0),
    mNumberOfDistinctPSegmentsTPCRat(0),
    mNumberOfDistinctRSegmentsTPCRat(0),
    mCoordinatesSegmentsZTPCRat(nullptr),
    mCoordinatesSegmentsPTPCRat(nullptr),
    mCoordinatesSegmentsRTPCRat(nullptr),
    mBeginningOfSegmentsPTPCRat(nullptr),
    mNumberOfSegmentsPTPCRat(nullptr),
    mBeginningOfSegmentsRTPCRat(nullptr),
    mNumberOfRSegmentsTPCRat(nullptr),
    mSegmentIdTPCRat(nullptr),
    mMinZTPCRat(1.e6),
    mMaxZTPCRat(-1.e6),
    mParameterizationTPCRat(nullptr),
    mMaxRadiusTPCRat(0),
    mNumberOfParameterizationDipole(0),
    mNumberOfDistinctZSegmentsDipole(0),
    mNumberOfDistinctYSegmentsDipole(0),
    mNumberOfDistinctXSegmentsDipole(0),
    mCoordinatesSegmentsZDipole(nullptr),
    mCoordinatesSegmentsYDipole(nullptr),
    mCoordinatesSegmentsXDipole(nullptr),
    mBeginningOfSegmentsYDipole(nullptr),
    mNumberOfSegmentsYDipole(nullptr),
    mBeginningOfSegmentsXDipole(nullptr),
    mNumberOfSegmentsXDipole(nullptr),
    mSegmentIdDipole(nullptr),
    mMinDipoleZ(1.e6),
    mMaxDipoleZ(-1.e6),
    mParameterizationDipole(nullptr),
    mLogger(FairLogger::GetLogger())
{
  copyFrom(src);
}

void MagneticWrapperChebyshev::copyFrom(const MagneticWrapperChebyshev &src)
{
  Clear();
  SetName(src.GetName());
  SetTitle(src.GetTitle());

  mNumberOfParameterizationSolenoid = src.mNumberOfParameterizationSolenoid;
  mNumberOfDistinctZSegmentsSolenoid = src.mNumberOfDistinctZSegmentsSolenoid;
  mNumberOfDistinctPSegmentsSolenoid = src.mNumberOfDistinctPSegmentsSolenoid;
  mNumberOfDistinctRSegmentsSolenoid = src.mNumberOfDistinctRSegmentsSolenoid;
  mMinZSolenoid = src.mMinZSolenoid;
  mMaxZSolenoid = src.mMaxZSolenoid;
  mMaxRadiusSolenoid = src.mMaxRadiusSolenoid;
  if (src.mNumberOfParameterizationSolenoid) {
    memcpy(mCoordinatesSegmentsZSolenoid = new Float_t[mNumberOfDistinctZSegmentsSolenoid],
           src.mCoordinatesSegmentsZSolenoid, sizeof(Float_t) * mNumberOfDistinctZSegmentsSolenoid);
    memcpy(mCoordinatesSegmentsPSolenoid = new Float_t[mNumberOfDistinctPSegmentsSolenoid],
           src.mCoordinatesSegmentsPSolenoid, sizeof(Float_t) * mNumberOfDistinctPSegmentsSolenoid);
    memcpy(mCoordinatesSegmentsRSolenoid = new Float_t[mNumberOfDistinctRSegmentsSolenoid],
           src.mCoordinatesSegmentsRSolenoid, sizeof(Float_t) * mNumberOfDistinctRSegmentsSolenoid);
    memcpy(mBeginningOfSegmentsPSolenoid = new Int_t[mNumberOfDistinctZSegmentsSolenoid],
           src.mBeginningOfSegmentsPSolenoid, sizeof(Int_t) * mNumberOfDistinctZSegmentsSolenoid);
    memcpy(mNumberOfSegmentsPSolenoid = new Int_t[mNumberOfDistinctZSegmentsSolenoid], src.mNumberOfSegmentsPSolenoid,
           sizeof(Int_t) * mNumberOfDistinctZSegmentsSolenoid);
    memcpy(mBeginningOfSegmentsRSolenoid = new Int_t[mNumberOfDistinctPSegmentsSolenoid],
           src.mBeginningOfSegmentsRSolenoid, sizeof(Int_t) * mNumberOfDistinctPSegmentsSolenoid);
    memcpy(mNumberOfRSegmentsSolenoid = new Int_t[mNumberOfDistinctPSegmentsSolenoid], src.mNumberOfRSegmentsSolenoid,
           sizeof(Int_t) * mNumberOfDistinctPSegmentsSolenoid);
    memcpy(mSegmentIdSolenoid = new Int_t[mNumberOfDistinctRSegmentsSolenoid], src.mSegmentIdSolenoid,
           sizeof(Int_t) * mNumberOfDistinctRSegmentsSolenoid);
    mParameterizationSolenoid = new TObjArray(mNumberOfParameterizationSolenoid);
    for (int i = 0; i < mNumberOfParameterizationSolenoid; i++) {
      mParameterizationSolenoid->AddAtAndExpand(new Chebyshev3D(*src.getParameterSolenoid(i)), i);
    }
  }

  mNumberOfParameterizationTPC = src.mNumberOfParameterizationTPC;
  mNumberOfDistinctZSegmentsTPC = src.mNumberOfDistinctZSegmentsTPC;
  mNumberOfDistinctPSegmentsTPC = src.mNumberOfDistinctPSegmentsTPC;
  mNumberOfDistinctRSegmentsTPC = src.mNumberOfDistinctRSegmentsTPC;
  mMinZTPC = src.mMinZTPC;
  mMaxZTPC = src.mMaxZTPC;
  mMaxRadiusTPC = src.mMaxRadiusTPC;
  if (src.mNumberOfParameterizationTPC) {
    memcpy(mCoordinatesSegmentsZTPC = new Float_t[mNumberOfDistinctZSegmentsTPC], src.mCoordinatesSegmentsZTPC,
           sizeof(Float_t) * mNumberOfDistinctZSegmentsTPC);
    memcpy(mCoordinatesSegmentsPTPC = new Float_t[mNumberOfDistinctPSegmentsTPC], src.mCoordinatesSegmentsPTPC,
           sizeof(Float_t) * mNumberOfDistinctPSegmentsTPC);
    memcpy(mCoordinatesSegmentsRTPC = new Float_t[mNumberOfDistinctRSegmentsTPC], src.mCoordinatesSegmentsRTPC,
           sizeof(Float_t) * mNumberOfDistinctRSegmentsTPC);
    memcpy(mBeginningOfSegmentsPTPC = new Int_t[mNumberOfDistinctZSegmentsTPC], src.mBeginningOfSegmentsPTPC,
           sizeof(Int_t) * mNumberOfDistinctZSegmentsTPC);
    memcpy(mNumberOfSegmentsPTPC = new Int_t[mNumberOfDistinctZSegmentsTPC], src.mNumberOfSegmentsPTPC,
           sizeof(Int_t) * mNumberOfDistinctZSegmentsTPC);
    memcpy(mBeginningOfSegmentsRTPC = new Int_t[mNumberOfDistinctPSegmentsTPC], src.mBeginningOfSegmentsRTPC,
           sizeof(Int_t) * mNumberOfDistinctPSegmentsTPC);
    memcpy(mNumberOfRSegmentsTPC = new Int_t[mNumberOfDistinctPSegmentsTPC], src.mNumberOfRSegmentsTPC,
           sizeof(Int_t) * mNumberOfDistinctPSegmentsTPC);
    memcpy(mSegmentIdTPC = new Int_t[mNumberOfDistinctRSegmentsTPC], src.mSegmentIdTPC,
           sizeof(Int_t) * mNumberOfDistinctRSegmentsTPC);
    mParameterizationTPC = new TObjArray(mNumberOfParameterizationTPC);
    for (int i = 0; i < mNumberOfParameterizationTPC; i++) {
      mParameterizationTPC->AddAtAndExpand(new Chebyshev3D(*src.getParameterTPCIntegral(i)), i);
    }
  }

  mNumberOfParameterizationTPCRat = src.mNumberOfParameterizationTPCRat;
  mNumberOfDistinctZSegmentsTPCRat = src.mNumberOfDistinctZSegmentsTPCRat;
  mNumberOfDistinctPSegmentsTPCRat = src.mNumberOfDistinctPSegmentsTPCRat;
  mNumberOfDistinctRSegmentsTPCRat = src.mNumberOfDistinctRSegmentsTPCRat;
  mMinZTPCRat = src.mMinZTPCRat;
  mMaxZTPCRat = src.mMaxZTPCRat;
  mMaxRadiusTPCRat = src.mMaxRadiusTPCRat;
  if (src.mNumberOfParameterizationTPCRat) {
    memcpy(mCoordinatesSegmentsZTPCRat = new Float_t[mNumberOfDistinctZSegmentsTPCRat], src.mCoordinatesSegmentsZTPCRat,
           sizeof(Float_t) * mNumberOfDistinctZSegmentsTPCRat);
    memcpy(mCoordinatesSegmentsPTPCRat = new Float_t[mNumberOfDistinctPSegmentsTPCRat], src.mCoordinatesSegmentsPTPCRat,
           sizeof(Float_t) * mNumberOfDistinctPSegmentsTPCRat);
    memcpy(mCoordinatesSegmentsRTPCRat = new Float_t[mNumberOfDistinctRSegmentsTPCRat], src.mCoordinatesSegmentsRTPCRat,
           sizeof(Float_t) * mNumberOfDistinctRSegmentsTPCRat);
    memcpy(mBeginningOfSegmentsPTPCRat = new Int_t[mNumberOfDistinctZSegmentsTPCRat], src.mBeginningOfSegmentsPTPCRat,
           sizeof(Int_t) * mNumberOfDistinctZSegmentsTPCRat);
    memcpy(mNumberOfSegmentsPTPCRat = new Int_t[mNumberOfDistinctZSegmentsTPCRat], src.mNumberOfSegmentsPTPCRat,
           sizeof(Int_t) * mNumberOfDistinctZSegmentsTPCRat);
    memcpy(mBeginningOfSegmentsRTPCRat = new Int_t[mNumberOfDistinctPSegmentsTPCRat], src.mBeginningOfSegmentsRTPCRat,
           sizeof(Int_t) * mNumberOfDistinctPSegmentsTPCRat);
    memcpy(mNumberOfRSegmentsTPCRat = new Int_t[mNumberOfDistinctPSegmentsTPCRat], src.mNumberOfRSegmentsTPCRat,
           sizeof(Int_t) * mNumberOfDistinctPSegmentsTPCRat);
    memcpy(mSegmentIdTPCRat = new Int_t[mNumberOfDistinctRSegmentsTPCRat], src.mSegmentIdTPCRat,
           sizeof(Int_t) * mNumberOfDistinctRSegmentsTPCRat);
    mParameterizationTPCRat = new TObjArray(mNumberOfParameterizationTPCRat);
    for (int i = 0; i < mNumberOfParameterizationTPCRat; i++) {
      mParameterizationTPCRat->AddAtAndExpand(new Chebyshev3D(*src.getParameterTPCRatIntegral(i)), i);
    }
  }

  mNumberOfParameterizationDipole = src.mNumberOfParameterizationDipole;
  mNumberOfDistinctZSegmentsDipole = src.mNumberOfDistinctZSegmentsDipole;
  mNumberOfDistinctYSegmentsDipole = src.mNumberOfDistinctYSegmentsDipole;
  mNumberOfDistinctXSegmentsDipole = src.mNumberOfDistinctXSegmentsDipole;
  mMinDipoleZ = src.mMinDipoleZ;
  mMaxDipoleZ = src.mMaxDipoleZ;
  if (src.mNumberOfParameterizationDipole) {
    memcpy(mCoordinatesSegmentsZDipole = new Float_t[mNumberOfDistinctZSegmentsDipole], src.mCoordinatesSegmentsZDipole,
           sizeof(Float_t) * mNumberOfDistinctZSegmentsDipole);
    memcpy(mCoordinatesSegmentsYDipole = new Float_t[mNumberOfDistinctYSegmentsDipole], src.mCoordinatesSegmentsYDipole,
           sizeof(Float_t) * mNumberOfDistinctYSegmentsDipole);
    memcpy(mCoordinatesSegmentsXDipole = new Float_t[mNumberOfDistinctXSegmentsDipole], src.mCoordinatesSegmentsXDipole,
           sizeof(Float_t) * mNumberOfDistinctXSegmentsDipole);
    memcpy(mBeginningOfSegmentsYDipole = new Int_t[mNumberOfDistinctZSegmentsDipole], src.mBeginningOfSegmentsYDipole,
           sizeof(Int_t) * mNumberOfDistinctZSegmentsDipole);
    memcpy(mNumberOfSegmentsYDipole = new Int_t[mNumberOfDistinctZSegmentsDipole], src.mNumberOfSegmentsYDipole,
           sizeof(Int_t) * mNumberOfDistinctZSegmentsDipole);
    memcpy(mBeginningOfSegmentsXDipole = new Int_t[mNumberOfDistinctYSegmentsDipole], src.mBeginningOfSegmentsXDipole,
           sizeof(Int_t) * mNumberOfDistinctYSegmentsDipole);
    memcpy(mNumberOfSegmentsXDipole = new Int_t[mNumberOfDistinctYSegmentsDipole], src.mNumberOfSegmentsXDipole,
           sizeof(Int_t) * mNumberOfDistinctYSegmentsDipole);
    memcpy(mSegmentIdDipole = new Int_t[mNumberOfDistinctXSegmentsDipole], src.mSegmentIdDipole,
           sizeof(Int_t) * mNumberOfDistinctXSegmentsDipole);
    mParameterizationDipole = new TObjArray(mNumberOfParameterizationDipole);
    for (int i = 0; i < mNumberOfParameterizationDipole; i++) {
      mParameterizationDipole->AddAtAndExpand(new Chebyshev3D(*src.getParameterDipole(i)), i);
    }
  }
}

MagneticWrapperChebyshev &MagneticWrapperChebyshev::operator=(const MagneticWrapperChebyshev &rhs)
{
  if (this != &rhs) {
    Clear();
    copyFrom(rhs);
  }
  return *this;
}

void MagneticWrapperChebyshev::Clear(const Option_t *)
{
  if (mNumberOfParameterizationSolenoid) {
    mParameterizationSolenoid->SetOwner(kTRUE);
    delete mParameterizationSolenoid;
    mParameterizationSolenoid = nullptr;
    delete[] mCoordinatesSegmentsZSolenoid;
    mCoordinatesSegmentsZSolenoid = nullptr;
    delete[] mCoordinatesSegmentsPSolenoid;
    mCoordinatesSegmentsPSolenoid = nullptr;
    delete[] mCoordinatesSegmentsRSolenoid;
    mCoordinatesSegmentsRSolenoid = nullptr;
    delete[] mBeginningOfSegmentsPSolenoid;
    mBeginningOfSegmentsPSolenoid = nullptr;
    delete[] mNumberOfSegmentsPSolenoid;
    mNumberOfSegmentsPSolenoid = nullptr;
    delete[] mBeginningOfSegmentsRSolenoid;
    mBeginningOfSegmentsRSolenoid = nullptr;
    delete[] mNumberOfRSegmentsSolenoid;
    mNumberOfRSegmentsSolenoid = nullptr;
    delete[] mSegmentIdSolenoid;
    mSegmentIdSolenoid = nullptr;
  }

  mNumberOfParameterizationSolenoid = mNumberOfDistinctZSegmentsSolenoid = mNumberOfDistinctPSegmentsSolenoid =
  mNumberOfDistinctRSegmentsSolenoid = 0;
  mMinZSolenoid = 1e6;
  mMaxZSolenoid = -1e6;
  mMaxRadiusSolenoid = 0;

  if (mNumberOfParameterizationTPC) {
    mParameterizationTPC->SetOwner(kTRUE);
    delete mParameterizationTPC;
    mParameterizationTPC = nullptr;
    delete[] mCoordinatesSegmentsZTPC;
    mCoordinatesSegmentsZTPC = nullptr;
    delete[] mCoordinatesSegmentsPTPC;
    mCoordinatesSegmentsPTPC = nullptr;
    delete[] mCoordinatesSegmentsRTPC;
    mCoordinatesSegmentsRTPC = nullptr;
    delete[] mBeginningOfSegmentsPTPC;
    mBeginningOfSegmentsPTPC = nullptr;
    delete[] mNumberOfSegmentsPTPC;
    mNumberOfSegmentsPTPC = nullptr;
    delete[] mBeginningOfSegmentsRTPC;
    mBeginningOfSegmentsRTPC = nullptr;
    delete[] mNumberOfRSegmentsTPC;
    mNumberOfRSegmentsTPC = nullptr;
    delete[] mSegmentIdTPC;
    mSegmentIdTPC = nullptr;
  }

  mNumberOfParameterizationTPC = mNumberOfDistinctZSegmentsTPC = mNumberOfDistinctPSegmentsTPC =
  mNumberOfDistinctRSegmentsTPC = 0;
  mMinZTPC = 1e6;
  mMaxZTPC = -1e6;
  mMaxRadiusTPC = 0;

  if (mNumberOfParameterizationTPCRat) {
    mParameterizationTPCRat->SetOwner(kTRUE);
    delete mParameterizationTPCRat;
    mParameterizationTPCRat = nullptr;
    delete[] mCoordinatesSegmentsZTPCRat;
    mCoordinatesSegmentsZTPCRat = nullptr;
    delete[] mCoordinatesSegmentsPTPCRat;
    mCoordinatesSegmentsPTPCRat = nullptr;
    delete[] mCoordinatesSegmentsRTPCRat;
    mCoordinatesSegmentsRTPCRat = nullptr;
    delete[] mBeginningOfSegmentsPTPCRat;
    mBeginningOfSegmentsPTPCRat = nullptr;
    delete[] mNumberOfSegmentsPTPCRat;
    mNumberOfSegmentsPTPCRat = nullptr;
    delete[] mBeginningOfSegmentsRTPCRat;
    mBeginningOfSegmentsRTPCRat = nullptr;
    delete[] mNumberOfRSegmentsTPCRat;
    mNumberOfRSegmentsTPCRat = nullptr;
    delete[] mSegmentIdTPCRat;
    mSegmentIdTPCRat = nullptr;
  }

  mNumberOfParameterizationTPCRat = mNumberOfDistinctZSegmentsTPCRat = mNumberOfDistinctPSegmentsTPCRat =
  mNumberOfDistinctRSegmentsTPCRat = 0;
  mMinZTPCRat = 1e6;
  mMaxZTPCRat = -1e6;
  mMaxRadiusTPCRat = 0;

  if (mNumberOfParameterizationDipole) {
    mParameterizationDipole->SetOwner(kTRUE);
    delete mParameterizationDipole;
    mParameterizationDipole = nullptr;
    delete[] mCoordinatesSegmentsZDipole;
    mCoordinatesSegmentsZDipole = nullptr;
    delete[] mCoordinatesSegmentsYDipole;
    mCoordinatesSegmentsYDipole = nullptr;
    delete[] mCoordinatesSegmentsXDipole;
    mCoordinatesSegmentsXDipole = nullptr;
    delete[] mBeginningOfSegmentsYDipole;
    mBeginningOfSegmentsYDipole = nullptr;
    delete[] mNumberOfSegmentsYDipole;
    mNumberOfSegmentsYDipole = nullptr;
    delete[] mBeginningOfSegmentsXDipole;
    mBeginningOfSegmentsXDipole = nullptr;
    delete[] mNumberOfSegmentsXDipole;
    mNumberOfSegmentsXDipole = nullptr;
    delete[] mSegmentIdDipole;
    mSegmentIdDipole = nullptr;
  }

  mNumberOfParameterizationDipole = mNumberOfDistinctZSegmentsDipole = mNumberOfDistinctYSegmentsDipole =
  mNumberOfDistinctXSegmentsDipole = 0;
  mMinDipoleZ = 1e6;
  mMaxDipoleZ = -1e6;
}

void MagneticWrapperChebyshev::Field(const Double_t *xyz, Double_t *b) const
{
  Double_t rphiz[3];

#ifndef _BRING_TO_BOUNDARY_ // exact matching to fitted volume is requested
  b[0] = b[1] = b[2] = 0;
#endif

  if (xyz[2] > mMinZSolenoid) {
    cartesianToCylindrical(xyz, rphiz);
    fieldCylindricalSolenoid(rphiz, b);
    // convert field to cartesian system
    cylindricalToCartesianCylB(rphiz, b, b);
    return;
  }

  int iddip = findDipoleSegment(xyz);
  if (iddip < 0) {
    return;
  }
  Chebyshev3D *par = getParameterDipole(iddip);
#ifndef _BRING_TO_BOUNDARY_
  if (!par->isInside(xyz)) {
    return;
  }
#endif
  par->Eval(xyz, b);
}

Double_t MagneticWrapperChebyshev::getBz(const Double_t *xyz) const
{
  Double_t rphiz[3];

  if (xyz[2] > mMinZSolenoid) {
    cartesianToCylindrical(xyz, rphiz);
    return fieldCylindricalSolenoidBz(rphiz);
  }

  int iddip = findDipoleSegment(xyz);
  if (iddip < 0) {
    return 0.;
  }
  Chebyshev3D *par = getParameterDipole(iddip);
#ifndef _BRING_TO_BOUNDARY_
  if (!par->isInside(xyz)) {
    return 0.;
  }
#endif
  return par->Eval(xyz, 2);
}

void MagneticWrapperChebyshev::Print(Option_t *) const
{
  printf("Alice magnetic field parameterized by Chebyshev polynomials\n");
  printf("Segmentation for Solenoid (%+.2f<Z<%+.2f cm | R<%.2f cm)\n", mMinZSolenoid, mMaxZSolenoid,
         mMaxRadiusSolenoid);

  if (mParameterizationSolenoid) {
    for (int i = 0; i < mNumberOfParameterizationSolenoid; i++) {
      printf("SOL%4d ", i);
      getParameterSolenoid(i)->Print();
    }
  }

  printf("Segmentation for TPC field integral (%+.2f<Z<%+.2f cm | R<%.2f cm)\n", mMinZTPC, mMaxZTPC, mMaxRadiusTPC);

  if (mParameterizationTPC) {
    for (int i = 0; i < mNumberOfParameterizationTPC; i++) {
      printf("TPC%4d ", i);
      getParameterTPCIntegral(i)->Print();
    }
  }

  printf("Segmentation for TPC field ratios integral (%+.2f<Z<%+.2f cm | R<%.2f cm)\n", mMinZTPCRat, mMaxZTPCRat,
         mMaxRadiusTPCRat);

  if (mParameterizationTPCRat) {
    for (int i = 0; i < mNumberOfParameterizationTPCRat; i++) {
      printf("TPCRat%4d ", i);
      getParameterTPCRatIntegral(i)->Print();
    }
  }

  printf("Segmentation for Dipole (%+.2f<Z<%+.2f cm)\n", mMinDipoleZ, mMaxDipoleZ);
  if (mParameterizationDipole) {
    for (int i = 0; i < mNumberOfParameterizationDipole; i++) {
      printf("DIP%4d ", i);
      getParameterDipole(i)->Print();
    }
  }
}

Int_t MagneticWrapperChebyshev::findDipoleSegment(const Double_t *xyz) const
{
  if (!mNumberOfParameterizationDipole) {
    return -1;
  }
  int xid, yid, zid = TMath::BinarySearch(mNumberOfDistinctZSegmentsDipole, mCoordinatesSegmentsZDipole,
                                          (Float_t) xyz[2]); // find zsegment

  Bool_t reCheck = kFALSE;
  while (true) {
    int ysegBeg = mBeginningOfSegmentsYDipole[zid];

    for (yid = 0; yid < mNumberOfSegmentsYDipole[zid]; yid++) {
      if (xyz[1] < mCoordinatesSegmentsYDipole[ysegBeg + yid]) {
        break;
      }
    }
    if (--yid < 0) {
      yid = 0;
    }
    yid += ysegBeg;

    int xsegBeg = mBeginningOfSegmentsXDipole[yid];
    for (xid = 0; xid < mNumberOfSegmentsXDipole[yid]; xid++) {
      if (xyz[0] < mCoordinatesSegmentsXDipole[xsegBeg + xid]) {
        break;
      }
    }

    if (--xid < 0) {
      xid = 0;
    }
    xid += xsegBeg;

    // to make sure that due to the precision problems we did not pick the next Zbin
    if (!reCheck && (xyz[2] - mCoordinatesSegmentsZDipole[zid] < 3.e-5) && zid &&
        !getParameterDipole(mSegmentIdDipole[xid])->isInside(xyz)) { // check the previous Z bin
      zid--;
      reCheck = kTRUE;
      continue;
    }
    break;
  }
  return mSegmentIdDipole[xid];
}

Int_t MagneticWrapperChebyshev::findSolenoidSegment(const Double_t *rpz) const
{
  if (!mNumberOfParameterizationSolenoid) {
    return -1;
  }
  int rid, pid, zid = TMath::BinarySearch(mNumberOfDistinctZSegmentsSolenoid, mCoordinatesSegmentsZSolenoid,
                                          (Float_t) rpz[2]); // find zsegment

  Bool_t reCheck = kFALSE;
  while (true) {
    int psegBeg = mBeginningOfSegmentsPSolenoid[zid];
    for (pid = 0; pid < mNumberOfSegmentsPSolenoid[zid]; pid++) {
      if (rpz[1] < mCoordinatesSegmentsPSolenoid[psegBeg + pid]) {
        break;
      }
    }
    if (--pid < 0) {
      pid = 0;
    }
    pid += psegBeg;

    int rsegBeg = mBeginningOfSegmentsRSolenoid[pid];
    for (rid = 0; rid < mNumberOfRSegmentsSolenoid[pid]; rid++) {
      if (rpz[0] < mCoordinatesSegmentsRSolenoid[rsegBeg + rid]) {
        break;
      }
    }
    if (--rid < 0) {
      rid = 0;
    }
    rid += rsegBeg;

    // to make sure that due to the precision problems we did not pick the next Zbin
    if (!reCheck && (rpz[2] - mCoordinatesSegmentsZSolenoid[zid] < 3.e-5) && zid &&
        !getParameterSolenoid(mSegmentIdSolenoid[rid])->isInside(rpz)) { // check the previous Z bin
      zid--;
      reCheck = kTRUE;
      continue;
    }
    break;
  }
  return mSegmentIdSolenoid[rid];
}

Int_t MagneticWrapperChebyshev::findTPCSegment(const Double_t *rpz) const
{
  if (!mNumberOfParameterizationTPC) {
    return -1;
  }
  int rid, pid, zid = TMath::BinarySearch(mNumberOfDistinctZSegmentsTPC, mCoordinatesSegmentsZTPC,
                                          (Float_t) rpz[2]); // find zsegment

  Bool_t reCheck = kFALSE;
  while (true) {
    int psegBeg = mBeginningOfSegmentsPTPC[zid];

    for (pid = 0; pid < mNumberOfSegmentsPTPC[zid]; pid++) {
      if (rpz[1] < mCoordinatesSegmentsPTPC[psegBeg + pid]) {
        break;
      }
    }
    if (--pid < 0) {
      pid = 0;
    }
    pid += psegBeg;

    int rsegBeg = mBeginningOfSegmentsRTPC[pid];
    for (rid = 0; rid < mNumberOfRSegmentsTPC[pid]; rid++) {
      if (rpz[0] < mCoordinatesSegmentsRTPC[rsegBeg + rid]) {
        break;
      }
    }
    if (--rid < 0) {
      rid = 0;
    }
    rid += rsegBeg;

    // to make sure that due to the precision problems we did not pick the next Zbin
    if (!reCheck && (rpz[2] - mCoordinatesSegmentsZTPC[zid] < 3.e-5) && zid &&
        !getParameterTPCIntegral(mSegmentIdTPC[rid])->isInside(rpz)) { // check the previous Z bin
      zid--;
      reCheck = kTRUE;
      continue;
    }
    break;
  }
  return mSegmentIdTPC[rid];
}

Int_t MagneticWrapperChebyshev::findTPCRatSegment(const Double_t *rpz) const
{
  if (!mNumberOfParameterizationTPCRat) {
    return -1;
  }
  int rid, pid, zid = TMath::BinarySearch(mNumberOfDistinctZSegmentsTPCRat, mCoordinatesSegmentsZTPCRat,
                                          (Float_t) rpz[2]); // find zsegment

  Bool_t reCheck = kFALSE;
  while (true) {
    int psegBeg = mBeginningOfSegmentsPTPCRat[zid];

    for (pid = 0; pid < mNumberOfSegmentsPTPCRat[zid]; pid++) {
      if (rpz[1] < mCoordinatesSegmentsPTPCRat[psegBeg + pid]) {
        break;
      }
    }
    if (--pid < 0) {
      pid = 0;
    }
    pid += psegBeg;

    int rsegBeg = mBeginningOfSegmentsRTPCRat[pid];
    for (rid = 0; rid < mNumberOfRSegmentsTPCRat[pid]; rid++) {
      if (rpz[0] < mCoordinatesSegmentsRTPCRat[rsegBeg + rid]) {
        break;
      }
    }
    if (--rid < 0) {
      rid = 0;
    }
    rid += rsegBeg;

    // to make sure that due to the precision problems we did not pick the next Zbin
    if (!reCheck && (rpz[2] - mCoordinatesSegmentsZTPCRat[zid] < 3.e-5) && zid &&
        !getParameterTPCRatIntegral(mSegmentIdTPCRat[rid])->isInside(rpz)) { // check the previous Z bin
      zid--;
      reCheck = kTRUE;
      continue;
    }
    break;
  }
  return mSegmentIdTPCRat[rid];
}

void MagneticWrapperChebyshev::getTPCIntegral(const Double_t *xyz, Double_t *b) const
{
  static Double_t rphiz[3];

  // TPCInt region
  // convert coordinates to cyl system
  cartesianToCylindrical(xyz, rphiz);
#ifndef _BRING_TO_BOUNDARY_
  if ((rphiz[2] > getMaxZTPCIntegral() || rphiz[2] < getMinZTPCIntegral()) || rphiz[0] > getMaxRTPCIntegral()) {
    for (int i = 3; i--;) {
      b[i] = 0;
    }
    return;
  }
#endif

  getTPCIntegralCylindrical(rphiz, b);

  // convert field to cartesian system
  cylindricalToCartesianCylB(rphiz, b, b);
}

void MagneticWrapperChebyshev::getTPCRatIntegral(const Double_t *xyz, Double_t *b) const
{
  static Double_t rphiz[3];

  // TPCRatIntegral region
  // convert coordinates to cylindrical system
  cartesianToCylindrical(xyz, rphiz);
#ifndef _BRING_TO_BOUNDARY_
  if ((rphiz[2] > getMaxZTPCRatIntegral() || rphiz[2] < getMinZTPCRatIntegral()) ||
      rphiz[0] > getMaxRTPCRatIntegral()) {
    for (int i = 3; i--;) {
      b[i] = 0;
    }
    return;
  }
#endif

  getTPCRatIntegralCylindrical(rphiz, b);

  // convert field to cartesian system
  cylindricalToCartesianCylB(rphiz, b, b);
}

void MagneticWrapperChebyshev::fieldCylindricalSolenoid(const Double_t *rphiz, Double_t *b) const
{
  int id = findSolenoidSegment(rphiz);
  if (id < 0) {
    return;
  }
  Chebyshev3D *par = getParameterSolenoid(id);
#ifndef _BRING_TO_BOUNDARY_ // exact matching to fitted volume is requested
  if (!par->isInside(rphiz)) {
    return;
  }
#endif
  par->Eval(rphiz, b);
  return;
}

Double_t MagneticWrapperChebyshev::fieldCylindricalSolenoidBz(const Double_t *rphiz) const
{
  int id = findSolenoidSegment(rphiz);
  if (id < 0) {
    return 0.;
  }
  Chebyshev3D *par = getParameterSolenoid(id);
#ifndef _BRING_TO_BOUNDARY_
  return par->isInside(rphiz) ? par->Eval(rphiz, 2) : 0;
#else
  return par->Eval(rphiz, 2);
#endif
}

void MagneticWrapperChebyshev::getTPCIntegralCylindrical(const Double_t *rphiz, Double_t *b) const
{
  int id = findTPCSegment(rphiz);
  if (id < 0) {
    b[0] = b[1] = b[2] = 0;
    return;
  }
  if (id >= mNumberOfParameterizationTPC) {
    mLogger->Error(MESSAGE_ORIGIN, "Wrong TPCParam segment %d", id);
    b[0] = b[1] = b[2] = 0;
    return;
  }
  Chebyshev3D *par = getParameterTPCIntegral(id);
  if (par->isInside(rphiz)) {
    par->Eval(rphiz, b);
    return;
  }
  b[0] = b[1] = b[2] = 0;
  return;
}

void MagneticWrapperChebyshev::getTPCRatIntegralCylindrical(const Double_t *rphiz, Double_t *b) const
{
  int id = findTPCRatSegment(rphiz);
  if (id < 0) {
    b[0] = b[1] = b[2] = 0;
    return;
  }
  if (id >= mNumberOfParameterizationTPCRat) {
    mLogger->Error(MESSAGE_ORIGIN, "Wrong TPCRatParam segment %d", id);
    b[0] = b[1] = b[2] = 0;
    return;
  }
  Chebyshev3D *par = getParameterTPCRatIntegral(id);
  if (par->isInside(rphiz)) {
    par->Eval(rphiz, b);
    return;
  }
  b[0] = b[1] = b[2] = 0;
  return;
}

#ifdef _INC_CREATION_Chebyshev3D_

void MagneticWrapperChebyshev::loadData(const char* inpfile)
{
  TString strf = inpfile;
  gSystem->ExpandPathName(strf);
  FILE* stream = fopen(strf, "r");

  if (!stream) {
    printf("Did not find input file %s\n", strf.Data());
    return;
  }

  TString buffs;
  Chebyshev3DCalc::readLine(buffs, stream);

  if (!buffs.BeginsWith("START")) {
    Error("LoadData", "Expected: \"START <name>\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }

  if (buffs.First(' ') > 0) {
    SetName(buffs.Data() + buffs.First(' ') + 1);
  }

  // Solenoid part
  Chebyshev3DCalc::readLine(buffs, stream);

  if (!buffs.BeginsWith("START SOLENOID")) {
    Error("LoadData", "Expected: \"START SOLENOID\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }
  Chebyshev3DCalc::readLine(buffs, stream); // nparam
  int nparSol = buffs.Atoi();

  for (int ip = 0; ip < nparSol; ip++) {
    auto* cheb = new Chebyshev3D();
    cheb->loadData(stream);
    addParameterSolenoid(cheb);
  }

  Chebyshev3DCalc::readLine(buffs, stream);
  if (!buffs.BeginsWith("END SOLENOID")) {
    Error("LoadData", "Expected \"END SOLENOID\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }

  // TPCInt part
  Chebyshev3DCalc::readLine(buffs, stream);
  if (!buffs.BeginsWith("START TPCINT")) {
    Error("LoadData", "Expected: \"START TPCINT\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }
  Chebyshev3DCalc::readLine(buffs, stream); // nparam
  int nparTPCInt = buffs.Atoi();

  for (int ip = 0; ip < nparTPCInt; ip++) {
    auto* cheb = new Chebyshev3D();
    cheb->loadData(stream);
    addParameterTPCIntegral(cheb);
  }

  Chebyshev3DCalc::readLine(buffs, stream);

  if (!buffs.BeginsWith("END TPCINT")) {
    Error("LoadData", "Expected \"END TPCINT\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }

  // TPCRatInt part
  Chebyshev3DCalc::readLine(buffs, stream);

  if (!buffs.BeginsWith("START TPCRatINT")) {
    Error("LoadData", "Expected: \"START TPCRatINT\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }

  Chebyshev3DCalc::readLine(buffs, stream); // nparam
  int nparTPCRatInt = buffs.Atoi();

  for (int ip = 0; ip < nparTPCRatInt; ip++) {
    auto* cheb = new Chebyshev3D();
    cheb->loadData(stream);
    addParameterTPCRatIntegral(cheb);
  }

  Chebyshev3DCalc::readLine(buffs, stream);

  if (!buffs.BeginsWith("END TPCRatINT")) {
    Error("LoadData", "Expected \"END TPCRatINT\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }

  // Dipole part
  Chebyshev3DCalc::readLine(buffs, stream);

  if (!buffs.BeginsWith("START DIPOLE")) {
    Error("LoadData", "Expected: \"START DIPOLE\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }

  Chebyshev3DCalc::readLine(buffs, stream); // nparam
  int nparDip = buffs.Atoi();

  for (int ip = 0; ip < nparDip; ip++) {
    auto* cheb = new Chebyshev3D();
    cheb->loadData(stream);
    addParameterDipole(cheb);
  }

  Chebyshev3DCalc::readLine(buffs, stream);

  if (!buffs.BeginsWith("END DIPOLE")) {
    Error("LoadData", "Expected \"END DIPOLE\", found \"%s\"\nStop\n", buffs.Data());
    exit(1);
  }

  Chebyshev3DCalc::readLine(buffs, stream);

  if (!buffs.BeginsWith("END ") && !buffs.Contains(GetName())) {
    Error("LoadData", "Expected: \"END %s\", found \"%s\"\nStop\n", GetName(), buffs.Data());
    exit(1);
  }

  fclose(stream);
  buildTableSolenoid();
  buildTableDipole();
  buildTableTPCIntegral();
  buildTableTPCRatIntegral();

  printf("Loaded magnetic field \"%s\" from %s\n", GetName(), strf.Data());
}

void MagneticWrapperChebyshev::buildTableSolenoid()
{
  buildTable(mNumberOfParameterizationSolenoid, mParameterizationSolenoid, mNumberOfDistinctZSegmentsSolenoid,
             mNumberOfDistinctPSegmentsSolenoid, mNumberOfDistinctRSegmentsSolenoid, mMinZSolenoid, mMaxZSolenoid,
             &mCoordinatesSegmentsZSolenoid, &mCoordinatesSegmentsPSolenoid, &mCoordinatesSegmentsRSolenoid,
             &mBeginningOfSegmentsPSolenoid, &mNumberOfSegmentsPSolenoid, &mBeginningOfSegmentsRSolenoid,
             &mNumberOfRSegmentsSolenoid, &mSegmentIdSolenoid);
}

void MagneticWrapperChebyshev::buildTableDipole()
{
  buildTable(mNumberOfParameterizationDipole, mParameterizationDipole, mNumberOfDistinctZSegmentsDipole,
             mNumberOfDistinctYSegmentsDipole, mNumberOfDistinctXSegmentsDipole, mMinDipoleZ, mMaxDipoleZ,
             &mCoordinatesSegmentsZDipole, &mCoordinatesSegmentsYDipole, &mCoordinatesSegmentsXDipole,
             &mBeginningOfSegmentsYDipole, &mNumberOfSegmentsYDipole, &mBeginningOfSegmentsXDipole,
             &mNumberOfSegmentsXDipole, &mSegmentIdDipole);
}

void MagneticWrapperChebyshev::buildTableTPCIntegral()
{
  buildTable(mNumberOfParameterizationTPC, mParameterizationTPC, mNumberOfDistinctZSegmentsTPC,
             mNumberOfDistinctPSegmentsTPC, mNumberOfDistinctRSegmentsTPC, mMinZTPC, mMaxZTPC,
             &mCoordinatesSegmentsZTPC, &mCoordinatesSegmentsPTPC, &mCoordinatesSegmentsRTPC, &mBeginningOfSegmentsPTPC,
             &mNumberOfSegmentsPTPC, &mBeginningOfSegmentsRTPC, &mNumberOfRSegmentsTPC, &mSegmentIdTPC);
}

void MagneticWrapperChebyshev::buildTableTPCRatIntegral()
{
  buildTable(mNumberOfParameterizationTPCRat, mParameterizationTPCRat, mNumberOfDistinctZSegmentsTPCRat,
             mNumberOfDistinctPSegmentsTPCRat, mNumberOfDistinctRSegmentsTPCRat, mMinZTPCRat, mMaxZTPCRat,
             &mCoordinatesSegmentsZTPCRat, &mCoordinatesSegmentsPTPCRat, &mCoordinatesSegmentsRTPCRat,
             &mBeginningOfSegmentsPTPCRat, &mNumberOfSegmentsPTPCRat, &mBeginningOfSegmentsRTPCRat,
             &mNumberOfRSegmentsTPCRat, &mSegmentIdTPCRat);
}

#endif

#ifdef _INC_CREATION_Chebyshev3D_
MagneticWrapperChebyshev::MagneticWrapperChebyshev(const char* inputFile)
  : mNumberOfParameterizationSolenoid(0),
    mNumberOfDistinctZSegmentsSolenoid(0),
    mNumberOfDistinctPSegmentsSolenoid(0),
    mNumberOfDistinctRSegmentsSolenoid(0),
    mCoordinatesSegmentsZSolenoid(nullptr),
    mCoordinatesSegmentsPSolenoid(nullptr),
    mCoordinatesSegmentsRSolenoid(nullptr),
    mBeginningOfSegmentsPSolenoid(nullptr),
    mNumberOfSegmentsPSolenoid(nullptr),
    mBeginningOfSegmentsRSolenoid(nullptr),
    mNumberOfRSegmentsSolenoid(nullptr),
    mSegmentIdSolenoid(nullptr),
    mMinZSolenoid(1.e6),
    mMaxZSolenoid(-1.e6),
    mParameterizationSolenoid(nullptr),
    mMaxRadiusSolenoid(0),
    mNumberOfParameterizationTPC(0),
    mNumberOfDistinctZSegmentsTPC(0),
    mNumberOfDistinctPSegmentsTPC(0),
    mNumberOfDistinctRSegmentsTPC(0),
    mCoordinatesSegmentsZTPC(nullptr),
    mCoordinatesSegmentsPTPC(nullptr),
    mCoordinatesSegmentsRTPC(nullptr),
    mBeginningOfSegmentsPTPC(nullptr),
    mNumberOfSegmentsPTPC(nullptr),
    mBeginningOfSegmentsRTPC(nullptr),
    mNumberOfRSegmentsTPC(nullptr),
    mSegmentIdTPC(nullptr),
    mMinZTPC(1.e6),
    mMaxZTPC(-1.e6),
    mParameterizationTPC(nullptr),
    mMaxRadiusTPC(0),
    mNumberOfParameterizationTPCRat(0),
    mNumberOfDistinctZSegmentsTPCRat(0),
    mNumberOfDistinctPSegmentsTPCRat(0),
    mNumberOfDistinctRSegmentsTPCRat(0),
    mCoordinatesSegmentsZTPCRat(nullptr),
    mCoordinatesSegmentsPTPCRat(nullptr),
    mCoordinatesSegmentsRTPCRat(nullptr),
    mBeginningOfSegmentsPTPCRat(nullptr),
    mNumberOfSegmentsPTPCRat(nullptr),
    mBeginningOfSegmentsRTPCRat(nullptr),
    mNumberOfRSegmentsTPCRat(nullptr),
    mSegmentIdTPCRat(nullptr),
    mMinZTPCRat(1.e6),
    mMaxZTPCRat(-1.e6),
    mParameterizationTPCRat(nullptr),
    mMaxRadiusTPCRat(0),
    mNumberOfParameterizationDipole(0),
    mNumberOfDistinctZSegmentsDipole(0),
    mNumberOfDistinctYSegmentsDipole(0),
    mNumberOfDistinctXSegmentsDipole(0),
    mCoordinatesSegmentsZDipole(nullptr),
    mCoordinatesSegmentsYDipole(nullptr),
    mCoordinatesSegmentsXDipole(nullptr),
    mBeginningOfSegmentsYDipole(nullptr),
    mNumberOfSegmentsYDipole(nullptr),
    mBeginningOfSegmentsXDipole(nullptr),
    mNumberOfSegmentsXDipole(nullptr),
    mSegmentIdDipole(nullptr),
    mMinDipoleZ(1.e6),
    mMaxDipoleZ(-1.e6),
    mParameterizationDipole(nullptr)
{
  loadData(inputFile);
}

void MagneticWrapperChebyshev::addParameterSolenoid(const Chebyshev3D* param)
{
  if (!mParameterizationSolenoid) {
    mParameterizationSolenoid = new TObjArray();
  }
  mParameterizationSolenoid->Add((Chebyshev3D*)param);
  mNumberOfParameterizationSolenoid++;
  if (mMaxRadiusSolenoid < param->getBoundMax(0)) {
    mMaxRadiusSolenoid = param->getBoundMax(0);
  }
}

void MagneticWrapperChebyshev::addParameterTPCIntegral(const Chebyshev3D* param)
{
  if (!mParameterizationTPC) {
    mParameterizationTPC = new TObjArray();
  }
  mParameterizationTPC->Add((Chebyshev3D*)param);
  mNumberOfParameterizationTPC++;
  if (mMaxRadiusTPC < param->getBoundMax(0)) {
    mMaxRadiusTPC = param->getBoundMax(0);
  }
}

void MagneticWrapperChebyshev::addParameterTPCRatIntegral(const Chebyshev3D* param)
{
  if (!mParameterizationTPCRat) {
    mParameterizationTPCRat = new TObjArray();
  }
  mParameterizationTPCRat->Add((Chebyshev3D*)param);
  mNumberOfParameterizationTPCRat++;
  if (mMaxRadiusTPCRat < param->getBoundMax(0)) {
    mMaxRadiusTPCRat = param->getBoundMax(0);
  }
}

void MagneticWrapperChebyshev::addParameterDipole(const Chebyshev3D* param)
{
  if (!mParameterizationDipole) {
    mParameterizationDipole = new TObjArray();
  }
  mParameterizationDipole->Add((Chebyshev3D*)param);
  mNumberOfParameterizationDipole++;
}

void MagneticWrapperChebyshev::resetDipole()
{
  if (mNumberOfParameterizationDipole) {
    delete mParameterizationDipole;
    mParameterizationDipole = nullptr;
    delete[] mCoordinatesSegmentsZDipole;
    mCoordinatesSegmentsZDipole = nullptr;
    delete[] mCoordinatesSegmentsXDipole;
    mCoordinatesSegmentsXDipole = nullptr;
    delete[] mCoordinatesSegmentsYDipole;
    mCoordinatesSegmentsYDipole = nullptr;
    delete[] mBeginningOfSegmentsYDipole;
    mBeginningOfSegmentsYDipole = nullptr;
    delete[] mNumberOfSegmentsYDipole;
    mNumberOfSegmentsYDipole = nullptr;
    delete[] mBeginningOfSegmentsXDipole;
    mBeginningOfSegmentsXDipole = nullptr;
    delete[] mNumberOfSegmentsXDipole;
    mNumberOfSegmentsXDipole = nullptr;
    delete[] mSegmentIdDipole;
    mSegmentIdDipole = nullptr;
  }
  mNumberOfParameterizationDipole = mNumberOfDistinctZSegmentsDipole = mNumberOfDistinctXSegmentsDipole =
    mNumberOfDistinctYSegmentsDipole = 0;
  mMinDipoleZ = 1e6;
  mMaxDipoleZ = -1e6;
}

void MagneticWrapperChebyshev::resetSolenoid()
{
  if (mNumberOfParameterizationSolenoid) {
    delete mParameterizationSolenoid;
    mParameterizationSolenoid = nullptr;
    delete[] mCoordinatesSegmentsZSolenoid;
    mCoordinatesSegmentsZSolenoid = nullptr;
    delete[] mCoordinatesSegmentsPSolenoid;
    mCoordinatesSegmentsPSolenoid = nullptr;
    delete[] mCoordinatesSegmentsRSolenoid;
    mCoordinatesSegmentsRSolenoid = nullptr;
    delete[] mBeginningOfSegmentsPSolenoid;
    mBeginningOfSegmentsPSolenoid = nullptr;
    delete[] mNumberOfSegmentsPSolenoid;
    mNumberOfSegmentsPSolenoid = nullptr;
    delete[] mBeginningOfSegmentsRSolenoid;
    mBeginningOfSegmentsRSolenoid = nullptr;
    delete[] mNumberOfRSegmentsSolenoid;
    mNumberOfRSegmentsSolenoid = nullptr;
    delete[] mSegmentIdSolenoid;
    mSegmentIdSolenoid = nullptr;
  }
  mNumberOfParameterizationSolenoid = mNumberOfDistinctZSegmentsSolenoid = mNumberOfDistinctPSegmentsSolenoid =
    mNumberOfDistinctRSegmentsSolenoid = 0;
  mMinZSolenoid = 1e6;
  mMaxZSolenoid = -1e6;
  mMaxRadiusSolenoid = 0;
}

void MagneticWrapperChebyshev::resetTPCIntegral()
{
  if (mNumberOfParameterizationTPC) {
    delete mParameterizationTPC;
    mParameterizationTPC = nullptr;
    delete[] mCoordinatesSegmentsZTPC;
    mCoordinatesSegmentsZTPC = nullptr;
    delete[] mCoordinatesSegmentsPTPC;
    mCoordinatesSegmentsPTPC = nullptr;
    delete[] mCoordinatesSegmentsRTPC;
    mCoordinatesSegmentsRTPC = nullptr;
    delete[] mBeginningOfSegmentsPTPC;
    mBeginningOfSegmentsPTPC = nullptr;
    delete[] mNumberOfSegmentsPTPC;
    mNumberOfSegmentsPTPC = nullptr;
    delete[] mBeginningOfSegmentsRTPC;
    mBeginningOfSegmentsRTPC = nullptr;
    delete[] mNumberOfRSegmentsTPC;
    mNumberOfRSegmentsTPC = nullptr;
    delete[] mSegmentIdTPC;
    mSegmentIdTPC = nullptr;
  }
  mNumberOfParameterizationTPC = mNumberOfDistinctZSegmentsTPC = mNumberOfDistinctPSegmentsTPC =
    mNumberOfDistinctRSegmentsTPC = 0;
  mMinZTPC = 1e6;
  mMaxZTPC = -1e6;
  mMaxRadiusTPC = 0;
}

void MagneticWrapperChebyshev::resetTPCRatIntegral()
{
  if (mNumberOfParameterizationTPCRat) {
    delete mParameterizationTPCRat;
    mParameterizationTPCRat = nullptr;
    delete[] mCoordinatesSegmentsZTPCRat;
    mCoordinatesSegmentsZTPCRat = nullptr;
    delete[] mCoordinatesSegmentsPTPCRat;
    mCoordinatesSegmentsPTPCRat = nullptr;
    delete[] mCoordinatesSegmentsRTPCRat;
    mCoordinatesSegmentsRTPCRat = nullptr;
    delete[] mBeginningOfSegmentsPTPCRat;
    mBeginningOfSegmentsPTPCRat = nullptr;
    delete[] mNumberOfSegmentsPTPCRat;
    mNumberOfSegmentsPTPCRat = nullptr;
    delete[] mBeginningOfSegmentsRTPCRat;
    mBeginningOfSegmentsRTPCRat = nullptr;
    delete[] mNumberOfRSegmentsTPCRat;
    mNumberOfRSegmentsTPCRat = nullptr;
    delete[] mSegmentIdTPCRat;
    mSegmentIdTPCRat = nullptr;
  }
  mNumberOfParameterizationTPCRat = mNumberOfDistinctZSegmentsTPCRat = mNumberOfDistinctPSegmentsTPCRat =
    mNumberOfDistinctRSegmentsTPCRat = 0;
  mMinZTPCRat = 1e6;
  mMaxZTPCRat = -1e6;
  mMaxRadiusTPCRat = 0;
}

void MagneticWrapperChebyshev::buildTable(Int_t npar, TObjArray* parArr, Int_t& nZSeg, Int_t& nYSeg, Int_t& nXSeg,
                                          Float_t& minZ, Float_t& maxZ, Float_t** segZ, Float_t** segY, Float_t** segX,
                                          Int_t** begSegY, Int_t** nSegY, Int_t** begSegX, Int_t** nSegX, Int_t** segID)
{
  if (npar < 1) {
    return;
  }
  TArrayF segYArr, segXArr;
  TArrayI begSegYDipArr, begSegXDipArr;
  TArrayI nSegYDipArr, nSegXDipArr;
  TArrayI segIDArr;
  float* tmpSegZ, *tmpSegY, *tmpSegX;

  // create segmentation in Z
  nZSeg = segmentDimension(&tmpSegZ, parArr, npar, 2, 1, -1, 1, -1, 1, -1) - 1;
  nYSeg = 0;
  nXSeg = 0;

  // for each Z slice create segmentation in Y
  begSegYDipArr.Set(nZSeg);
  nSegYDipArr.Set(nZSeg);
  float xyz[3];
  for (int iz = 0; iz < nZSeg; iz++) {
    printf("\nZSegment#%d  %+e : %+e\n", iz, tmpSegZ[iz], tmpSegZ[iz + 1]);
    int ny = segmentDimension(&tmpSegY, parArr, npar, 1, 1, -1, 1, -1, tmpSegZ[iz], tmpSegZ[iz + 1]) - 1;
    segYArr.Set(ny + nYSeg);
    for (int iy = 0; iy < ny; iy++) {
      segYArr[nYSeg + iy] = tmpSegY[iy];
    }
    begSegYDipArr[iz] = nYSeg;
    nSegYDipArr[iz] = ny;
    printf(" Found %d YSegments, to start from %d\n", ny, begSegYDipArr[iz]);

    // for each slice in Z and Y create segmentation in X
    begSegXDipArr.Set(nYSeg + ny);
    nSegXDipArr.Set(nYSeg + ny);
    xyz[2] = (tmpSegZ[iz] + tmpSegZ[iz + 1]) / 2.; // mean Z of this segment

    for (int iy = 0; iy < ny; iy++) {
      int isg = nYSeg + iy;
      printf("\n   YSegment#%d  %+e : %+e\n", iy, tmpSegY[iy], tmpSegY[iy + 1]);
      int nx =
        segmentDimension(&tmpSegX, parArr, npar, 0, 1, -1, tmpSegY[iy], tmpSegY[iy + 1], tmpSegZ[iz], tmpSegZ[iz + 1]) -
        1;

      segXArr.Set(nx + nXSeg);
      for (int ix = 0; ix < nx; ix++) {
        segXArr[nXSeg + ix] = tmpSegX[ix];
      }
      begSegXDipArr[isg] = nXSeg;
      nSegXDipArr[isg] = nx;
      printf("   Found %d XSegments, to start from %d\n", nx, begSegXDipArr[isg]);

      segIDArr.Set(nXSeg + nx);

      // find corresponding params
      xyz[1] = (tmpSegY[iy] + tmpSegY[iy + 1]) / 2.; // mean Y of this segment

      for (int ix = 0; ix < nx; ix++) {
        xyz[0] = (tmpSegX[ix] + tmpSegX[ix + 1]) / 2.; // mean X of this segment
        for (int ipar = 0; ipar < npar; ipar++) {
          Chebyshev3D* cheb = (Chebyshev3D*)parArr->At(ipar);
          if (!cheb->isInside(xyz)) {
            continue;
          }
          segIDArr[nXSeg + ix] = ipar;
          break;
        }
      }
      nXSeg += nx;

      delete[] tmpSegX;
    }
    delete[] tmpSegY;
    nYSeg += ny;
  }

  minZ = tmpSegZ[0];
  maxZ = tmpSegZ[nZSeg];
  (*segZ) = new Float_t[nZSeg];
  for (int i = nZSeg; i--;) {
    (*segZ)[i] = tmpSegZ[i];
  }
  delete[] tmpSegZ;

  (*segY) = new Float_t[nYSeg];
  (*segX) = new Float_t[nXSeg];
  (*begSegY) = new Int_t[nZSeg];
  (*nSegY) = new Int_t[nZSeg];
  (*begSegX) = new Int_t[nYSeg];
  (*nSegX) = new Int_t[nYSeg];
  (*segID) = new Int_t[nXSeg];

  for (int i = nYSeg; i--;) {
    (*segY)[i] = segYArr[i];
  }
  for (int i = nXSeg; i--;) {
    (*segX)[i] = segXArr[i];
  }
  for (int i = nZSeg; i--;) {
    (*begSegY)[i] = begSegYDipArr[i];
    (*nSegY)[i] = nSegYDipArr[i];
  }
  for (int i = nYSeg; i--;) {
    (*begSegX)[i] = begSegXDipArr[i];
    (*nSegX)[i] = nSegXDipArr[i];
  }
  for (int i = nXSeg; i--;) {
    (*segID)[i] = segIDArr[i];
  }
}

// void MagneticWrapperChebyshev::BuildTableDip()
// {
//   // build lookup table for dipole
//
//   if (mNumberOfParameterizationDipole<1) return;
//   TArrayF segY,segX;
//   TArrayI begSegYDip,begSegXDip;
//   TArrayI nsegYDip,nsegXDip;
//   TArrayI segID;
//   float *tmpSegZ,*tmpSegY,*tmpSegX;
//
//   // create segmentation in Z
//   mNumberOfDistinctZSegmentsDipole = segmentDimension(&tmpSegZ, mParameterizationDipole,
// mNumberOfParameterizationDipole, 2, 1,-1, 1,-1, 1,-1) - 1;
//   mNumberOfDistinctYSegmentsDipole = 0;
//   mNumberOfDistinctXSegmentsDipole = 0;
//
//   // for each Z slice create segmentation in Y
//   begSegYDip.Set(mNumberOfDistinctZSegmentsDipole);
//   nsegYDip.Set(mNumberOfDistinctZSegmentsDipole);
//   float xyz[3];
//   for (int iz=0;iz<mNumberOfDistinctZSegmentsDipole;iz++) {
//     printf("\nZSegment#%d  %+e : %+e\n",iz,tmpSegZ[iz],tmpSegZ[iz+1]);
//     int ny = segmentDimension(&tmpSegY, mParameterizationDipole, mNumberOfParameterizationDipole, 1,
//          1,-1, 1,-1, tmpSegZ[iz],tmpSegZ[iz+1]) - 1;
//     segY.Set(ny + mNumberOfDistinctYSegmentsDipole);
//     for (int iy=0;iy<ny;iy++) segY[mNumberOfDistinctYSegmentsDipole+iy] = tmpSegY[iy];
//     begSegYDip[iz] = mNumberOfDistinctYSegmentsDipole;
//     nsegYDip[iz] = ny;
//     printf(" Found %d YSegments, to start from %d\n",ny, begSegYDip[iz]);
//
//     // for each slice in Z and Y create segmentation in X
//     begSegXDip.Set(mNumberOfDistinctYSegmentsDipole+ny);
//     nsegXDip.Set(mNumberOfDistinctYSegmentsDipole+ny);
//     xyz[2] = (tmpSegZ[iz]+tmpSegZ[iz+1])/2.; // mean Z of this segment
//
//     for (int iy=0;iy<ny;iy++) {
//       int isg = mNumberOfDistinctYSegmentsDipole+iy;
//       printf("\n   YSegment#%d  %+e : %+e\n",iy, tmpSegY[iy],tmpSegY[iy+1]);
//       int nx = segmentDimension(&tmpSegX, mParameterizationDipole, mNumberOfParameterizationDipole, 0,
//         1,-1, tmpSegY[iy],tmpSegY[iy+1], tmpSegZ[iz],tmpSegZ[iz+1]) - 1;
//
//       segX.Set(nx + mNumberOfDistinctXSegmentsDipole);
//       for (int ix=0;ix<nx;ix++) segX[mNumberOfDistinctXSegmentsDipole+ix] = tmpSegX[ix];
//       begSegXDip[isg] = mNumberOfDistinctXSegmentsDipole;
//       nsegXDip[isg] = nx;
//       printf("   Found %d XSegments, to start from %d\n",nx, begSegXDip[isg]);
//
//       segID.Set(mNumberOfDistinctXSegmentsDipole+nx);
//
//       // find corresponding params
//       xyz[1] = (tmpSegY[iy]+tmpSegY[iy+1])/2.; // mean Y of this segment
//
//       for (int ix=0;ix<nx;ix++) {
//   xyz[0] = (tmpSegX[ix]+tmpSegX[ix+1])/2.; // mean X of this segment
//   for (int ipar=0;ipar<mNumberOfParameterizationDipole;ipar++) {
//     Chebyshev3D* cheb = (Chebyshev3D*) mParameterizationDipole->At(ipar);
//     if (!cheb->isInside(xyz)) continue;
//     segID[mNumberOfDistinctXSegmentsDipole+ix] = ipar;
//     break;
//   }
//       }
//       mNumberOfDistinctXSegmentsDipole += nx;
//
//       delete[] tmpSegX;
//     }
//     delete[] tmpSegY;
//     mNumberOfDistinctYSegmentsDipole += ny;
//   }
//
//   mMinDipoleZ = tmpSegZ[0];
//   mMaxDipoleZ = tmpSegZ[mNumberOfDistinctZSegmentsDipole];
//   mCoordinatesSegmentsZDipole    = new Float_t[mNumberOfDistinctZSegmentsDipole];
//   for (int i=mNumberOfDistinctZSegmentsDipole;i--;) mCoordinatesSegmentsZDipole[i] = tmpSegZ[i];
//   delete[] tmpSegZ;
//
//   mCoordinatesSegmentsYDipole    = new Float_t[mNumberOfDistinctYSegmentsDipole];
//   mCoordinatesSegmentsXDipole    = new Float_t[mNumberOfDistinctXSegmentsDipole];
//   mBeginningOfSegmentsYDipole = new Int_t[mNumberOfDistinctZSegmentsDipole];
//   mNumberOfSegmentsYDipole   = new Int_t[mNumberOfDistinctZSegmentsDipole];
//   mBeginningOfSegmentsXDipole = new Int_t[mNumberOfDistinctYSegmentsDipole];
//   mNumberOfSegmentsXDipole   = new Int_t[mNumberOfDistinctYSegmentsDipole];
//   mSegmentIdDipole   = new Int_t[mNumberOfDistinctXSegmentsDipole];
//
//   for (int i=mNumberOfDistinctYSegmentsDipole;i--;) mCoordinatesSegmentsYDipole[i] = segY[i];
//   for (int i=mNumberOfDistinctXSegmentsDipole;i--;) mCoordinatesSegmentsXDipole[i] = segX[i];
//   for (int i=mNumberOfDistinctZSegmentsDipole;i--;) {mBeginningOfSegmentsYDipole[i] = begSegYDip[i];
// mNumberOfSegmentsYDipole[i] = nsegYDip[i];}
//   for (int i=mNumberOfDistinctYSegmentsDipole;i--;) {mBeginningOfSegmentsXDipole[i] = begSegXDip[i];
// mNumberOfSegmentsXDipole[i] = nsegXDip[i];}
//   for (int i=mNumberOfDistinctXSegmentsDipole;i--;) {mSegmentIdDipole[i]   = segID[i];}
// }

void MagneticWrapperChebyshev::saveData(const char* outfile) const
{
  TString strf = outfile;
  gSystem->ExpandPathName(strf);
  FILE* stream = fopen(strf, "w+");

  // Solenoid part
  fprintf(stream, "# Set of Chebyshev parameterizations for ALICE magnetic field\nSTART %s\n", GetName());
  fprintf(stream, "START SOLENOID\n#Number of pieces\n%d\n", mNumberOfParameterizationSolenoid);
  for (int ip = 0; ip < mNumberOfParameterizationSolenoid; ip++) {
    getParameterSolenoid(ip)->saveData(stream);
  }
  fprintf(stream, "#\nEND SOLENOID\n");

  // TPCIntegral part
  fprintf(stream, "# Set of Chebyshev parameterizations for ALICE magnetic field\nSTART %s\n", GetName());
  fprintf(stream, "START TPCINT\n#Number of pieces\n%d\n", mNumberOfParameterizationTPC);
  for (int ip = 0; ip < mNumberOfParameterizationTPC; ip++)
    getParameterTPCIntegral(ip)->saveData(stream);
  fprintf(stream, "#\nEND TPCINT\n");

  // TPCRatIntegral part
  fprintf(stream, "# Set of Chebyshev parameterizations for ALICE magnetic field\nSTART %s\n", GetName());
  fprintf(stream, "START TPCRatINT\n#Number of pieces\n%d\n", mNumberOfParameterizationTPCRat);
  for (int ip = 0; ip < mNumberOfParameterizationTPCRat; ip++) {
    getParameterTPCRatIntegral(ip)->saveData(stream);
  }
  fprintf(stream, "#\nEND TPCRatINT\n");

  // Dipole part
  fprintf(stream, "START DIPOLE\n#Number of pieces\n%d\n", mNumberOfParameterizationDipole);
  for (int ip = 0; ip < mNumberOfParameterizationDipole; ip++) {
    getParameterDipole(ip)->saveData(stream);
  }
  fprintf(stream, "#\nEND DIPOLE\n");

  fprintf(stream, "#\nEND %s\n", GetName());

  fclose(stream);
}

Int_t MagneticWrapperChebyshev::segmentDimension(float** seg, const TObjArray* par, int npar, int dim, float xmn,
                                                 float xmx, float ymn, float ymx, float zmn, float zmx)
{
  auto* tmpC = new float[2 * npar];
  auto* tmpInd = new int[2 * npar];
  int nseg0 = 0;
  for (int ip = 0; ip < npar; ip++) {
    Chebyshev3D* cheb = (Chebyshev3D*)par->At(ip);
    if (xmn < xmx && (cheb->getBoundMin(0) > (xmx + xmn) / 2 || cheb->getBoundMax(0) < (xmn + xmx) / 2)) {
      continue;
    }
    if (ymn < ymx && (cheb->getBoundMin(1) > (ymx + ymn) / 2 || cheb->getBoundMax(1) < (ymn + ymx) / 2)) {
      continue;
    }
    if (zmn < zmx && (cheb->getBoundMin(2) > (zmx + zmn) / 2 || cheb->getBoundMax(2) < (zmn + zmx) / 2)) {
      continue;
    }

    tmpC[nseg0++] = cheb->getBoundMin(dim);
    tmpC[nseg0++] = cheb->getBoundMax(dim);
  }
  // range Dim's boundaries in increasing order
  TMath::Sort(nseg0, tmpC, tmpInd, kFALSE);
  // count number of really different Z's
  int nseg = 0;
  float cprev = -1e6;
  for (int ip = 0; ip < nseg0; ip++) {
    if (TMath::Abs(cprev - tmpC[tmpInd[ip]]) > 1e-4) {
      cprev = tmpC[tmpInd[ip]];
      nseg++;
    } else {
      tmpInd[ip] = -1; // supress redundant Z
    }
  }

  *seg = new float[nseg]; // create final Z segmenations
  nseg = 0;
  for (int ip = 0; ip < nseg0; ip++) {
    if (tmpInd[ip] >= 0) {
      (*seg)[nseg++] = tmpC[tmpInd[ip]];
    }
  }

  delete[] tmpC;
  delete[] tmpInd;
  return nseg;
}

#endif

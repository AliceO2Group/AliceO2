/// \file Detector.cxx
/// \brief Implementation of the Detector class

#include "include/Detector.h"
#include <TVirtualMC.h>  // for TVirtualMC, gMC
#include "TString.h"     // for TString

using std::endl;
using std::cout;
using std::fstream;
using std::ios;
using std::ostream;

using namespace AliceO2::Base;


Float_t Detector::mDensityFactor = 1.0;

Detector::Detector()
    : FairDetector()
{
}

Detector::Detector(const char* name, Bool_t Active, Int_t DetId)
    : FairDetector(name, Active, DetId)
{
}

Detector::Detector(const Detector& rhs)
    : FairDetector(rhs) {}

Detector::~Detector()
{
}

Detector& Detector::operator=(const Detector& rhs)
{
  // check assignment to self
  if (this == &rhs) return *this;

  // base class assignment
  FairDetector::operator=(rhs);

  return *this;
}

void Detector::Material(Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens,
                        Float_t radl, Float_t absl, Float_t* buf, Int_t nwbuf) const
{
  TString uniquename = GetName();
  uniquename.Append("_");
  uniquename.Append(name);

  // Check this!!!
  TVirtualMC::GetMC()->Material(imat, uniquename.Data(), a, z, dens * mDensityFactor, radl, absl, buf, nwbuf);
}

void Detector::Mixture(Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens,
                       Int_t nlmat, Float_t* wmat) const
{
  TString uniquename = GetName();
  uniquename.Append("_");
  uniquename.Append(name);

  // Check this!!!
  TVirtualMC::GetMC()->Mixture(imat, uniquename.Data(), a, z, dens * mDensityFactor, nlmat, wmat);
}

void Detector::Medium(Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield,
                      Float_t fieldm, Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil,
                      Float_t stmin, Float_t* ubuf, Int_t nbuf) const
{
  TString uniquename = GetName();
  uniquename.Append("_");
  uniquename.Append(name);

  // Check this!!!
  TVirtualMC::GetMC()->Medium(numed, uniquename.Data(), nmat, isvol, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
              stmin, ubuf, nbuf);
}

void Detector::Matrix(Int_t& nmat, Float_t theta1, Float_t phi1, Float_t theta2, Float_t phi2,
                      Float_t theta3, Float_t phi3) const
{
  TVirtualMC::GetMC()->Matrix(nmat, theta1, phi1, theta2, phi2, theta3, phi3);
}

void Detector::defineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax, Double_t zspan)
{
}

void Detector::setNumberOfWrapperVolumes(Int_t n)
{
}

void Detector::defineLayer(const Int_t nlay, const double phi0, const Double_t r,
                           const Double_t zlen, const Int_t nladd, const Int_t nmod,
                           const Double_t lthick, const Double_t dthick, const UInt_t dettypeID,
                           const Int_t buildLevel)
{
}

void Detector::defineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen, Int_t nladd,
                                Int_t nmod, Double_t width, Double_t tilt, Double_t lthick,
                                Double_t dthick, UInt_t dettypeID, Int_t buildLevel)
{
}
ClassImp(AliceO2::Base::Detector)

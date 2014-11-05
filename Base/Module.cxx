/// \file Module.cxx
/// \brief Implementation of the Module class

#include "Module.h"
#include <Riostream.h>
#include <TVirtualMC.h>

using std::endl;
using std::cout;
using std::fstream;
using std::ios;
using std::ostream;
using namespace AliceO2::Base;

ClassImp(AliceO2::Base::Module)

Float_t Module::mDensityFactor = 1.0;

Module::Module()
    : FairModule()
{
}

Module::Module(const char* name, const char* title, Bool_t Active)
    : FairModule(name, title, Active)
{
}

Module::~Module()
{
}

void Module::Material(Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens,
                      Float_t radl, Float_t absl, Float_t* buf, Int_t nwbuf) const
{
  TString uniquename = GetName();
  uniquename.Append("_");
  uniquename.Append(name);

  // Check this!!!
  gMC->Material(imat, uniquename.Data(), a, z, dens * mDensityFactor, radl, absl, buf, nwbuf);
}

void Module::Mixture(Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens,
                     Int_t nlmat, Float_t* wmat) const
{
  TString uniquename = GetName();
  uniquename.Append("_");
  uniquename.Append(name);

  // Check this!!!
  gMC->Mixture(imat, uniquename.Data(), a, z, dens * mDensityFactor, nlmat, wmat);
}

void Module::Medium(Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield,
                    Float_t fieldm, Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil,
                    Float_t stmin, Float_t* ubuf, Int_t nbuf) const
{
  TString uniquename = GetName();
  uniquename.Append("_");
  uniquename.Append(name);

  // Check this!!!
  gMC->Medium(numed, uniquename.Data(), nmat, isvol, ifield, fieldm, tmaxfd, stemax, deemax, epsil,
              stmin, ubuf, nbuf);
}

void Module::Matrix(Int_t& nmat, Float_t theta1, Float_t phi1, Float_t theta2, Float_t phi2,
                    Float_t theta3, Float_t phi3) const
{
  gMC->Matrix(nmat, theta1, phi1, theta2, phi2, theta3, phi3);
}

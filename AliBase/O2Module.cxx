#include "O2Module.h"
#include <Riostream.h>
#include <TVirtualMC.h>

using std::endl;
using std::cout;
using std::fstream;
using std::ios;
using std::ostream;

ClassImp(O2Module)
 
Float_t O2Module::fgDensityFactor = 1.0;
 
//_______________________________________________________________________
O2Module::O2Module():
FairModule()
{
  //
  // Default constructor for the O2Module class
  //
}
 
//_______________________________________________________________________
O2Module::O2Module(const char* name,const char *title, Bool_t Active):
  FairModule(name,title, Active)
{
}
 
//_______________________________________________________________________
O2Module::~O2Module()
{

} 

void O2Module::AliMaterial(Int_t imat, const char* name, Float_t a, 
                            Float_t z, Float_t dens, Float_t radl,
                            Float_t absl, Float_t *buf, Int_t nwbuf) const
{
    TString uniquename = GetName();
    uniquename.Append("_");
    uniquename.Append(name);

    //Check this!!!
    gMC->Material(imat, uniquename.Data(), a, z, dens * fgDensityFactor, radl, absl, buf, nwbuf);
}

void O2Module::AliMixture(Int_t imat, const char *name, Float_t *a,
                           Float_t *z, Float_t dens, Int_t nlmat,
                           Float_t *wmat) const
{
    TString uniquename = GetName();
    uniquename.Append("_");
    uniquename.Append(name);
  
    //Check this!!!
    gMC->Mixture(imat, uniquename.Data(), a, z, dens * fgDensityFactor, nlmat, wmat);
}
 
void O2Module::AliMedium(Int_t numed, const char *name, Int_t nmat,
                          Int_t isvol, Int_t ifield, Float_t fieldm,
                          Float_t tmaxfd, Float_t stemax, Float_t deemax,
                          Float_t epsil, Float_t stmin, Float_t *ubuf,
                          Int_t nbuf) const
{
    TString uniquename = GetName();
    uniquename.Append("_");
    uniquename.Append(name);
    
    //Check this!!!
    gMC->Medium(numed, uniquename.Data(), nmat, isvol, ifield,
                fieldm, tmaxfd, stemax, deemax, epsil, stmin, ubuf, nbuf);
}
 
void O2Module::AliMatrix(Int_t &nmat, Float_t theta1, Float_t phi1,
                          Float_t theta2, Float_t phi2, Float_t theta3,
                          Float_t phi3) const
{
  // 
  // Define a rotation matrix. Angles are in degrees.
  //
  // nmat        on output contains the number assigned to the rotation matrix
  // theta1      polar angle for axis I
  // phi1        azimuthal angle for axis I
  // theta2      polar angle for axis II
  // phi2        azimuthal angle for axis II
  // theta3      polar angle for axis III
  // phi3        azimuthal angle for axis III
  //
  gMC->Matrix(nmat, theta1, phi1, theta2, phi2, theta3, phi3); 
}

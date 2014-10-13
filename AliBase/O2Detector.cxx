#include "O2Detector.h"

#include <Riostream.h>
#include <TVirtualMC.h>


using std::endl;
using std::cout;
using std::fstream;
using std::ios;
using std::ostream;

ClassImp(O2Detector)
 
Float_t O2Detector::fgDensityFactor = 1.0;
 
//_______________________________________________________________________
O2Detector::O2Detector():
FairDetector()
{
  //
  // Default constructor for the O2Detector class
  //
}
 
//_______________________________________________________________________
O2Detector::O2Detector(const char* name,Bool_t Active, Int_t DetId):
  FairDetector(name,Active,DetId)
{
}
 
//_______________________________________________________________________
O2Detector::~O2Detector()
{

} 

//_______________________________________________________________________
void O2Detector::AliMaterial(Int_t imat, const char* name, Float_t a, 
                            Float_t z, Float_t dens, Float_t radl,
                            Float_t absl, Float_t *buf, Int_t nwbuf) const
{
    TString uniquename = GetName();
    uniquename.Append("_");
    uniquename.Append(name);
    
     //Check this!!!
     gMC->Material(imat, uniquename.Data(), a, z, dens * fgDensityFactor, radl, absl, buf, nwbuf);
}


//_______________________________________________________________________
void O2Detector::AliMixture(Int_t imat, const char *name, Float_t *a,
                           Float_t *z, Float_t dens, Int_t nlmat,
                           Float_t *wmat) const
{
    TString uniquename = GetName();
    uniquename.Append("_");
    uniquename.Append(name);
    
    //Check this!!!
    gMC->Mixture(imat, uniquename.Data(), a, z, dens * fgDensityFactor, nlmat, wmat);
}
 
//_______________________________________________________________________
void O2Detector::AliMedium(Int_t numed, const char *name, Int_t nmat,
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
 
//_______________________________________________________________________
void O2Detector::AliMatrix(Int_t &nmat, Float_t theta1, Float_t phi1,
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

void O2Detector::DefineWrapVolume(Int_t id, Double_t rmin,Double_t rmax, Double_t zspan)
{
}

void O2Detector::SetNWrapVolumes(Int_t n)
{
}

void O2Detector::DefineLayer(const Int_t nlay, const double phi0, const Double_t r,
			    const Double_t zlen, const Int_t nladd,
			    const Int_t nmod, const Double_t lthick,
			    const Double_t dthick, const UInt_t dettypeID,
			    const Int_t buildLevel)
{
}

void O2Detector::DefineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen, Int_t nladd,
				 Int_t nmod, Double_t width, Double_t tilt,
				 Double_t lthick,Double_t dthick,
				 UInt_t dettypeID, Int_t buildLevel)
{
}

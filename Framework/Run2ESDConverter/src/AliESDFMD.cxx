/**************************************************************************
 * Copyright(c) 2004, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/* $Id$ */

//____________________________________________________________________
//
//  ESD information from the FMD 
//  Contains information on:
//	Charged particle multiplicty per strip (rough estimate)
//	Psuedo-rapdity per strip
//  Latest changes by Christian Holm Christensen
//
#include "AliESDFMD.h"		// ALIFMDESD_H
#include "AliLog.h"		// ALILOG_H
#include "Riostream.h"		// ROOT_Riostream
#include <TMath.h>

//____________________________________________________________________
ClassImp(AliESDFMD)
#if 0
  ; // This is here to keep Emacs for indenting the next line
#endif


//____________________________________________________________________
namespace {
  // Private implementation of a AliFMDMap::ForOne to use in
  // forwarding to AliESDFMD::ForOne
  class ForMultiplicity : public AliFMDMap::ForOne 
  {
  public:
    ForMultiplicity(const AliESDFMD& o, AliESDFMD::ForOne& a)
      : fObject(o), fAlgo(a) 
    {}
    Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
		      Float_t m)
    {
      Float_t e = fObject.Eta(d, r, 0, t);
      return fAlgo.operator()(d, r, s, t, m, e);
    }
    Bool_t operator()(UShort_t, Char_t, UShort_t, UShort_t, Int_t)
    {
      return kTRUE;
    }
    Bool_t operator()(UShort_t, Char_t, UShort_t, UShort_t, UShort_t)
    {
      return kTRUE;
    }
    Bool_t operator()(UShort_t, Char_t, UShort_t, UShort_t, Bool_t)
    {
      return kTRUE;
    }
  protected:
    const AliESDFMD&   fObject;
    AliESDFMD::ForOne& fAlgo;
  };

  // Private implementation of AliESDFMD::ForOne to print an 
  // object 
  class Printer : public AliESDFMD::ForOne
  {
  public:
    Printer() : fOldD(0), fOldR('-'), fOldS(1024) {}
    Bool_t operator()(UShort_t d, Char_t r, UShort_t s, UShort_t t, 
		      Float_t m, Float_t e)
    {
      if (d != fOldD) { 
	if (fOldD != 0) printf("\n");
	fOldD = d;
	fOldR = '-';
	printf("FMD%d", fOldD);
      }
      if (r != fOldR) { 
	fOldR = r;
	fOldS = 1024;
	printf("\n %s ring", (r == 'I' ? "Inner" : "Outer"));
      }
      if (s != fOldS) { 
	fOldS = s;
	printf("\n  Sector %d", fOldS);
      }
      if (t % 4 == 0)                   printf("\n   %3d-%3d ", t, t+3);
      if (m == AliESDFMD::kInvalidMult) printf("------/");
      else                              printf("%6.3f/", m);
      if (e == AliESDFMD::kInvalidEta)  printf("------ ");
      else                              printf("%6.3f ", e);

      return kTRUE;
    }
  private:
    UShort_t fOldD;
    Char_t   fOldR;
    UShort_t fOldS;
  };
}

//____________________________________________________________________
AliESDFMD::AliESDFMD()
  : fMultiplicity(0, 0, 0, 0),
    fEta(AliFMDFloatMap::kMaxDetectors, 
	 AliFMDFloatMap::kMaxRings, 
	 1,
	 AliFMDFloatMap::kMaxStrips), 
    fNoiseFactor(0),
    fAngleCorrected(kFALSE)
{
  // Default CTOR
}
  
//____________________________________________________________________
AliESDFMD::AliESDFMD(const AliESDFMD& other)
  : TObject(other), 
    fMultiplicity(other.fMultiplicity),
    fEta(other.fEta),
    fNoiseFactor(other.fNoiseFactor),
    fAngleCorrected(other.fAngleCorrected)
{
  // Default CTOR
}

//____________________________________________________________________
AliESDFMD& 
AliESDFMD::operator=(const AliESDFMD& other)
{
  // Default CTOR
  if(this == &other) return *this;

  TObject::operator=(other);
  fMultiplicity   = other.fMultiplicity;
  fEta            = other.fEta;

  // These two lines were missing prior to version 4 of this class 
  fNoiseFactor    = other.fNoiseFactor;
  fAngleCorrected = other.fAngleCorrected;

  return *this;
}

//____________________________________________________________________
void 
AliESDFMD::Copy(TObject &obj) const
{
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDFMD *robj = dynamic_cast<AliESDFMD*>(&obj);
  if(!robj)return; // not an AliESDFMD
  *robj = *this;
}

//____________________________________________________________________
void
AliESDFMD::CheckNeedUShort(TFile* file) 
{
  fMultiplicity.CheckNeedUShort(file);
  fEta.CheckNeedUShort(file);
}

//____________________________________________________________________
void
AliESDFMD::Clear(Option_t* )
{
  fMultiplicity.Reset(kInvalidMult);
  fEta.Reset(kInvalidEta);
}


//____________________________________________________________________
Float_t
AliESDFMD::Multiplicity(UShort_t detector, Char_t ring, UShort_t sector, 
			UShort_t strip) const
{
  // Return rough estimate of charged particle multiplicity in the
  // strip FMD<detector><ring>[<sector>,<strip>]. 
  // 
  // Note, that this should at most be interpreted as the sum
  // multiplicity of secondaries and primaries. 
  return fMultiplicity(detector, ring, sector, strip);
}

//____________________________________________________________________
Float_t
AliESDFMD::Eta(UShort_t detector, Char_t ring, UShort_t /* sector */, 
	       UShort_t strip) const
{
  // Return pseudo-rapidity of the strip
  // FMD<detector><ring>[<sector>,<strip>].  (actually, the sector
  // argument is ignored, as it is assumed that the primary vertex is
  // a (x,y) = (0,0), and that the modules are aligned with a
  // precision better than 2 degrees in the azimuthal angle). 
  // 
  return fEta(detector, ring, 0, strip);
}

//____________________________________________________________________
Float_t
AliESDFMD::Phi(UShort_t detector, Char_t ring, UShort_t sector, UShort_t) const
{
  // Return azimuthal angle (in degrees) of the strip
  // FMD<detector><ring>[<sector>,<strip>].  
  // 
  Float_t baseAng = (detector == 1 ? 90 : 
		     detector == 2 ?  0 : 180);
  Float_t dAng    = ((detector == 3 ? -1 : 1) * 360 / 
		     (ring == 'I' || ring == 'i' ? 
		      AliFMDMap::kNSectorInner : 
		      AliFMDMap::kNSectorOuter));
  Float_t ret =  baseAng + dAng * (sector + .5);
  if (ret > 360) ret -= 360;
  if (ret <   0) ret += 360;
  return ret;
  
}

//____________________________________________________________________
Float_t
AliESDFMD::Theta(UShort_t detector, Char_t ring, UShort_t, UShort_t strip) const
{
  // Return polar angle from beam line (in degrees) of the strip
  // FMD<detector><ring>[<sector>,<strip>].  
  // 
  // This value is calculated from eta and therefor takes into account
  // the Z position of the interaction point. 
  Float_t eta   = Eta(detector, ring, 0, strip);
  Float_t theta = TMath::ATan(2 * TMath::Exp(-eta));
  if (theta < 0) theta += TMath::Pi();
  theta *= 180. / TMath::Pi();
  return theta;
}

//____________________________________________________________________
Float_t
AliESDFMD::R(UShort_t, Char_t ring, UShort_t, UShort_t strip) const
{
  // Return radial distance from beam line (in cm) of the strip
  // FMD<detector><ring>[<sector>,<strip>].  
  // 
  
  // Numbers are from AliFMDRing
  Float_t  lR  = (ring == 'I' || ring == 'i' ?  4.522 : 15.4);
  Float_t  hR  = (ring == 'I' || ring == 'i' ? 17.2   : 28.0);
  UShort_t nS  = (ring == 'I' || ring == 'i' ? 
		  AliFMDMap::kNStripInner : 
		  AliFMDMap::kNStripOuter);
  Float_t  dR  = (hR - lR) / nS;
  Float_t  ret = lR + dR * (strip + .5);
  return ret;
  
}

//____________________________________________________________________
void
AliESDFMD::SetMultiplicity(UShort_t detector, Char_t ring, UShort_t sector, 
			   UShort_t strip, Float_t mult)
{
  // Return rough estimate of charged particle multiplicity in the
  // strip FMD<detector><ring>[<sector>,<strip>]. 
  // 
  // Note, that this should at most be interpreted as the sum
  // multiplicity of secondaries and primaries. 
  fMultiplicity(detector, ring, sector, strip) = mult;
}

//____________________________________________________________________
void
AliESDFMD::SetEta(UShort_t detector, Char_t ring, UShort_t /* sector */, 
		  UShort_t strip, Float_t eta)
{
  // Set pseudo-rapidity of the strip
  // FMD<detector><ring>[<sector>,<strip>].  (actually, the sector
  // argument is ignored, as it is assumed that the primary vertex is
  // a (x,y) = (0,0), and that the modules are aligned with a
  // precision better than 2 degrees in the azimuthal angle). 
  // 
  fEta(detector, ring, 0, strip) = eta;
}

//____________________________________________________________________
Bool_t
AliESDFMD::ForEach(AliESDFMD::ForOne& a) const
{
  ForMultiplicity i(*this, a);
  return fMultiplicity.ForEach(i);
}
//____________________________________________________________________
void
AliESDFMD::Print(Option_t* /* option*/) const
{
  // Print all information to standard output. 
  std::cout << "AliESDFMD:" << std::endl;
  Printer p;
  ForEach(p);
  printf("\n");
#if 0
  for (UShort_t det = 1; det <= fMultiplicity.MaxDetectors(); det++) {
    for (UShort_t ir = 0; ir < fMultiplicity.MaxRings(); ir++) {
      Char_t ring = (ir == 0 ? 'I' : 'O');
      std::cout << "FMD" << det << ring << ":" << std::endl;
      for  (UShort_t sec = 0; sec < fMultiplicity.MaxSectors(); sec++) {
	std::cout << " Sector # " << sec << ":" << std::flush;
	for (UShort_t str = 0; str < fMultiplicity.MaxStrips(); str++) {
	  if (str % 6 == 0) std::cout << "\n  " << std::flush;
	  Float_t m = fMultiplicity(det, ring, sec, str);
	  Float_t e = fEta(det, ring, 0, str);
	  if (m == kInvalidMult && e == kInvalidEta) break;
	  if (m == kInvalidMult) std::cout << " ---- ";
	  else                   std::cout << Form("%6.3f", m);
	  std::cout << "/";
	  if (e == kInvalidEta)  std::cout << " ---- ";
	  else                   std::cout << Form("%-6.3f", e);
	  std::cout << " " << std::flush;
	}
	std::cout << std::endl;
      }
    }
  }
#endif
}


//____________________________________________________________________
//
// EOF
//

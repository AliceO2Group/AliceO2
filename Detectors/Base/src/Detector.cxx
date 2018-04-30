// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Detector.cxx
/// \brief Implementation of the Detector class

#include "DetectorsBase/Detector.h"
#include <TVirtualMC.h> // for TVirtualMC, gMC
#include "DetectorsBase/MaterialManager.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Field/MagneticField.h"
#include "TString.h" // for TString

using std::cout;
using std::endl;
using std::fstream;
using std::ios;
using std::ostream;

using namespace o2::Base;
using namespace o2::detectors;

Float_t Detector::mDensityFactor = 1.0;

Detector::Detector() : FairDetector(), mMapMaterial(), mMapMedium() {}
Detector::Detector(const char* name, Bool_t Active)
  : FairDetector(name, Active, DetID(name)), mMapMaterial(), mMapMedium()
{
}

Detector::Detector(const Detector& rhs) : FairDetector(rhs), mMapMaterial(rhs.mMapMaterial), mMapMedium(rhs.mMapMedium)
{
}

Detector::~Detector() = default;

Detector& Detector::operator=(const Detector& rhs)
{
  // check assignment to self
  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  FairDetector::operator=(rhs);

  return *this;
}

void Detector::Material(Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens, Float_t radl, Float_t absl,
                        Float_t* buf, Int_t nwbuf)
{
  auto& mgr = o2::Base::MaterialManager::Instance();
  mgr.Material(GetName(), imat, name, a, z, dens, radl, absl, buf, nwbuf);
}

void Detector::Mixture(Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens, Int_t nlmat, Float_t* wmat)
{
  auto& mgr = o2::Base::MaterialManager::Instance();
  mgr.Mixture(GetName(), imat, name, a, z, dens, nlmat, wmat);
}

void Detector::Medium(Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield, Float_t fieldm,
                      Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil, Float_t stmin, Float_t* ubuf,
                      Int_t nbuf)
{
  auto& mgr = o2::Base::MaterialManager::Instance();
  mgr.Medium(GetName(), numed, name, nmat, isvol, ifield, fieldm, tmaxfd, stemax, deemax, epsil, stmin, ubuf, nbuf);
}

void Detector::SpecialCuts(Int_t numed, const std::initializer_list<std::pair<ECut, Float_t>>& parIDValMap)
{
  auto& mgr = MaterialManager::Instance();
  mgr.SpecialCuts(GetName(), numed, parIDValMap);
}

void Detector::SpecialCut(Int_t numed, ECut parID, Float_t val)
{
  auto& mgr = MaterialManager::Instance();
  mgr.SpecialCut(GetName(), numed, parID, val);
}

void Detector::SpecialProcesses(Int_t numed, const std::initializer_list<std::pair<EProc, int>>& parIDValMap)
{
  auto& mgr = MaterialManager::Instance();
  mgr.SpecialProcesses(GetName(), numed, parIDValMap);
}

void Detector::SpecialProcess(Int_t numed, EProc parID, int val)
{
  auto& mgr = MaterialManager::Instance();
  mgr.SpecialProcess(GetName(), numed, parID, val);
}

void Detector::Matrix(Int_t& nmat, Float_t theta1, Float_t phi1, Float_t theta2, Float_t phi2, Float_t theta3,
                      Float_t phi3) const
{
  TVirtualMC::GetMC()->Matrix(nmat, theta1, phi1, theta2, phi2, theta3, phi3);
}

void Detector::defineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax, Double_t zspan) {}
void Detector::setNumberOfWrapperVolumes(Int_t n) {}
void Detector::defineLayer(const Int_t nlay, const double phi0, const Double_t r, const Int_t nladd, const Int_t nmod,
                           const Double_t lthick, const Double_t dthick, const UInt_t dettypeID, const Int_t buildLevel)
{
}

void Detector::defineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Int_t nladd, Int_t nmod, Double_t width,
                                Double_t tilt, Double_t lthick, Double_t dthick, UInt_t dettypeID, Int_t buildLevel)
{
}

void Detector::initFieldTrackingParams(int& integration, float& maxfield)
{
  // set reasonable default values
  integration = 2;
  maxfield = 10;
  auto vmc = TVirtualMC::GetMC();
  if (vmc) {
    auto field = vmc->GetMagField();
    // see if we can query the o2 field
    if (auto o2field = dynamic_cast<o2::field::MagneticField*>(field)) {
      integration = o2field->Integral(); // default integration method?
      maxfield = o2field->Max();
      return;
    }
  }
  LOG(INFO) << "No magnetic field found; using default tracking values " << integration << " " << maxfield
            << " to initialize media\n";
}

TClonesArray* Detector::GetCollection(int) const
{
  LOG(WARNING) << "GetCollection interface no longer supported" << FairLogger::endl;
  LOG(WARNING) << "Use the GetHits function on invidiual detectors" << FairLogger::endl;
  return nullptr;
}

void Detector::addAlignableVolumes() const
{
  LOG(WARNING) << "Alignable volumes are not yet defined for " << GetName() << FairLogger::endl;
}

ClassImp(o2::Base::Detector)

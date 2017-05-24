/// \file SimulationAlpide.cxx
/// \brief Simulation of the ALIPIDE chip response

#include <TRandom.h>
#include <TLorentzVector.h>
#include <TClonesArray.h>
#include <TSeqCollection.h>

#include "FairLogger.h"

#include "ITSMFTBase/Digit.h"
#include "ITSMFTBase/SegmentationPixel.h"
#include "ITSMFTSimulation/SimulationAlpide.h"
#include "ITSMFTSimulation/SimuClusterShaper.h"
#include "ITSMFTSimulation/Point.h"
#include "ITSMFTSimulation/DigitContainer.h"

using namespace o2::ITSMFT;

//______________________________________________________________________
SimulationAlpide::SimulationAlpide():
Chip() {
  for (Int_t i=0; i<NumberOfParameters; i++) mParam[i]=0.;
}

//______________________________________________________________________
SimulationAlpide::SimulationAlpide(Double_t par[NumberOfParameters], Int_t n, const TGeoHMatrix *m):
Chip(n,m) {
  for (Int_t i=0; i<NumberOfParameters; i++) mParam[i]=par[i];
}


//______________________________________________________________________
SimulationAlpide::SimulationAlpide(const SimulationAlpide &s):
Chip(s) {
  for (Int_t i=0; i<NumberOfParameters; i++) mParam[i]=s.mParam[i];
}


//______________________________________________________________________
Double_t SimulationAlpide::betaGammaFunction(Double_t Par0, Double_t Par1, Double_t Par2, Double_t x) const {
  Double_t y = Par0*((1+TMath::Power(x, 2))/TMath::Power(x, 2))*(0.5*TMath::Log(Par1*TMath::Power(x, 2)) - (TMath::Power(x, 2)/(1+TMath::Power(x, 2))) - Par2*TMath::Log(x));
  return std::max(0.35, y);
}


//______________________________________________________________________
Double_t SimulationAlpide::getACSFromBetaGamma(Double_t x, Double_t theta) const {
  Double_t evalX = betaGammaFunction(mParam[ACSFromBGPar0], mParam[ACSFromBGPar1], mParam[ACSFromBGPar2], x);
  return evalX/fabs(cos(theta));
}


//______________________________________________________________________
Double_t SimulationAlpide::gaussian2D(Double_t sigma, Double_t offc, Double_t x, Double_t y) const {
  return (offc-1)*(1-TMath::Gaus(x,0,sigma)*TMath::Gaus(y,0,sigma))+1;
}


//______________________________________________________________________
Int_t SimulationAlpide::getPixelPositionResponse(const SegmentationPixel *seg, Int_t idPadX, Int_t idPadZ, Float_t locx, Float_t locz, Double_t acs) const {
  Float_t centerX, centerZ;
  seg->detectorToLocal(idPadX, idPadZ, centerX, centerZ);

  Double_t Dx = locx-centerX;
  Double_t Dy = locz-centerZ;
  Double_t sigma = 0.001; // = 10 um
  Double_t offc  = acs; // WARNING: this is just temporary! (a function for this is ready but need further testing)
  Int_t cs = (Int_t) round(gaussian2D(sigma, offc, Dx, Dy));
  return cs;
}


//______________________________________________________________________
Int_t SimulationAlpide::sampleCSFromLandau(Double_t mpv, Double_t w) const {
  Double_t x = std::max(1., gRandom->Landau(mpv, w));
  Int_t cs = (Int_t) round(x);
  return cs;
}


//______________________________________________________________________
Double_t SimulationAlpide::computeIncidenceAngle(TLorentzVector dir) const {
  Double_t glob[3], loc[3];
  glob[0] = dir.Px()/dir.P();
  glob[1] = dir.Py()/dir.P();
  glob[2] = dir.Pz()/dir.P();

  Chip::globalToLocalVector(glob, loc);

  TVector3 pdirection(loc[0], loc[1], loc[2]);
  TVector3 normal(0., -1., 0.);

  return pdirection.Angle(normal);
}


//______________________________________________________________________
void SimulationAlpide::addNoise(Double_t mean, const SegmentationPixel* seg, DigitContainer *digitContainer) {
  UInt_t row = 0;
  UInt_t col = 0;
  Int_t nhits = 0;
  Int_t chipId = Chip::GetChipIndex();
  nhits = gRandom->Poisson(mean);
  for (Int_t i = 0; i < nhits; ++i) {
    row = gRandom->Integer(seg->getNumberOfRows());
    col = gRandom->Integer(seg->getNumberOfColumns());
    Digit *noiseD = digitContainer->addDigit(chipId, row, col, 0., 0.);
    noiseD->setLabel(0, -1);
  }
}


//______________________________________________________________________
void SimulationAlpide::generateClusters(const SegmentationPixel *seg, DigitContainer *digitContainer) {
  Int_t nhits = Chip::GetNumberOfPoints();

  // Add noise to the chip
  addNoise(5, seg, digitContainer);

  if (nhits <= 0) return;

  for (Int_t h = 0; h < nhits; ++h) {
    Double_t x0, x1, y0, y1, z0, z1, tof, de;
    if (!Chip::LineSegmentLocal(h, x0, x1, y0, y1, z0, z1, tof, de)) continue;

    // To local coordinates
    Float_t x = x0 + 0.5*x1;
    Float_t y = y0 + 0.5*y1;
    Float_t z = z0 + 0.5*z1;

    const Point *hit = Chip::GetPointAt(h);
    TLorentzVector trackP4;
    trackP4.SetPxPyPzE(hit->GetPx(), hit->GetPy(), hit->GetPz(), hit->GetTotalEnergy());
    Double_t beta = std::min(0.99999, trackP4.Beta());
    Double_t bgamma = beta / sqrt(1 - pow(beta, 2));
    if (bgamma < 0.001) continue;
    Double_t theta = computeIncidenceAngle(trackP4);

    // Get the pixel ID
    Int_t ix, iz;
    if (!seg->localToDetector(x, z, ix, iz)) continue;

    Double_t acs = getACSFromBetaGamma(bgamma, theta);
    UInt_t cs = getPixelPositionResponse(seg, ix, iz, x, z, acs);
    //cs = 3; // uncomment to set the cluster size manually

    // Create the shape
    std::vector<UInt_t> cshape;
    auto csManager = std::make_unique<SimuClusterShaper>(cs);
    csManager->SetFireCenter(true);

    csManager->SetHit(ix, iz, x, z, seg);
    csManager->FillClusterSorted();

    cs = csManager->GetCS();
    csManager->GetShape(cshape);
    UInt_t nrows = csManager->GetNRows();
    UInt_t ncols = csManager->GetNCols();
    Int_t cx = csManager->GetCenterC();
    Int_t cz = csManager->GetCenterR();

    LOG(DEBUG) << "_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << FairLogger::endl;
    LOG(DEBUG) << "_/_/_/ pALPIDE debug  _/_/_/" << FairLogger::endl;
    LOG(DEBUG) << "_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << FairLogger::endl;
    LOG(DEBUG) << " Beta*Gamma: " << bgamma << FairLogger::endl;
    LOG(DEBUG) << "        ACS: " << acs << FairLogger::endl;
    LOG(DEBUG) << "         CS: " <<  cs << FairLogger::endl;
    LOG(DEBUG) << "      Shape: " << ClusterShape::ShapeSting(cshape).c_str() << FairLogger::endl;
    LOG(DEBUG) << "     Center: " << cx << ' ' << cz << FairLogger::endl;
    LOG(DEBUG) << "_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << FairLogger::endl;

    for (Int_t ipix = 0; ipix < cs; ++ipix) {
      Int_t r = (Int_t) cshape[ipix] / nrows;
      Int_t c = (Int_t) cshape[ipix] % nrows;
      Int_t nx = ix - cx + c;
      if (nx<0) continue;
      if (nx>=seg->getNumberOfRows()) continue;
      Int_t nz = iz - cz + r;
      if (nz<0) continue;
      if (nz>=seg->getNumberOfColumns()) continue;
      Int_t chipID = hit->GetDetectorID();
      Double_t charge = hit->GetEnergyLoss();
      Digit *digit = digitContainer->addDigit(chipID, nx, nz, charge, hit->GetTime());
      digit->setLabel(0, hit->GetTrackID());
    }
  }
}

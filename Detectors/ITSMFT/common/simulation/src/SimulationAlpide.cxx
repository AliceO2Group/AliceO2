/// \file SimulationAlpide.cxx
/// \brief Simulation of the ALIPIDE chip response

#include <TF1.h>
#include <TF2.h>
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

ClassImp(o2::ITSMFT::SimulationAlpide)

using namespace o2::ITSMFT;

//______________________________________________________________________
SimulationAlpide::SimulationAlpide():
mSeg(nullptr),
mChip(nullptr)
{
  for (Int_t i=0; i<NumberOfParameters; i++) mParam[i]=0.;
}

//______________________________________________________________________
SimulationAlpide::SimulationAlpide
(Double_t par[NumberOfParameters], SegmentationPixel *seg, Chip *chip):
mSeg(seg),
mChip(chip)
{
  for (Int_t i=0; i<NumberOfParameters; i++) mParam[i]=par[i];
}

//______________________________________________________________________
SimulationAlpide::SimulationAlpide(const SimulationAlpide &s):
mSeg(s.mSeg),
mChip(s.mChip)
{
  for (Int_t i=0; i<NumberOfParameters; i++) mParam[i]=s.mParam[i];
}

//______________________________________________________________________
Double_t SimulationAlpide::getACSFromBetaGamma(Double_t x, Double_t theta) const {
  auto *acs = new TF1("acs", "[0]*((1+TMath::Power(x, 2))/TMath::Power(x, 2))*(0.5*TMath::Log([1]*TMath::Power(x, 2)) - (TMath::Power(x, 2)/(1+TMath::Power(x, 2))) - [2]*TMath::Log(x))", 0, 10000);
  acs->SetParameter(0, mParam[ACSFromBGPar0]);
  acs->SetParameter(1, mParam[ACSFromBGPar1]);
  acs->SetParameter(2, mParam[ACSFromBGPar2]);
  Double_t val = acs->Eval(x)/fabs(cos(theta));
  delete acs;
  return val;
}


//______________________________________________________________________
Int_t SimulationAlpide::getPixelPositionResponse(Int_t idPadX, Int_t idPadZ, Float_t locx, Float_t locz, Double_t acs) const {
  Float_t centerX, centerZ;
  mSeg->detectorToLocal(idPadX, idPadZ, centerX, centerZ);

  Double_t Dx = locx-centerX;
  Double_t Dy = locz-centerZ;
  Double_t sigma = 0.001; // = 10 um
  Double_t offc  = acs; // WARNING: this is just temporary! (a function for this is ready but need further testing)

  auto *respf = new TF2("respf", "([1]-1)*(1-TMath::Gaus(x,0,[0])*TMath::Gaus(y,0,[0]))+1",
  -mSeg->cellSizeX()/2, mSeg->cellSizeX()/2, -mSeg->cellSizeZ(0)/2, mSeg->cellSizeZ(0)/2);
  respf->SetParameter(0, sigma);
  respf->SetParameter(1, offc);
  Int_t cs = (Int_t) round(respf->Eval(Dx, Dy));
  delete respf;
  return cs;
}


//______________________________________________________________________
Int_t SimulationAlpide::sampleCSFromLandau(Double_t mpv, Double_t w) const {
  auto *landauDistr = new TF1("landauDistr","TMath::Landau(x,[0],[1])", 0, 20);
  landauDistr->SetParameter(0, mpv);
  landauDistr->SetParameter(1, w);

  // Generation according to the Landau distribution defined above
  Double_t fmax = landauDistr->GetMaximum();
  Double_t x1 = gRandom->Uniform(0, 20);
  Double_t y1 = gRandom->Uniform(0, fmax);
  while (y1 > landauDistr->Eval(x1)) {
    x1 = gRandom->Uniform(0, 20);
    y1 = gRandom->Uniform(0, fmax);
  }
  Int_t cs = (Int_t) round(x1);
  delete landauDistr;
  return cs;
}


//______________________________________________________________________
Double_t SimulationAlpide::computeIncidenceAngle(TLorentzVector dir) const {
  Double_t glob[3], loc[3];
  glob[0] = dir.Px()/dir.P();
  glob[1] = dir.Py()/dir.P();
  glob[2] = dir.Pz()/dir.P();

  mChip->globalToLocalVector(glob, loc);

  TVector3 pdirection(loc[0], loc[1], loc[2]);
  TVector3 normal(0., -1., 0.);

  return pdirection.Angle(normal);
}


//______________________________________________________________________
void SimulationAlpide::generateClusters(DigitContainer *digitContainer) {
  Int_t nhits = mChip->GetNumberOfPoints();
  if (nhits <= 0) return;

  for (Int_t h = 0; h < nhits; ++h) {
    Double_t x0, x1, y0, y1, z0, z1, tof, de;
    if (!mChip->LineSegmentLocal(h, x0, x1, y0, y1, z0, z1, tof, de)) continue;

    // To local coordinates
    Float_t x = x0 + 0.5*x1;
    Float_t y = y0 + 0.5*y1;
    Float_t z = z0 + 0.5*z1;

    const Point *hit = mChip->GetPointAt(h);
    TLorentzVector trackP4;
    trackP4.SetPxPyPzE(hit->GetPx(), hit->GetPy(), hit->GetPz(), hit->GetTotalEnergy());
    Double_t beta = std::min(0.99999, trackP4.Beta());
    Double_t bgamma = beta / sqrt(1 - pow(beta, 2));
    if (bgamma < 0.001) continue;
    Double_t theta = computeIncidenceAngle(trackP4);

    // Get the pixel ID
    Int_t ix, iz;
    if (!mSeg->localToDetector(x, z, ix, iz)) continue;

    Double_t acs = getACSFromBetaGamma(bgamma, theta);
    UInt_t cs = getPixelPositionResponse(ix, iz, x, z, acs);
    //cs = 3; // uncomment to set the cluster size manually

    // Create the shape
    std::vector<UInt_t> cshape;
    auto *csManager = new SimuClusterShaper(cs);
    csManager->SetFireCenter(true);
    //csManager->FillClusterRandomly();

    csManager->SetHit(ix, iz, x, z, mSeg);
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
        if (nx>=mSeg->getNumberOfRows()) continue;
      Int_t nz = iz - cz + r;
        if (nz<0) continue;
        if (nz>=mSeg->getNumberOfColumns()) continue;
      Int_t chipID = hit->GetDetectorID();
      Double_t charge = hit->GetEnergyLoss();
      Digit *digit = digitContainer->addDigit(chipID, nx, nz, charge, hit->GetTime());
      digit->setLabel(0, hit->GetTrackID());
    }

    delete csManager;
  }
}

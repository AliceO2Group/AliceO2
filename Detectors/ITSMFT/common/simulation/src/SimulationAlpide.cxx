/// \file SimulationAlpide.cxx
/// \brief Simulation of the ALIPIDE chip response

#include <TF1.h>
#include <TF2.h>
#include <TRandom.h>
#include <TLorentzVector.h>
#include <TClonesArray.h>
#include <TSeqCollection.h>

#include "FairLogger.h"

#include "ITSMFTBase/SDigit.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTBase/SegmentationPixel.h"
#include "ITSMFTSimulation/SimulationAlpide.h"
#include "ITSMFTSimulation/SimuClusterShaper.h"
#include "ITSMFTSimulation/Point.h"


ClassImp(o2::ITSMFT::SimulationAlpide)

using namespace o2::ITSMFT;

//______________________________________________________________________
SimulationAlpide::SimulationAlpide():
mSeg(nullptr),
mSensMap(nullptr),
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
  mSensMap=new SensMap("o2::ITSMFT::SDigit",
  seg->getNumberOfColumns(), seg->getNumberOfRows());
}

//______________________________________________________________________
SimulationAlpide::SimulationAlpide(const SimulationAlpide &s):
mSeg(s.mSeg),
mChip(s.mChip)
{
  for (Int_t i=0; i<NumberOfParameters; i++) mParam[i]=s.mParam[i];
  mSensMap=new SensMap(*(s.mSensMap));
}

//______________________________________________________________________
SimulationAlpide::~SimulationAlpide() {
  if (mSensMap) mSensMap->Clear();
  delete mSensMap;
}

//______________________________________________________________________
void SimulationAlpide::Init
(Double_t par[NumberOfParameters], SegmentationPixel *seg, Chip *chip)
{
  for (Int_t i=0; i<NumberOfParameters; i++) mParam[i]=par[i];
  mSeg=seg;
  mChip=chip;
  mSensMap=new SensMap("o2::ITSMFT::SDigit",
  seg->getNumberOfColumns(), seg->getNumberOfRows());
}

//______________________________________________________________________
void SimulationAlpide::SDigitiseChip(TClonesArray *sdarray) {
  if (mChip->GetNumberOfPoints()) GenerateCluster();
  if (!mSensMap->getEntries()) return;
  WriteSDigits(sdarray);
  mSensMap->Clear();
}


//______________________________________________________________________
void SimulationAlpide::FrompListToDigits(TClonesArray *detDigits) {
  int nsd = mSensMap->getEntries();
  if (!nsd) return; // nothing to digitize

  UInt_t row,col;
  Int_t iCycle, modId = mChip->GetChipIndex();
  static Digit dig;

  for (int i = 0; i < nsd; ++i) {
    SDigit* sd = (SDigit*) mSensMap->At(i); // ordered in index
    if (mSensMap->isDisabled(sd)) continue;

    mSensMap->getMapIndex(sd->GetUniqueID(),col,row,iCycle);
    dig.setPixelIndex(row,col);
    dig.setChipIndex(modId);

    //dig.SetROCycle(iCycle);
    dig.setCharge(sd->getSumSignal());
    for (Int_t j=0; j<3; j++) dig.setLabel(j,sd->getTrack(j));

    TClonesArray &ldigits = *detDigits;
    int nd = ldigits.GetEntriesFast();
    new (ldigits[nd]) Digit(dig);
  }
}


//______________________________________________________________________
void SimulationAlpide::WriteSDigits(TClonesArray *sdarray) {
  //  This function adds each S-Digit to pList
  int nsd = mSensMap->getEntries();

  for (int i = 0; i < nsd; ++i) {
    SDigit* sd = (SDigit*)mSensMap->At(i); // ordered in index
    if (mSensMap->isDisabled(sd)) continue;
    new ((*sdarray)[sdarray->GetEntriesFast()]) SDigit(*sd);
  }
  return;
}


//______________________________________________________________________
Bool_t SimulationAlpide::AddSDigitsToChip(TSeqCollection *pItemArr, Int_t mask) {
  //    pItemArr  Array of AliITSpListItems (SDigits).
  //    mask    Track number off set value
  Int_t nItems = pItemArr->GetEntries();

  for( Int_t i=0; i<nItems; i++ ) {
    SDigit * pItem = (SDigit *)(pItemArr->At( i ));
    if(pItem->getChip() != int(mChip->GetChipIndex()) )
    LOG(FATAL) << "SDigits chip " << pItem->getChip() << " != current chip " << mChip->GetChipIndex() << ": exit" << FairLogger::endl;

    SDigit* oldItem = (SDigit*)mSensMap->getItem(pItem);
    if (!oldItem) oldItem = (SDigit*)mSensMap->registerItem( new(mSensMap->getFree()) SDigit(*pItem) );
  }
  return true;
}


//______________________________________________________________________
void SimulationAlpide::FinishSDigitiseChip(TClonesArray *detDigits) {
  //  This function calls SDigitsToDigits which creates Digits from SDigits
  FrompListToDigits(detDigits);
  clearSimulation();
}


//______________________________________________________________________
void SimulationAlpide::DigitiseChip(TClonesArray *detDigits) {
  GenerateCluster();
  FinishSDigitiseChip(detDigits);
}


//______________________________________________________________________
Double_t SimulationAlpide::ACSFromBetaGamma(Double_t x, Double_t theta) const {
  auto *acs = new TF1("acs", "[0]*((1+TMath::Power(x, 2))/TMath::Power(x, 2))*(0.5*TMath::Log([1]*TMath::Power(x, 2)) - (TMath::Power(x, 2)/(1+TMath::Power(x, 2))) - [2]*TMath::Log(x))", 0, 10000);
  acs->SetParameter(0, mParam[ACSFromBGPar0]);
  acs->SetParameter(1, mParam[ACSFromBGPar1]);
  acs->SetParameter(2, mParam[ACSFromBGPar2]);
  Double_t val = acs->Eval(x)/fabs(cos(theta));
  delete acs;
  return val;
}


//______________________________________________________________________
Int_t SimulationAlpide::GetPixelPositionResponse(Int_t idPadX, Int_t idPadZ, Float_t locx, Float_t locz, Double_t acs) const {
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
Int_t SimulationAlpide::CSSampleFromLandau(Double_t mpv, Double_t w) const {
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
Double_t SimulationAlpide::ComputeIncidenceAngle(TLorentzVector dir) const {
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
void SimulationAlpide::GenerateCluster() {
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
    Double_t theta = ComputeIncidenceAngle(trackP4);

    // Get the pixel ID
    Int_t ix, iz;
    if (!mSeg->localToDetector(x, z, ix, iz)) continue;

    Double_t acs = ACSFromBetaGamma(bgamma, theta);
    UInt_t cs = GetPixelPositionResponse(ix, iz, x, z, acs);

    // Create the shape
    std::vector<UInt_t> cshape;
    auto *csManager = new SimuClusterShaper(cs);
    csManager->FillClusterRandomly();
    csManager->GetShape(cshape);
    UInt_t nrows = csManager->GetNRows();
    UInt_t ncols = csManager->GetNCols();
    Int_t cx = gRandom->Integer(ncols);
    Int_t cz = gRandom->Integer(nrows);

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
      Int_t nz = iz - cz + r;
      CreateDigi(nz, nx, hit->GetTrackID(), h);
    }

    delete csManager;
  }
}

//______________________________________________________________________
void SimulationAlpide::CreateDigi(UInt_t col, UInt_t row, Int_t track, Int_t hit) {
  UInt_t index = mSensMap->getIndex(col, row, 0);
  UInt_t chip  = mChip->GetChipIndex();

  mSensMap->registerItem(new (mSensMap->getFree()) SDigit(track, hit, chip, index, 0.1, 0));
}

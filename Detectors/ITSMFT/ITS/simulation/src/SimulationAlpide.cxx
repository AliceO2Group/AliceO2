/// \file SimulationAlpide.cxx
/// \brief Simulation of the ALIPIDE chip response

#include <TF1.h>
#include <TF2.h>
#include <TRandom.h>
#include <TLorentzVector.h>
#include <TClonesArray.h>
#include <TSeqCollection.h>

#include "FairLogger.h"

#include "ITSBase/SDigit.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSBase/Digit.h"
#include "ITSBase/SegmentationPixel.h"
#include "ITSSimulation/SimulationAlpide.h"
#include "ITSSimulation/SimuClusterShaper.h"
#include "ITSSimulation/Point.h"


ClassImp(AliceO2::ITS::SimulationAlpide)

using namespace AliceO2::ITS;

//______________________________________________________________________
SimulationAlpide::SimulationAlpide():
fSeg(0),
fSensMap(0),
fChip(0)
{
  for (Int_t i=0; i<NumberOfParameters; i++) fParam[i]=0.;
}

//______________________________________________________________________
SimulationAlpide::SimulationAlpide
(Double_t par[NumberOfParameters], SegmentationPixel *seg, Chip *chip):
fSeg(seg),
fChip(chip)
{
  for (Int_t i=0; i<NumberOfParameters; i++) fParam[i]=par[i];
  fSensMap=new SensMap("AliceO2::ITS::SDigit",
  seg->getNumberOfColumns(), seg->getNumberOfRows());
}

//______________________________________________________________________
SimulationAlpide::SimulationAlpide(const SimulationAlpide &s):
fSeg(s.fSeg),
fChip(s.fChip)
{
  for (Int_t i=0; i<NumberOfParameters; i++) fParam[i]=s.fParam[i];
  fSensMap=new SensMap(*(s.fSensMap));
}

//______________________________________________________________________
SimulationAlpide::~SimulationAlpide() {
  if (fSensMap) fSensMap->Clear();
  delete fSensMap;
}

//______________________________________________________________________
void SimulationAlpide::Init
(Double_t par[NumberOfParameters], SegmentationPixel *seg, Chip *chip)
{
  for (Int_t i=0; i<NumberOfParameters; i++) fParam[i]=par[i];
  fSeg=seg;
  fChip=chip;
  fSensMap=new SensMap("AliceO2::ITS::SDigit",
  seg->getNumberOfColumns(), seg->getNumberOfRows());
}

//______________________________________________________________________
void SimulationAlpide::SDigitiseChip(TClonesArray *sdarray) {
  if (fChip->GetNumberOfPoints()) GenerateCluster();
  if (!fSensMap->getEntries()) return;
  WriteSDigits(sdarray);
  fSensMap->Clear();
}


//______________________________________________________________________
void SimulationAlpide::FrompListToDigits(TClonesArray *detDigits) {
  int nsd = fSensMap->getEntries();
  if (!nsd) return; // nothing to digitize

  UInt_t row,col;
  Int_t iCycle, modId = fChip->GetChipIndex();
  static Digit dig;

  for (int i = 0; i < nsd; ++i) {
    SDigit* sd = (SDigit*) fSensMap->At(i); // ordered in index
    if (fSensMap->isDisabled(sd)) continue;

    fSensMap->getMapIndex(sd->GetUniqueID(),col,row,iCycle);
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
  int nsd = fSensMap->getEntries();

  for (int i = 0; i < nsd; ++i) {
    SDigit* sd = (SDigit*)fSensMap->At(i); // ordered in index
    if (fSensMap->isDisabled(sd)) continue;
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
    if(pItem->getChip() != int(fChip->GetChipIndex()) )
    LOG(FATAL) << "SDigits chip " << pItem->getChip() << " != current chip " << fChip->GetChipIndex() << ": exit" << FairLogger::endl;

    SDigit* oldItem = (SDigit*)fSensMap->getItem(pItem);
    if (!oldItem) oldItem = (SDigit*)fSensMap->registerItem( new(fSensMap->getFree()) SDigit(*pItem) );
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
  TF1 *acs = new TF1("acs", "[0]*((1+TMath::Power(x, 2))/TMath::Power(x, 2))*(0.5*TMath::Log([1]*TMath::Power(x, 2)) - (TMath::Power(x, 2)/(1+TMath::Power(x, 2))) - [2]*TMath::Log(x))", 0, 10000);
  acs->SetParameter(0, fParam[ACSFromBGPar0]);
  acs->SetParameter(1, fParam[ACSFromBGPar1]);
  acs->SetParameter(2, fParam[ACSFromBGPar2]);
  Double_t val = acs->Eval(x)/fabs(cos(theta));
  delete acs;
  return val;
}


//______________________________________________________________________
Int_t SimulationAlpide::GetPixelPositionResponse(Int_t idPadX, Int_t idPadZ, Float_t locx, Float_t locz, Double_t acs) const {
  Float_t centerX, centerZ;
  fSeg->detectorToLocal(idPadX, idPadZ, centerX, centerZ);

  Double_t Dx = locx-centerX;
  Double_t Dy = locz-centerZ;
  Double_t sigma = 0.001; // = 10 um
  Double_t offc  = acs; // WARNING: this is just temporary! (a function for this is ready but need further testing)

  TF2 *respf = new TF2("respf", "([1]-1)*(1-TMath::Gaus(x,0,[0])*TMath::Gaus(y,0,[0]))+1",
  -fSeg->cellSizeX()/2, fSeg->cellSizeX()/2, -fSeg->cellSizeZ(0)/2, fSeg->cellSizeZ(0)/2);
  respf->SetParameter(0, sigma);
  respf->SetParameter(1, offc);
  Int_t cs = (Int_t) round(respf->Eval(Dx, Dy));
  delete respf;
  return cs;
}


//______________________________________________________________________
Int_t SimulationAlpide::CSSampleFromLandau(Double_t mpv, Double_t w) const {
  TF1 *landauDistr = new TF1("landauDistr","TMath::Landau(x,[0],[1])", 0, 20);
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

  GeometryTGeo* geomTG = fChip->GetGeometry();
  geomTG->globalToLocalVector(fChip->GetChipIndex(), glob, loc);

  TVector3 pdirection(loc[0], loc[1], loc[2]);
  TVector3 normal(0., -1., 0.);

  return pdirection.Angle(normal);
}

//______________________________________________________________________
void SimulationAlpide::GenerateCluster() {
  TObjArray *hits = fChip->GetPoints();
  Int_t nhits = hits->GetEntriesFast();
  if (nhits <= 0) return;

  Float_t px, py, pz, x,  y,  z;
  Double_t etot, beta, gamma, acs, theta;
  Double_t tof, x0, x1, y0, y1, z0, z1, el, de;
  UInt_t cs, nrows, ncols;
  Int_t ix, iz, nx, nz, cx, cz, idtrack;

  for (Int_t h=0; h < nhits; ++h) {
    if (!fChip->LineSegmentLocal(h, x0, x1, y0, y1, z0, z1, tof, de)) continue;

    // local coordinates
    x = x0 + 0.5*x1;
    y = y0 + 0.5*y1;
    z = z0 + 0.5*z1;

    Point *hit = static_cast<Point*>(hits->UncheckedAt(h));
    px=hit->GetPx();
    py=hit->GetPy();
    pz=hit->GetPz();
    etot = hit->GetTotalEnergy();
    idtrack=hit->GetTrackID();

    TLorentzVector momen;
    std::vector<UInt_t> cshape;
    momen.SetPxPyPzE(px, py, pz, etot);
    beta = momen.Beta();
    if (beta > 0.99999) beta=0.99999;
    gamma = momen.Gamma();
    if (beta*gamma < 0.1) return;
    theta = ComputeIncidenceAngle(momen);

    // Get the pixel ID
    if(!fSeg->localToDetector(x,z,ix,iz)) return;

    acs = ACSFromBetaGamma(beta*gamma, theta);
    cs = GetPixelPositionResponse(ix, iz, x, z, acs);
    SimuClusterShaper *csManager = new SimuClusterShaper(cs);
    csManager->FillClusterRandomly();
    csManager->GetShape(cshape);
    nrows = csManager->GetNRows();
    ncols = csManager->GetNCols();
    cx = gRandom->Integer(ncols);
    cz = gRandom->Integer(nrows);

    LOG(DEBUG) << "_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << FairLogger::endl;
    LOG(DEBUG) << "_/_/_/ pALPIDE debug  _/_/_/" << FairLogger::endl;
    LOG(DEBUG) << "_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << FairLogger::endl;
    LOG(DEBUG) << " Beta*Gamma: " << beta*gamma << FairLogger::endl;
    LOG(DEBUG) << "        ACS: " << acs << FairLogger::endl;
    LOG(DEBUG) << "         CS: " <<  cs << FairLogger::endl;
    LOG(DEBUG) << "      Shape: " << ClusterShape::ShapeSting(cshape).c_str() << FairLogger::endl;
    LOG(DEBUG) << "     Center: " << cx << ' ' << cz << FairLogger::endl;
    LOG(DEBUG) << "_/_/_/_/_/_/_/_/_/_/_/_/_/_/" << FairLogger::endl;

    for (Int_t ipix = 0; ipix < cs; ++ipix) {
      UInt_t r = (Int_t) cshape[ipix] / nrows;
      UInt_t c = cshape[ipix] % nrows;
      nx = ix - cx + c;
      nz = iz - cz + r;
      CreateDigi(nz, nx, idtrack, h);
    }

    delete csManager;
  }
}

//______________________________________________________________________
void SimulationAlpide::CreateDigi(UInt_t col, UInt_t row, Int_t track, Int_t hit) {
  UInt_t index = fSensMap->getIndex(col, row, 0);
  UInt_t chip  = fChip->GetChipIndex();

  fSensMap->registerItem(new (fSensMap->getFree()) SDigit(track, hit, chip, index, 0.1, 0));
}

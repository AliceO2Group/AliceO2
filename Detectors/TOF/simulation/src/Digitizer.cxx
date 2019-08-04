// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TOFSimulation/Digitizer.h"

#include "TCanvas.h"
#include "TFile.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TLeaf.h"
#include "TMath.h"
#include "TProfile2D.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>

using namespace o2::tof;

ClassImp(Digitizer);

void Digitizer::init()
{

  // method to initialize the parameters neede to digitize and the array of strip objects containing
  // the digits belonging to a strip

  initParameters();

  for (Int_t i = 0; i < Geo::NSTRIPS; i++) {
    for (Int_t j = 0; j < MAXWINDOWS; j++) {
      mStrips[j].emplace_back(i);
      if (j < MAXWINDOWS - 1) {
        mMCTruthContainerNext[j] = &(mMCTruthContainer[j + 1]);
        mStripsNext[j] = &(mStrips[j + 1]);
      }
    }
  }
}

//______________________________________________________________________

void Digitizer::process(const std::vector<HitType>* hits, std::vector<Digit>* digits)
{
  // hits array of TOF hits for a given simulated event
  // digits passed from external to be filled, in continuous readout mode we will push it on mDigitsPerTimeFrame vector of vectors of digits

  Int_t readoutwindow = Int_t((mEventTime)*Geo::READOUTWINDOW_INV); // to be replaced with "uncalibrated" time

  printf("process TOF -> continuous = %i, %i > %i?\n", mContinuous, readoutwindow, mReadoutWindowCurrent);

  if (mContinuous && readoutwindow > mReadoutWindowCurrent) { // if we are moving in future readout windows flush previous ones (only for continuous readout mode)
    digits->clear();

    for (; mReadoutWindowCurrent < readoutwindow; mReadoutWindowCurrent++) {
      fillOutputContainer(*digits); // fill all windows which are before (not yet stored) of the new current one
      checkIfReuseFutureDigits();
    } // close loop readout window
  }   // close if continuous

  for (auto& hit : *hits) {
    //TODO: put readout window counting/selection

    processHit(hit, mEventTime);
  } // end loop over hits

  if (!mContinuous) { // fill output container per event
    digits->clear();
    fillOutputContainer(*digits);
  }
}

//______________________________________________________________________

Int_t Digitizer::processHit(const HitType& hit, Double_t event_time)
{
  Float_t pos[3] = {hit.GetX(), hit.GetY(), hit.GetZ()};
  Float_t deltapos[3];
  Int_t detInd[5];
  Int_t detIndOtherPad[5];

  Geo::getPadDxDyDz(pos, detInd, deltapos); // Get DetId and residuals

  detIndOtherPad[0] = detInd[0], detIndOtherPad[1] = detInd[1],
  detIndOtherPad[2] = detInd[2]; // same sector, plate, strip

  Int_t otherraw = 0;
  if (detInd[3] == 0)
    otherraw = 1;

  Int_t iZshift = otherraw ? 1 : -1;

  Int_t channel = Geo::getIndex(detInd);

  Float_t charge = getCharge(hit.GetEnergyLoss());
  // NOTE: FROM NOW ON THE TIME IS IN PS ... AND NOT IN NS
  Float_t time = getShowerTimeSmeared((event_time + hit.GetTime()) * 1E3, charge);

  Float_t xLocal = deltapos[0];
  Float_t zLocal = deltapos[2];

  // extract trackID
  auto trackID = hit.GetTrackID();

  //         PadId - Pad Identifier
  //                    E | F    -->   PadId = 5 | 6
  //                    A | B    -->   PadId = 1 | 2
  //                    C | D    -->   PadId = 3 | 4

  Int_t ndigits = 0; //Number of digits added

  UInt_t istrip = channel / Geo::NPADS;

  // check the fired PAD 1 (A)
  if (isFired(xLocal, zLocal, charge)) {
    ndigits++;
    addDigit(channel, istrip, time, xLocal, zLocal, charge, 0, 0, detInd[3], trackID);
  }

  // check PAD 2
  detIndOtherPad[3] = otherraw;
  detIndOtherPad[4] = detInd[4];
  channel = Geo::getIndex(detIndOtherPad);
  xLocal = deltapos[0]; // recompute local coordinates
  if (otherraw)
    zLocal = deltapos[2] - Geo::ZPAD; // recompute local coordinates
  else
    zLocal = deltapos[2] + Geo::ZPAD;
  if (isFired(xLocal, zLocal, charge)) {
    ndigits++;
    addDigit(channel, istrip, time, xLocal, zLocal, charge, 0, iZshift, detInd[3], trackID);
  }

  // check PAD 3
  detIndOtherPad[3] = detInd[3];
  detIndOtherPad[4] = detInd[4] - 1;
  if (detIndOtherPad[4] >= 0) {
    channel = Geo::getIndex(detIndOtherPad);
    xLocal = deltapos[0] + Geo::XPAD; // recompute local coordinates
    zLocal = deltapos[2];             // recompute local coordinates
    if (isFired(xLocal, zLocal, charge)) {
      ndigits++;
      addDigit(channel, istrip, time, xLocal, zLocal, charge, -1, 0, detInd[3], trackID);
    }
  }

  // check PAD 5
  detIndOtherPad[3] = detInd[3];
  detIndOtherPad[4] = detInd[4] + 1;
  if (detIndOtherPad[4] < Geo::NPADX) {
    channel = Geo::getIndex(detIndOtherPad);
    xLocal = deltapos[0] - Geo::XPAD; // recompute local coordinates
    zLocal = deltapos[2];             // recompute local coordinates
    if (isFired(xLocal, zLocal, charge)) {
      ndigits++;
      addDigit(channel, istrip, time, xLocal, zLocal, charge, 1, 0, detInd[3], trackID);
    }
  }

  // check PAD 4
  detIndOtherPad[3] = otherraw;
  detIndOtherPad[4] = detInd[4] - 1;
  if (detIndOtherPad[4] >= 0) {
    channel = Geo::getIndex(detIndOtherPad);
    xLocal = deltapos[0] + Geo::XPAD; // recompute local coordinates
    if (otherraw)
      zLocal = deltapos[2] - Geo::ZPAD; // recompute local coordinates
    else
      zLocal = deltapos[2] + Geo::ZPAD;
    if (isFired(xLocal, zLocal, charge)) {
      ndigits++;
      addDigit(channel, istrip, time, xLocal, zLocal, charge, -1, iZshift, detInd[3], trackID);
    }
  }

  // check PAD 6
  detIndOtherPad[3] = otherraw;
  detIndOtherPad[4] = detInd[4] + 1;
  if (detIndOtherPad[4] < Geo::NPADX) {
    channel = Geo::getIndex(detIndOtherPad);
    xLocal = deltapos[0] - Geo::XPAD; // recompute local coordinates
    if (otherraw)
      zLocal = deltapos[2] - Geo::ZPAD; // recompute local coordinates
    else
      zLocal = deltapos[2] + Geo::ZPAD;
    if (isFired(xLocal, zLocal, charge)) {
      ndigits++;
      addDigit(channel, istrip, time, xLocal, zLocal, charge, 1, iZshift, detInd[3], trackID);
    }
  }
  return ndigits;
}

//______________________________________________________________________
void Digitizer::addDigit(Int_t channel, UInt_t istrip, Float_t time, Float_t x, Float_t z, Float_t charge, Int_t iX, Int_t iZ,
                         Int_t padZfired, Int_t trackID)
{
  // TOF digit requires: channel, time and time-over-threshold

  time = getDigitTimeSmeared(time, x, z, charge); // add time smearing

  charge *= getFractionOfCharge(x, z);
  Float_t tot = 12; // time-over-threshold

  Float_t xborder = Geo::XPAD * 0.5 - TMath::Abs(x);
  Float_t zborder = Geo::ZPAD * 0.5 - TMath::Abs(z);
  Float_t border = TMath::Min(xborder, zborder);

  Float_t timewalkX = x * mTimeWalkeSlope;
  Float_t timewalkZ = (z - (padZfired - 0.5) * Geo::ZPAD) * mTimeWalkeSlope;

  if (border < 0) { // keep the effect onlu if hit out of pad
    border *= -1;
    Float_t extraTimeSmear = border * mTimeSlope;
    time += gRandom->Gaus(mTimeDelay, extraTimeSmear);
  } else {
    border = 1 - border;
    // if(border > 0)  printf("deltat =%f\n",mTimeDelay*border*border*border);
    // else printf("deltat=0\n");
    // getchar();
    if (border > 0)
      time += mTimeDelay * border * border * border;
  }
  time += TMath::Sqrt(timewalkX * timewalkX + timewalkZ * timewalkZ) - mTimeDelayCorr - mTimeWalkeSlope * 2;

  Int_t nbc = Int_t(time * Geo::BC_TIME_INPS_INV); // time elapsed in number of bunch crossing
  //Digit newdigit(time, channel, (time - Geo::BC_TIME_INPS * nbc) * Geo::NTDCBIN_PER_PS, tot * Geo::NTOTBIN_PER_NS, nbc);

  auto tdc = (time - Geo::BC_TIME_INPS * nbc) * Geo::NTDCBIN_PER_PS;

  int lblCurrent = 0;

  bool iscurrent = true; // if we are in the current readout window
  Int_t isnext = -1;

  if (mContinuous) {
    isnext = Int_t(time * 1E-3 * Geo::READOUTWINDOW_INV) - mReadoutWindowCurrent; // to be replaced with uncalibrated time

    if (isnext < 0 || isnext >= MAXWINDOWS - 1) {

      lblCurrent = mFutureIevent.size(); // this is the size of mHeaderArray;
      mFutureIevent.push_back(mEventID);
      mFutureIsource.push_back(mSrcID);
      mFutureItrackID.push_back(trackID);

      // fill temporary digits array
      mFutureDigits.emplace_back(channel, tdc, tot * Geo::NTOTBIN_PER_NS, nbc, lblCurrent);

      return; // don't fill if doesn't match any available readout window
    }

    if (isnext)
      iscurrent = false;
  }

  //  printf("add TOF digit c=%i n=%i\n",iscurrent,isnext);

  std::vector<Strip>* strips;
  o2::dataformats::MCTruthContainer<o2::tof::MCLabel>* mcTruthContainer;

  if (iscurrent) {
    strips = mStripsCurrent;
    mcTruthContainer = mMCTruthContainerCurrent;
  } else {
    strips = mStripsNext[isnext - 1];
    mcTruthContainer = mMCTruthContainerNext[isnext - 1];
  }

  fillDigitsInStrip(strips, mcTruthContainer, channel, tdc, tot, nbc, istrip, trackID, mEventID, mSrcID);
}
//______________________________________________________________________
void Digitizer::fillDigitsInStrip(std::vector<Strip>* strips, o2::dataformats::MCTruthContainer<o2::tof::MCLabel>* mcTruthContainer, int channel, int tdc, int tot, int nbc, UInt_t istrip, Int_t trackID, Int_t eventID, Int_t sourceID)
{
  int lblCurrent;
  if (mcTruthContainer) {
    lblCurrent = mcTruthContainer->getIndexedSize(); // this is the size of mHeaderArray;
  }

  Int_t lbl = (*strips)[istrip].addDigit(channel, tdc, tot * Geo::NTOTBIN_PER_NS, nbc, lblCurrent);

  if (mcTruthContainer) {
    if (lbl == lblCurrent) { // it means that the digit was a new one --> we have to add the info in the MC container
      o2::tof::MCLabel label(trackID, eventID, sourceID, tdc);
      mcTruthContainer->addElement(lbl, label);
    } else {
      o2::tof::MCLabel label(trackID, eventID, sourceID, tdc);
      mcTruthContainer->addElementRandomAccess(lbl, label);

      // sort the labels according to increasing tdc value
      auto labels = mcTruthContainer->getLabels(lbl);
      std::sort(labels.begin(), labels.end(),
                [](o2::tof::MCLabel a, o2::tof::MCLabel b) { return a.getTDC() < b.getTDC(); });
    }
  }
}
//______________________________________________________________________
Float_t Digitizer::getShowerTimeSmeared(Float_t time, Float_t charge)
{
  // add the smearing common to all the digits belongin to the same shower
  return time + gRandom->Gaus(0, mShowerResolution);
}

//______________________________________________________________________
Float_t Digitizer::getDigitTimeSmeared(Float_t time, Float_t x, Float_t z, Float_t charge)
{
  // add the smearing component which is indepedent for any digit even if belonging to the same shower (in case of
  // multiple hits)
  return time + gRandom->Gaus(0, mDigitResolution); // sqrt(33**2 + 50**2) ps = 60 ps
}

//______________________________________________________________________
Float_t Digitizer::getCharge(Float_t eDep)
{
  // transform deposited energy in collected charge
  Float_t adcMean = 50;
  Float_t adcRms = 25;

  return gRandom->Landau(adcMean, adcRms);
}

//______________________________________________________________________
Bool_t Digitizer::isFired(Float_t x, Float_t z, Float_t charge)
{
  if (TMath::Abs(x) > Geo::XPAD * 0.5 + 0.3)
    return kFALSE;
  if (TMath::Abs(z) > Geo::ZPAD * 0.5 + 0.3)
    return kFALSE;

  Float_t effX = getEffX(x);
  Float_t effZ = getEffZ(z);

  Float_t efficiency = TMath::Min(effX, effZ);

  if (gRandom->Rndm() > efficiency)
    return kFALSE;

  return kTRUE;
}

//______________________________________________________________________
Float_t Digitizer::getEffX(Float_t x)
{
  Float_t xborder = Geo::XPAD * 0.5 - TMath::Abs(x);

  if (xborder > 0) {
    if (xborder > mBound1)
      return mEffCenter;
    else if (xborder > mBound2)
      return mEffBoundary1 + (mEffCenter - mEffBoundary1) * (xborder - mBound2) / (mBound1 - mBound2);
    else
      return mEffBoundary2 + (mEffBoundary1 - mEffBoundary2) * xborder / mBound2;
  } else {
    xborder *= -1;
    if (xborder > mBound4)
      return 0;
    else if (xborder > mBound3)
      return mEffBoundary3 - mEffBoundary3 * (xborder - mBound3) / (mBound4 - mBound3);
    else
      return mEffBoundary2 + (mEffBoundary3 - mEffBoundary2) * xborder / mBound3;
  }

  return 0;
}

//______________________________________________________________________
Float_t Digitizer::getEffZ(Float_t z)
{
  Float_t zborder = Geo::ZPAD * 0.5 - TMath::Abs(z);

  if (zborder > 0) {
    if (zborder > mBound1)
      return mEffCenter;
    else if (zborder > mBound2)
      return mEffBoundary1 + (mEffCenter - mEffBoundary1) * (zborder - mBound2) / (mBound1 - mBound2);
    else
      return mEffBoundary2 + (mEffBoundary1 - mEffBoundary2) * zborder / mBound2;
  } else {
    zborder *= -1;
    if (zborder > mBound4)
      return 0;
    else if (zborder > mBound3)
      return mEffBoundary3 - mEffBoundary3 * (zborder - mBound3) / (mBound4 - mBound3);
    else
      return mEffBoundary2 + (mEffBoundary3 - mEffBoundary2) * zborder / mBound3;
  }

  return 0;
}

//______________________________________________________________________
Float_t Digitizer::getFractionOfCharge(Float_t x, Float_t z) { return 1; }

//______________________________________________________________________
void Digitizer::initParameters()
{
  // boundary references for interpolation of efficiency and resolution
  mBound1 = 0.4;  // distance from border when efficiency starts to decrese
  mBound2 = 0.15; // second step in the fired pad
  mBound3 = 0.55; // distance from border (not fired pad)
  mBound4 = 0.9;  // distance from border (not fired pad) when efficiency vanishes

  // resolution parameters
  mTOFresolution = 60;    // TOF global resolution in ps
  mShowerResolution = 50; // smearing correlated for all digits of the same hit
  if (mTOFresolution > mShowerResolution)
    mDigitResolution = TMath::Sqrt(mTOFresolution * mTOFresolution -
                                   mShowerResolution * mShowerResolution); // independent smearing for each digit
  else {
    mShowerResolution = mTOFresolution;
    mDigitResolution = 0;
  }

  mTimeSlope = 100;     // ps/cm extra smearing if hit out of pad propto the distance from the border
  mTimeDelay = 70;      // time delay if hit out of pad
  mTimeWalkeSlope = 40; // ps/cm

  if (mMode == 0) {
    mTimeSlope = 0;
    mTimeDelay = 0;
    mTimeWalkeSlope = 0;
  }

  mTimeDelayCorr = mTimeDelay / 3.5;
  if (mShowerResolution > mTimeDelayCorr)
    mShowerResolution = TMath::Sqrt(mShowerResolution * mShowerResolution - mTimeDelayCorr * mTimeDelayCorr);
  else
    mShowerResolution = 0;

  if (mShowerResolution > mTimeWalkeSlope * 0.8)
    mShowerResolution = TMath::Sqrt(mShowerResolution * mShowerResolution - mTimeWalkeSlope * mTimeWalkeSlope * 0.64);
  else
    mShowerResolution = 0;

  // efficiency parameters
  mEffCenter = 0.995;    // efficiency in the center of the fired pad
  mEffBoundary1 = 0.94;  // efficiency in mBound2
  mEffBoundary2 = 0.833; // efficiency in the pad border
  mEffBoundary3 = 0.1;   // efficiency in mBound3
}

//______________________________________________________________________
void Digitizer::printParameters()
{
  printf("Efficiency in the pad center = %f\n", mEffCenter);
  printf("Efficiency in the pad border = %f\n", mEffBoundary2);
  printf("Time resolution = %f ps (shower=%f, digit=%f)\n", mTOFresolution, mShowerResolution, mDigitResolution);
  if (mTimeSlope > 0)
    printf("Degration resolution for pad with signal induced = %f ps/cm x border distance\n", mTimeSlope);
  if (mTimeDelay > 0)
    printf("Time delay for pad with signal induced = %f ps\n", mTimeDelay);
  if (mTimeWalkeSlope > 0)
    printf("Time walk ON = %f ps/cm\n", mTimeWalkeSlope);
}

//______________________________________________________________________
void Digitizer::test(const char* geo)
{
  Int_t nhit = 1000000;

  TFile* fgeo = new TFile(geo);
  fgeo->Get("FAIRGeom");

  o2::tof::HitType* hit = new o2::tof::HitType();

  printParameters();

  TH1F* h = new TH1F("hTime", "Time as from digitizer;time (ps);N", 100, -500, 500);
  TH1F* h2 = new TH1F("hTot", "Tot as from digitizer;time (ns);N", 100, 0, 30);
  TH1F* h3 = new TH1F("hNdig", "N_{digitis} distribution from one hit;N_{digits};N", 7, -0.5, 6.5);
  TH1F* h4 = new TH1F("hTimeCorr", "Time correlation for double digits;#Deltat (ps)", 200, -1000, 1000);
  TH1F* h5 = new TH1F("hTimeAv", "Time average for double digits;#Deltat (ps)", 200, -1000, 1000);

  TH1F* hpad[3][3];
  for (Int_t i = 0; i < 3; i++) {
    for (Int_t j = 0; j < 3; j++) {
      hpad[i][j] =
        new TH1F(Form("hpad%i_%i", i, j), Form("Time as from digitizer, pad(%i,%i);time (ps);N", i, j), 100, -500, 500);
    }
  }

  TProfile2D* hTimeWalk = new TProfile2D("hTimeWalk", "Time Walk;x (cm);z (cm)", 40, -1.25, 1.25, 40, -1.75, 1.75);

  TH2F* hpadAll;
  hpadAll = new TH2F("hpadAll", "all hits;x (cm);z (cm)", 40, -1.25, 1.25, 40, -1.75, 1.75);
  TH2F* hpadHit[3][3];
  TH2F* hpadEff[3][3];
  for (Int_t i = 0; i < 3; i++) {
    for (Int_t j = 0; j < 3; j++) {
      hpadHit[i][j] = new TH2F(Form("hpadHit%i_%i", i, j), Form("pad(%i,%i) hits;x (cm);z (cm)", i - 1, 1 - j), 40,
                               -1.25, 1.25, 40, -1.75, 1.75);
      hpadEff[i][j] = new TH2F(Form("hpadEff%i_%i", i, j), Form("pad(%i,%i) hits;x (cm);z (cm)", i - 1, 1 - j), 40,
                               -1.25, 1.25, 40, -1.75, 1.75);
    }
  }

  Int_t det1[5] = {0, 0, 0, 1, 23};
  Int_t det2[5] = {0, 0, 0, 0, 24};
  Int_t det3[5] = {0, 0, 0, 1, 24};
  Int_t det4[5] = {0, 0, 0, 0, 47};
  Float_t pos[3], pos2[3], pos3[3], pos4[3];

  o2::tof::Geo::getPos(det1, pos);
  o2::tof::Geo::getPos(det2, pos2);
  o2::tof::Geo::getPos(det3, pos3);
  o2::tof::Geo::getPos(det4, pos4);

  // Get strip center
  pos[0] += pos2[0];
  pos[1] += pos2[1];
  pos[2] += pos2[2];
  pos[0] *= 0.5;
  pos[1] *= 0.5;
  pos[2] *= 0.5;

  Float_t mod;
  Float_t vx[3];
  Float_t vz[3];

  // Get z versor
  pos3[0] -= pos2[0];
  pos3[1] -= pos2[1];
  pos3[2] -= pos2[2];
  mod = TMath::Sqrt(pos3[0] * pos3[0] + pos3[1] * pos3[1] + pos3[2] * pos3[2]);
  vz[0] = pos3[0] / mod;
  vz[1] = pos3[1] / mod;
  vz[2] = pos3[2] / mod;

  // Get x versor
  pos4[0] -= pos2[0];
  pos4[1] -= pos2[1];
  pos4[2] -= pos2[2];
  mod = TMath::Sqrt(pos4[0] * pos4[0] + pos4[1] * pos4[1] + pos4[2] * pos4[2]);
  vx[0] = pos4[0] / mod;
  vx[1] = pos4[1] / mod;
  vx[2] = pos4[2] / mod;

  Float_t x[3], dx, dz, xlocal, zlocal;

  for (Int_t i = 0; i < nhit; i++) {
    dx = gRandom->Rndm() * 2.5 * 48;
    dz = gRandom->Rndm() * 3.5 * 2;

    xlocal = dx - Int_t(dx / 2.5) * 2.5 - 1.25;
    zlocal = dz - Int_t(dz / 3.5) * 3.5 - 1.75;

    dx -= 2.5 * 24;
    dz -= 3.5;

    x[0] = pos[0] + vx[0] * dx + vz[0] * dz;
    x[1] = pos[1] + vx[1] * dx + vz[1] * dz;
    x[2] = pos[2] + vx[2] * dx + vz[2] * dz;

    Int_t detCur[5];
    o2::tof::Geo::getDetID(x, detCur);

    hit->SetTime(0); // t->GetLeaf("o2root.TOF.TOFHit.mTime")->GetValue(j));
    hit->SetXYZ(x[0], x[1], x[2]);

    hit->SetEnergyLoss(0.0001);

    Int_t ndigits = processHit(*hit, mEventTime);

    h3->Fill(ndigits);
    hpadAll->Fill(xlocal, zlocal);
    for (Int_t k = 0; k < ndigits; k++) {
      if (k == 0)
        h->Fill(getTimeLastHit(k));
      if (k == 0)
        h2->Fill(getTotLastHit(k));
      if (k == 0 && getXshift(k) == 0 && getZshift(k) == 0)
        hTimeWalk->Fill(xlocal, zlocal * (0.5 - detCur[3]) * 2, getTimeLastHit(k));

      hpad[getXshift(k) + 1][-getZshift(k) + 1]->Fill(getTimeLastHit(k));
      hpadHit[getXshift(k) + 1][-getZshift(k) + 1]->Fill(xlocal, zlocal);
    }

    // check double digits case (time correlations)
    if (ndigits == 2) {
      h4->Fill(getTimeLastHit(0) - getTimeLastHit(1));
      h5->Fill((getTimeLastHit(0) + getTimeLastHit(1)) * 0.5);
    }
  }

  h->Draw();
  new TCanvas();
  h2->Draw();
  new TCanvas();
  h3->Draw();
  new TCanvas();
  h4->Draw();
  new TCanvas();
  h5->Draw();
  new TCanvas();
  hTimeWalk->Draw("SURF");

  TCanvas* cpad = new TCanvas();
  cpad->Divide(3, 3);
  for (Int_t i = 0; i < 3; i++) {
    for (Int_t j = 0; j < 3; j++) {
      cpad->cd(j * 3 + i + 1);
      hpad[i][j]->Draw();
    }
  }
  TCanvas* cpadH = new TCanvas();
  cpadH->Divide(3, 3);
  for (Int_t i = 0; i < 3; i++) {
    for (Int_t j = 0; j < 3; j++) {
      cpadH->cd(j * 3 + i + 1);
      hpadHit[i][j]->Draw("colz");
      if (j != 1)
        hpadHit[i][j]->Scale(2);
      hpadEff[i][j]->Divide(hpadHit[i][j], hpadAll, 1, 1, "B");
      hpadEff[i][j]->Draw("surf");
      hpadEff[i][j]->SetMaximum(1);
      hpadEff[i][j]->SetMinimum(0);
      hpadEff[i][j]->SetStats(0);
    }
  }

  printf("\nEfficiency = %f\n", (h3->GetEntries() - h3->GetBinContent(1)) / h3->GetEntries());
  printf("Multiple digits fraction = %f\n\n",
         (h3->GetEntries() - h3->GetBinContent(1) - h3->GetBinContent(2)) / (h3->GetEntries() - h3->GetBinContent(1)));
}

//______________________________________________________________________
void Digitizer::testFromHits(const char* geo, const char* hits)
{
  TFile* fgeo = new TFile(geo);
  fgeo->Get("FAIRGeom");

  TFile* fHit = new TFile(hits);
  fHit->ls();

  TTree* t = (TTree*)fHit->Get("o2sim");
  Int_t nev = t->GetEntriesFast();

  o2::tof::HitType* hit = new o2::tof::HitType();

  printParameters();

  TH1F* h = new TH1F("hTime", "Time as from digitizer;time (ps);N", 100, -500, 500);
  TH1F* h2 = new TH1F("hTot", "Tot as from digitizer;time (ns);N", 100, 0, 30);
  TH1F* h3 = new TH1F("hNdig", "N_{digitis} distribution from one hit;N_{digits};N", 7, -0.5, 6.5);

  for (Int_t i = 0; i < nev; i++) {
    t->GetEvent(i);
    Int_t nhit = t->GetLeaf("o2root.TOF.TOFHit_")->GetLen();

    for (Int_t j = 0; j < nhit; j++) {
      hit->SetTime(0); // t->GetLeaf("o2root.TOF.TOFHit.mTime")->GetValue(j));
      hit->SetXYZ(t->GetLeaf("o2root.TOF.TOFHit.mPos.fCoordinates.fX")->GetValue(j),
                  t->GetLeaf("o2root.TOF.TOFHit.mPos.fCoordinates.fY")->GetValue(j),
                  t->GetLeaf("o2root.TOF.TOFHit.mPos.fCoordinates.fZ")->GetValue(j));

      hit->SetEnergyLoss(t->GetLeaf("o2root.TOF.TOFHit.mELoss")->GetValue(j));

      Int_t ndigits = processHit(*hit, mEventTime);

      h3->Fill(ndigits);
      for (Int_t k = 0; k < ndigits; k++) {
        h->Fill(getTimeLastHit(k));
        h2->Fill(getTotLastHit(k));
      }
    }
  }

  h->Draw();
  new TCanvas();
  h2->Draw();
  new TCanvas();
  h3->Draw();
}
//______________________________________________________________________
void Digitizer::fillOutputContainer(std::vector<Digit>& digits)
{
  if (mContinuous) {
    digits.clear();
    mMCTruthOutputContainer->clear();
  }

  printf("TOF fill output contatiner\n");
  // filling the digit container doing a loop on all strips
  for (auto& strip : *mStripsCurrent) {
    strip.fillOutputContainer(digits);
  }

  if (mContinuous) {
    if (digits.size())
      printf("%i) # TOF digits = %lu (%p)\n", mIcurrentReadoutWindow, digits.size(), mStripsCurrent);
    mDigitsPerTimeFrame.push_back(digits);
  }

  // if(! digits.size()) return;

  // copying the transient labels to the output labels (stripping the tdc information)
  if (mMCTruthOutputContainer) {
    // copy from transientTruthContainer to mMCTruthAray
    // a brute force solution for the moment; should be handled by a dedicated API
    for (int index = 0; index < mMCTruthContainerCurrent->getIndexedSize(); ++index) {
      mMCTruthOutputContainer->addElements(index, mMCTruthContainerCurrent->getLabels(index));
    }
  }

  if (mContinuous)
    mMCTruthOutputContainerPerTimeFrame.push_back(*mMCTruthOutputContainer);
  mMCTruthContainerCurrent->clear();

  // switch to next mStrip after flushing current readout window data
  mIcurrentReadoutWindow++;
  if (mIcurrentReadoutWindow >= MAXWINDOWS)
    mIcurrentReadoutWindow = 0;

  mStripsCurrent = &(mStrips[mIcurrentReadoutWindow]);
  mMCTruthContainerCurrent = &(mMCTruthContainer[mIcurrentReadoutWindow]);

  int k = mIcurrentReadoutWindow + 1;
  for (Int_t i = 0; i < MAXWINDOWS - 1; i++) {
    if (k >= MAXWINDOWS)
      k = 0;
    mMCTruthContainerNext[i] = &(mMCTruthContainer[k]);
    mStripsNext[i] = &(mStrips[k]);
    k++;
  }
}
//______________________________________________________________________
void Digitizer::flushOutputContainer(std::vector<Digit>& digits)
{ // flush all residual buffered data
  // TO be implemented
  printf("flushOutputContainer\n");
  if (!mContinuous)
    fillOutputContainer(digits);
  else {
    for (Int_t i = 0; i < MAXWINDOWS; i++) {
      fillOutputContainer(digits); // fill all windows which are before (not yet stored) of the new current one
      checkIfReuseFutureDigits();
      mReadoutWindowCurrent++;
    }

    while (mFutureDigits.size()) {
      fillOutputContainer(digits); // fill all windows which are before (not yet stored) of the new current one
      checkIfReuseFutureDigits();
      mReadoutWindowCurrent++;
    }
  }
}
//______________________________________________________________________
void Digitizer::checkIfReuseFutureDigits()
{
  // check if digits stored very far in future match the new readout windows currently available
  int idigit = 0;
  for (auto& digit : mFutureDigits) {
    double timestamp = digit.getBC() * 25 + digit.getTDC() * Geo::TDCBIN * 1E-3;          // in ns
    int isnext = Int_t(timestamp * Geo::READOUTWINDOW_INV) - (mReadoutWindowCurrent + 1); // to be replaced with uncalibrated time
    if (isnext < 0)                                                                       // we jump too ahead in future, digit will be not stored
      LOG(INFO) << "Digit lost because we jump too ahead in future. Current RO window=" << isnext << "\n";
    if (isnext < MAXWINDOWS - 1) { // move from digit buffer array to the proper window
      if (isnext >= 0) {
        std::vector<Strip>* strips = mStripsCurrent;
        o2::dataformats::MCTruthContainer<o2::tof::MCLabel>* mcTruthContainer = mMCTruthContainerCurrent;

        if (isnext) {
          strips = mStripsNext[isnext - 1];
          mcTruthContainer = mMCTruthContainerNext[isnext - 1];
        }

        int trackID = mFutureItrackID[digit.getLabel()];
        int sourceID = mFutureIsource[digit.getLabel()];
        int eventID = mFutureIevent[digit.getLabel()];
        fillDigitsInStrip(strips, mcTruthContainer, digit.getChannel(), digit.getTDC(), digit.getTOT(), digit.getBC(), digit.getChannel() / Geo::NPADS, trackID, eventID, sourceID);
      }

      // remove the element from the buffers
      mFutureItrackID.erase(mFutureItrackID.begin() + digit.getLabel());
      mFutureIsource.erase(mFutureIsource.begin() + digit.getLabel());
      mFutureIevent.erase(mFutureIevent.begin() + digit.getLabel());

      int labelremoved = digit.getLabel();
      // adjust labels
      for (auto& digit2 : mFutureDigits) {
        if (digit2.getLabel() > labelremoved)
          digit2.setLabel(digit2.getLabel() - 1);
      }
      // remove also digit from buffer
      mFutureDigits.erase(mFutureDigits.begin() + idigit);
    } else {
      idigit++; // increment when moving to the next only if the current is not removed from the buffer
    }
  } // close future digit loop
}

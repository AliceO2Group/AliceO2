// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  Strip.cxx: structure to store the TOF digits in strips - useful
// for clusterization purposes
//  ALICEO2
//
#include <TMath.h>
#include <TH1D.h>

#include "TOFBase/Utils.h"

#define MAX_NUM_EVENT_AUTODETECT 10000

using namespace o2::tof;

ClassImp(o2::tof::Utils);

std::vector<int> Utils::mFillScheme;
int Utils::mBCmult[o2::constants::lhc::LHCMaxBunches];
int Utils::mNautodet = 0;
int Utils::mMaxBC = 0;
bool Utils::mIsInit = false;
float Utils::mEventTimeSpread = 200;
float Utils::mEtaMin = -0.8;
float Utils::mEtaMax = 0.8;
float Utils::mLHCPhase = 0;
int Utils::mNCalibTracks = 0;
o2::dataformats::CalibInfoTOF Utils::mCalibTracks[NTRACKS_REQUESTED];
int Utils::mNsample = 0;
int Utils::mIsample = 0;
float Utils::mPhases[100];
uint64_t Utils::mMaskBC[16] = {};
uint64_t Utils::mMaskBCUsed[16] = {};
int Utils::mMaskBCchan[o2::tof::Geo::NCHANNELS][16] = {};
int Utils::mMaskBCchanUsed[o2::tof::Geo::NCHANNELS][16] = {};
TF1* Utils::mFitFunc = new TF1("fTOFfit", "gaus", -5000, 5000);
;
TChain* Utils::mTreeFit = nullptr;
std::vector<o2::dataformats::CalibInfoTOF> Utils::mVectC;
std::vector<o2::dataformats::CalibInfoTOF>* Utils::mPvectC = &mVectC;
int Utils::mNfits = 0;

void Utils::addInteractionBC(int bc, bool fromCollisonCotext)
{
  if (fromCollisonCotext) { // align to TOF
    if (bc + Geo::LATENCY_ADJ_LHC_IN_BC < 0) {
      mFillScheme.push_back(bc + Geo::LATENCY_ADJ_LHC_IN_BC + Geo::BC_IN_ORBIT);
    } else if (bc + Geo::LATENCY_ADJ_LHC_IN_BC >= Geo::BC_IN_ORBIT) {
      mFillScheme.push_back(bc + Geo::LATENCY_ADJ_LHC_IN_BC - Geo::BC_IN_ORBIT);
    } else {
      mFillScheme.push_back(bc + Geo::LATENCY_ADJ_LHC_IN_BC);
    }
  } else {
    mFillScheme.push_back(bc);
  }
}

void Utils::init()
{
  memset(mBCmult, 0, o2::constants::lhc::LHCMaxBunches * sizeof(mBCmult[0]));
}

void Utils::addCalibTrack(float ctime)
{
  mCalibTracks[mNCalibTracks].setDeltaTimePi(ctime);

  mNCalibTracks++;

  if (mNCalibTracks >= NTRACKS_REQUESTED) {
    computeLHCphase();
    mNCalibTracks = 0;
  }
}

void Utils::computeLHCphase()
{
  static std::vector<o2::dataformats::CalibInfoTOF> tracks;
  tracks.clear();
  for (int i = 0; i < NTRACKS_REQUESTED; i++) {
    tracks.push_back(mCalibTracks[i]);
  }

  auto evtime = evTimeMaker<std::vector<o2::dataformats::CalibInfoTOF>, o2::dataformats::CalibInfoTOF, filterCalib<o2::dataformats::CalibInfoTOF>>(tracks, 6.0f, true);

  if (evtime.mEventTimeError < 100) { // udpate LHCphase
    mPhases[mIsample] = evtime.mEventTime;
    mIsample = (mIsample + 1) % 100;
    if (mNsample < 100) {
      mNsample++;
    }
  }

  mLHCPhase = 0;
  for (int i = 0; i < mNsample; i++) {
    mLHCPhase += mPhases[i];
  }
  mLHCPhase /= mNsample;
}

void Utils::printFillScheme()
{
  printf("FILL SCHEME\n");
  for (int i = 0; i < getNinteractionBC(); i++) {
    printf("BC(%d) LHCref=%d TOFref=%d\n", i, mFillScheme[i] - Geo::LATENCY_ADJ_LHC_IN_BC, mFillScheme[i]);
  }
}

int Utils::getNinteractionBC()
{
  return mFillScheme.size();
}

double Utils::subtractInteractionBC(double time, int& mask, bool subLatency)
{
  static const int deltalat = o2::tof::Geo::BC_IN_ORBIT - o2::tof::Geo::LATENCYWINDOW_IN_BC;
  int bc = int(time * o2::tof::Geo::BC_TIME_INPS_INV + 0.2);

  if (subLatency) {
    if (bc >= o2::tof::Geo::LATENCYWINDOW_IN_BC) {
      bc -= o2::tof::Geo::LATENCYWINDOW_IN_BC;
      time -= o2::tof::Geo::LATENCYWINDOW_IN_BC * o2::tof::Geo::BC_TIME_INPS;
    } else {
      bc += deltalat;
      time += deltalat * o2::tof::Geo::BC_TIME_INPS;
    }
  }

  int bcOrbit = bc % o2::constants::lhc::LHCMaxBunches;

  int dbc = o2::constants::lhc::LHCMaxBunches, bcc = bc;
  int dbcSigned = 1000;
  for (int k = 0; k < getNinteractionBC(); k++) { // get bc from fill scheme closest
    int deltaCBC = bcOrbit - getInteractionBC(k);
    if (deltaCBC >= -8 && deltaCBC < 8) {
      mask += (1 << (deltaCBC + 8)); // fill bc candidates
    }
    if (abs(deltaCBC) < dbc) {
      bcc = bc - deltaCBC;
      dbcSigned = deltaCBC;
      dbc = abs(dbcSigned);
    }
    if (abs(deltaCBC + o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the right border (last BC of the orbit)
      bcc = bc - deltaCBC - o2::constants::lhc::LHCMaxBunches;
      dbcSigned = deltaCBC + o2::constants::lhc::LHCMaxBunches;
      dbc = abs(dbcSigned);
    }
    if (abs(deltaCBC - o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the left border (BC=0)
      bcc = bc - deltaCBC + o2::constants::lhc::LHCMaxBunches;
      dbcSigned = deltaCBC - o2::constants::lhc::LHCMaxBunches;
      dbc = abs(dbcSigned);
    }
  }
  if (dbcSigned >= -8 && dbcSigned < 8) {
    mask += (1 << (dbcSigned + 24)); // fill bc used
  }

  time -= o2::tof::Geo::BC_TIME_INPS * bcc;

  return time;
}

float Utils::subtractInteractionBC(float time, int& mask, bool subLatency)
{
  static const int deltalat = o2::tof::Geo::BC_IN_ORBIT - o2::tof::Geo::LATENCYWINDOW_IN_BC;
  int bc = int(time * o2::tof::Geo::BC_TIME_INPS_INV + 0.2);

  if (subLatency) {
    if (bc >= o2::tof::Geo::LATENCYWINDOW_IN_BC) {
      bc -= o2::tof::Geo::LATENCYWINDOW_IN_BC;
      time -= o2::tof::Geo::LATENCYWINDOW_IN_BC * o2::tof::Geo::BC_TIME_INPS;
    } else {
      bc += deltalat;
      time += deltalat * o2::tof::Geo::BC_TIME_INPS;
    }
  }

  int bcOrbit = bc % o2::constants::lhc::LHCMaxBunches;

  int dbc = o2::constants::lhc::LHCMaxBunches, bcc = bc;
  int dbcSigned = 1000;
  for (int k = 0; k < getNinteractionBC(); k++) { // get bc from fill scheme closest
    int deltaCBC = bcOrbit - getInteractionBC(k);
    if (deltaCBC >= -8 && deltaCBC < 8) {
      mask += (1 << (deltaCBC + 8)); // fill bc candidates
    }
    if (abs(deltaCBC) < dbc) {
      bcc = bc - deltaCBC;
      dbcSigned = deltaCBC;
      dbc = abs(dbcSigned);
    }
    if (abs(deltaCBC + o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the right border (last BC of the orbit)
      bcc = bc - deltaCBC - o2::constants::lhc::LHCMaxBunches;
      dbcSigned = deltaCBC + o2::constants::lhc::LHCMaxBunches;
      dbc = abs(dbcSigned);
    }
    if (abs(deltaCBC - o2::constants::lhc::LHCMaxBunches) < dbc) { // in case k is close to the left border (BC=0)
      bcc = bc - deltaCBC + o2::constants::lhc::LHCMaxBunches;
      dbcSigned = deltaCBC - o2::constants::lhc::LHCMaxBunches;
      dbc = abs(dbcSigned);
    }
  }
  if (dbcSigned >= -8 && dbcSigned < 8) {
    mask += (1 << (dbcSigned + 24)); // fill bc used
  }

  time -= o2::tof::Geo::BC_TIME_INPS * bcc;

  return time;
}

void Utils::addBC(float toftime, bool subLatency)
{
  if (!mIsInit) {
    init();
    mIsInit = true;
  }

  if (mNautodet > MAX_NUM_EVENT_AUTODETECT) {
    if (!hasFillScheme()) { // detect fill scheme
      int thres = mMaxBC / 2;
      for (int i = 0; i < o2::constants::lhc::LHCMaxBunches; i++) {
        if (mBCmult[i] > thres) { // good bunch
          addInteractionBC(i);
        }
      }
    }
    return;
  }

  // just fill
  static const int deltalat = o2::tof::Geo::BC_IN_ORBIT - o2::tof::Geo::LATENCYWINDOW_IN_BC;
  int bc = int(toftime * o2::tof::Geo::BC_TIME_INPS_INV + 0.2) % o2::constants::lhc::LHCMaxBunches;

  if (subLatency) {
    if (bc >= o2::tof::Geo::LATENCYWINDOW_IN_BC) {
      bc -= o2::tof::Geo::LATENCYWINDOW_IN_BC;
      toftime -= o2::tof::Geo::LATENCYWINDOW_IN_BC * o2::tof::Geo::BC_TIME_INPS;
    } else {
      bc += deltalat;
      toftime += deltalat * o2::tof::Geo::BC_TIME_INPS;
    }
  }

  mBCmult[bc]++;

  if (mBCmult[bc] > mMaxBC) {
    mMaxBC = mBCmult[bc];
  }
  mNautodet++;
}

bool Utils::hasFillScheme()
{
  if (getNinteractionBC()) {
    return true;
  }

  return false;
}

int Utils::addMaskBC(int mask, int channel)
{
  int mask2 = (mask >> 16);
  int cmask = 1;
  int used = 0;
  for (int ibit = 0; ibit < 16; ibit++) {
    if (mask & cmask) {
      mMaskBCchan[channel][ibit]++;
      mMaskBC[ibit]++;
    }
    if (mask2 & cmask) {
      mMaskBCchanUsed[channel][ibit]++;
      mMaskBCUsed[ibit]++;
      used = ibit - 8;
    }
    cmask *= 2;
  }
  return used;
}

int Utils::getMaxUsed()
{
  int cmask = 0;
  uint64_t val = 10; // at least 10 entry required
  for (int ibit = 0; ibit < 16; ibit++) {
    if (mMaskBC[ibit] > val) {
      val = mMaskBC[ibit];
      cmask = ibit - 8;
    }
  }
  return cmask;
}

int Utils::getMaxUsedChannel(int channel)
{
  int cmask = 0;
  int val = 10; // at least 10 entry required
  for (int ibit = 0; ibit < 16; ibit++) {
    if (mMaskBCchan[channel][ibit] > val) {
      val = mMaskBCchan[channel][ibit];
      cmask = ibit - 8;
    }
  }
  return cmask;
}

int Utils::extractNewTimeSlewing(const o2::dataformats::CalibTimeSlewingParamTOF* oldTS, o2::dataformats::CalibTimeSlewingParamTOF* newTS)
{
  if (!oldTS || !newTS) { // objects were not defined -> to nothing
    return 1;
  }
  newTS->bind();

  mFitFunc->SetParameter(0, 100);
  mFitFunc->SetParameter(1, 0);
  mFitFunc->SetParameter(2, 200);

  if (mTreeFit) { // remove previous tree
    delete mTreeFit;
  }

  mTreeFit = new TChain("treeCollectedCalibInfo", "treeCollectedCalibInfo");

  system("ls *collTOF*.root >listaCal"); // create list of calibInfo accumulated
  FILE* f = fopen("listaCal", "r");

  if (!f) { // no inputs -> return
    return 2;
  }

  char namefile[50];
  while (fscanf(f, "%s", namefile) == 1) {
    mTreeFit->AddFile(namefile);
  }

  if (!mTreeFit->GetEntries()) { // return if no entry available
    return 3;
  }

  mTreeFit->SetBranchAddress("TOFCollectedCalibInfo", &mPvectC);

  for (int isec = 0; isec < 18; isec++) {
    fitTimeSlewing(isec, oldTS, newTS);
  }

  return 0;
}

void Utils::fitTimeSlewing(int sector, const o2::dataformats::CalibTimeSlewingParamTOF* oldTS, o2::dataformats::CalibTimeSlewingParamTOF* newTS)
{
  const int nchPerSect = Geo::NCHANNELS / Geo::NSECTORS;
  for (int i = sector * nchPerSect; i < (sector + 1) * nchPerSect; i += NCHPERBUNCH) {
    fitChannelsTS(i, oldTS, newTS);
  }
}

void Utils::fitChannelsTS(int chStart, const o2::dataformats::CalibTimeSlewingParamTOF* oldTS, o2::dataformats::CalibTimeSlewingParamTOF* newTS)
{
  // fiting NCHPERBUNCH at the same time to optimze reading from tree
  TH2F* h[NCHPERBUNCH];
  float time, tot;
  int mask;
  int bcSel[NCHPERBUNCH];

  for (int ii = 0; ii < NCHPERBUNCH; ii++) {
    h[ii] = new TH2F(Form("h%d", chStart + ii), "", 1000, 0, 100, 100, -5000, 5000);
    bcSel[ii] = -9999;
  }

  for (int i = chStart; i + NCHPERBUNCH < mTreeFit->GetEntries(); i += 157248) {
    for (int ii = 0; ii < NCHPERBUNCH; ii++) {
      int ch = chStart + ii;
      mTreeFit->GetEvent(i + ii);
      int k = 0;
      bool skip = false;
      for (auto& obj : mVectC) {
        if (obj.getTOFChIndex() != ch || skip) {
          continue;
        }
        time = obj.getDeltaTimePi();
        tot = obj.getTot();
        mask = obj.getMask();
        time -= addMaskBC(mask, ch) * o2::tof::Geo::BC_TIME_INPS;
        if (time < -5000 || time > 20000) {
          continue;
        }
        float tscorr = oldTS->evalTimeSlewing(ch, tot);
        if (tscorr < -1000000 || tscorr > 1000000) {
          skip = true;
          continue;
        }
        time -= tscorr;

        if (bcSel[ii] > -9000) {
          time += bcSel[ii] * o2::tof::Geo::BC_TIME_INPS;
        } else {
          bcSel[ii] = 0;
        }
        while (time < -5000) {
          time += o2::tof::Geo::BC_TIME_INPS;
          bcSel[ii] += 1;
        }
        while (time > 20000) {
          time -= o2::tof::Geo::BC_TIME_INPS;
          bcSel[ii] -= 1;
        }

        // adjust to avoid borders effect
        if (time > 12500) {
          time -= o2::tof::Geo::BC_TIME_INPS;
        } else if (time < -12500) {
          time += o2::tof::Geo::BC_TIME_INPS;
        }

        h[ii]->Fill(tot, time);
      }
    }
  }

  for (int ii = 0; ii < NCHPERBUNCH; ii++) {
    mNfits += fitSingleChannel(chStart + ii, h[ii], oldTS, newTS);
    delete h[ii]; // clean histo once fitted
  }
}

int Utils::fitSingleChannel(int ch, TH2F* h, const o2::dataformats::CalibTimeSlewingParamTOF* oldTS, o2::dataformats::CalibTimeSlewingParamTOF* newTS)
{
  const int nchPerSect = Geo::NCHANNELS / Geo::NSECTORS;

  int fitted = 0;
  float offset = oldTS->getChannelOffset(ch);
  int sec = ch / nchPerSect;
  int chInSec = ch % nchPerSect;
  int istart = oldTS->getStartIndexForChannel(sec, chInSec);
  int istop = oldTS->getStopIndexForChannel(sec, chInSec);
  int nbinPrev = istop - istart;
  int np = 0;

  unsigned short oldtot[10000];
  short oldcorr[10000];
  unsigned short newtot[10000];
  short newcorr[10000];

  int count = 0;

  const std::vector<std::pair<unsigned short, short>>& vect = oldTS->getVector(sec);
  for (int i = istart; i < istop; i++) {
    oldtot[count] = vect[i].first;
    oldcorr[count] = vect[i].second;
    count++;
  }

  TH1D* hpro = h->ProjectionX("hpro");

  int ibin = 1;
  int nbin = h->GetXaxis()->GetNbins();
  float integralToEnd = h->Integral();

  if (nbinPrev == 0) {
    nbinPrev = 1;
    oldtot[0] = 0;
    oldcorr[0] = 0;
  }

  // propagate problematic from old TS
  newTS->setFractionUnderPeak(sec, chInSec, oldTS->getFractionUnderPeak(ch));
  newTS->setSigmaPeak(sec, chInSec, oldTS->getSigmaPeak(ch));
  bool isProb = oldTS->getFractionUnderPeak(ch) < 0.5 || oldTS->getSigmaPeak(ch) > 1000;

  if (isProb) { // if problematic
    // skip fit procedure
    integralToEnd = 0;
  }

  if (integralToEnd < NMINTOFIT) { // no update to be done
    np = 1;
    newtot[0] = 0;
    newcorr[0] = 0;
    newTS->setTimeSlewingInfo(ch, offset, nbinPrev, oldtot, oldcorr, np, newtot, newcorr);
    if (hpro) {
      delete hpro;
    }
    return fitted;
  }

  float totHalfWidth = h->GetXaxis()->GetBinWidth(1) * 0.5;

  int integral = 0;
  float x[10000], y[10000];
  for (int i = 1; i <= nbin; i++) {
    integral += hpro->GetBinContent(i);
    integralToEnd -= hpro->GetBinContent(i);

    if (integral < NMINTOFIT || (integralToEnd < NMINTOFIT && i < nbin)) {
      continue;
    }

    // get a point
    float xmin = h->GetXaxis()->GetBinCenter(ibin) - totHalfWidth;
    float xmax = h->GetXaxis()->GetBinCenter(i) + totHalfWidth;
    TH1D* hfit = h->ProjectionY(Form("mypro"), ibin, i);
    float xref = hfit->GetBinCenter(hfit->GetMaximumBin());

    hfit->Fit(mFitFunc, "QN0", "", xref - 500, xref + 500);
    fitted++;

    x[np] = (xmin + xmax) * 0.5;
    y[np] = mFitFunc->GetParameter(1);
    if (x[np] > 65.534) {
      continue; // max tot acceptable in ushort representation / 1000.
    }
    newtot[np] = x[np] * 1000;
    newcorr[np] = y[np];
    np++;
    ibin = i + 1;
    integral = 0;
    delete hfit;
  }

  newTS->setTimeSlewingInfo(ch, offset, nbinPrev, oldtot, oldcorr, np, newtot, newcorr);

  if (hpro) {
    delete hpro;
  }
  return fitted;
}

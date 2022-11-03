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

#include "PHOSCalibWorkflow/PHOSRunbyrunCalibrator.h"
#include "DataFormatsPHOS/TriggerRecord.h"
#include "Framework/ConfigParamRegistry.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/BasicCCDBManager.h"
#include "Framework/ControlService.h"
#include "TFile.h"
#include "TF1.h"

#include <fairlogger/Logger.h>

using namespace o2::phos;

using Slot = o2::calibration::TimeSlot<o2::phos::PHOSRunbyrunSlot>;

PHOSRunbyrunSlot::PHOSRunbyrunSlot(bool useCCDB, std::string path) : mUseCCDB(useCCDB), mCCDBPath(path)
{
  const int nMass = 150.;
  const float massMax = 0.3;
  for (int mod = 0; mod < 4; mod++) {
    mReMi[2 * mod] = boost::histogram::make_histogram(boost::histogram::axis::regular<>(nMass, 0., massMax, "mgg"));
    mReMi[2 * mod + 1] = boost::histogram::make_histogram(boost::histogram::axis::regular<>(nMass, 0., massMax, "mgg"));
  }
  mBuffer.reset(new RingBuffer());
}
PHOSRunbyrunSlot::PHOSRunbyrunSlot(const PHOSRunbyrunSlot& other)
{
  mUseCCDB = other.mUseCCDB;
  mRunStartTime = other.mRunStartTime;
  mCCDBPath = other.mCCDBPath;
  const int nMass = 150.;
  const float massMax = 0.3;
  for (int mod = 0; mod < 4; mod++) {
    mReMi[2 * mod] = boost::histogram::make_histogram(boost::histogram::axis::regular<>(nMass, 0., massMax, "mgg"));
    mReMi[2 * mod + 1] = boost::histogram::make_histogram(boost::histogram::axis::regular<>(nMass, 0., massMax, "mgg"));
  }
  mBuffer.reset(new RingBuffer());
  mBadMap = nullptr;
}

void PHOSRunbyrunSlot::print() const
{
  // to print number of entries in pi0 region
  double s[4];
  for (int mod = 0; mod < 4; mod++) {
    s[mod] = boost::histogram::algorithm::sum(mReMi[2 * mod]);
  }
  LOG(info) << "Total number of entries in pi0 region: " << s[0] << "," << s[1] << "," << s[2] << "," << s[3];
}

void PHOSRunbyrunSlot::fill(const gsl::span<const Cluster>& clusters, const gsl::span<const TriggerRecord>& trs)
{
  if (!mBadMap) {
    if (mUseCCDB) {
      LOG(info) << "Retrieving BadMap from CCDB";
      auto& ccdbManager = o2::ccdb::BasicCCDBManager::instance();
      ccdbManager.setURL(o2::base::NameConf::getCCDBServer());
      LOG(info) << " set-up CCDB " << o2::base::NameConf::getCCDBServer();
      mBadMap = std::make_unique<o2::phos::BadChannelsMap>(*(ccdbManager.get<o2::phos::BadChannelsMap>("PHS/Calib/BadMap")));

      if (!mBadMap) { // was not read from CCDB, but expected
        LOG(fatal) << "Can not read BadMap from CCDB, you may use --not-use-ccdb option to create default bad map";
      }
    } else {
      LOG(info) << "Do not use CCDB, create default BadMap";
      mBadMap.reset(new BadChannelsMap());
    }
  }

  for (auto& tr : trs) {

    int firstCluInEvent = tr.getFirstEntry();
    int lastCluInEvent = firstCluInEvent + tr.getNumberOfObjects();

    // TODO!!! Get MFT0 vertex
    TVector3 vertex = {0., 0., 0.};
    mBuffer->startNewEvent(); // mark stored clusters to be used for Mixing
    for (int i = firstCluInEvent; i < lastCluInEvent; i++) {
      const Cluster& clu = clusters[i];

      if (!checkCluster(clu)) {
        continue;
      }
      // prepare TLorentsVector
      float posX, posZ;
      clu.getLocalPosition(posX, posZ);
      TVector3 vec3;
      Geometry::GetInstance("Run3")->local2Global(clu.module(), posX, posZ, vec3);
      vec3 -= vertex;
      float e = clu.getEnergy();
      vec3 *= 1. / vec3.Mag();
      TLorentzVector v(vec3.X() * e, vec3.Y() * e, vec3.Z() * e, e);
      for (short ip = mBuffer->size(); ip--;) {
        const TLorentzVector& vp = mBuffer->getEntry(ip);
        TLorentzVector sum = v + vp;
        if (sum.Pt() > mPtCut) {
          if (mBuffer->isCurrentEvent(ip)) {      // same (real) event
            mReMi[2 * clu.module()](sum.M());     // put all high-pt pairs to bin 4-6 GeV
          } else {                                // Mixed
            mReMi[2 * clu.module() + 1](sum.M()); // put all high-pt pairs to bin 4-6 GeV
          }
        }
      }
      mBuffer->addEntry(v);
    }
  }
}
void PHOSRunbyrunSlot::merge(const PHOSRunbyrunSlot* prev)
{
  // Not used
}
bool PHOSRunbyrunSlot::checkCluster(const Cluster& clu)
{
  if (clu.getEnergy() > 1.e-4) {
    return false;
  }
  // First check BadMap
  float posX, posZ;
  clu.getLocalPosition(posX, posZ);
  short absId;
  Geometry::relPosToAbsId(clu.module(), posX, posZ, absId);
  if (!mBadMap->isChannelGood(absId)) {
    return false;
  }
  return (clu.getEnergy() > 0.3 && clu.getMultiplicity() > 1);
}
void PHOSRunbyrunSlot::clear()
{
  for (int mod = 0; mod < 8; mod++) {
    mReMi[mod].reset();
  }
}

//==============================================================

PHOSRunbyrunCalibrator::PHOSRunbyrunCalibrator()
{
  const int nMass = 150.;
  const float massMax = 0.3;
  for (int mod = 0; mod < 4; mod++) {
    mReMi[2 * mod] = new TH1F(Form("hReInvMassMod%d", mod), "Real inv. mass per module", nMass, 0., massMax);
    mReMi[2 * mod + 1] = new TH1F(Form("hMiInvMassMod%d", mod), "Mixed inv. mass per module", nMass, 0., massMax);
  }
}
PHOSRunbyrunCalibrator::~PHOSRunbyrunCalibrator()
{
  for (int mod = 0; mod < 8; mod++) {
    if (mReMi[mod]) {
      mReMi[mod]->Delete();
      mReMi[mod] = nullptr;
    }
  }
}

bool PHOSRunbyrunCalibrator::hasEnoughData(const Slot& slot) const
{
  // otherwize will be merged with next Slot
  return true;
}
void PHOSRunbyrunCalibrator::initOutput()
{
}
void PHOSRunbyrunCalibrator::finalizeSlot(Slot& slot)
{

  // Extract results for the single slot
  PHOSRunbyrunSlot* c = slot.getContainer();
  LOG(info) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();
  // Add histos
  for (int mod = 0; mod < 8; mod++) {
    PHOSRunbyrunSlot::boostHisto& tmp = c->getCollectedHistos(mod);
    int indx = 1;
    for (auto&& x : boost::histogram::indexed(tmp)) {
      mReMi[mod]->SetBinContent(indx, x.get() + mReMi[mod]->GetBinContent(indx));
      indx++;
    }
  }
  c->clear();
}

Slot& PHOSRunbyrunCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<PHOSRunbyrunSlot>(mUseCCDB, mCCDBPath));
  return slot;
}

bool PHOSRunbyrunCalibrator::process(TFType tf, const gsl::span<const Cluster>& clu, const gsl::span<const TriggerRecord>& tr)
{

  // process current TF
  auto& slotTF = getSlotForTF(tf);
  slotTF.getContainer()->setRunStartTime(tf);
  slotTF.getContainer()->fill(clu, tr);

  return true;
}
void PHOSRunbyrunCalibrator::writeHistos()
{

  // Merge collected in different slots histograms
  TF1 fRatio("ratio", this, &PHOSRunbyrunCalibrator::CBRatio, 0, 1, 6, "PHOSRunbyrunCalibrator", "CBRatio");
  TF1 fBg("background", this, &PHOSRunbyrunCalibrator::bg, 0, 1, 6, "PHOSRunbyrunCalibrator", "bg");
  TF1 fSignal("signal", this, &PHOSRunbyrunCalibrator::CBSignal, 0, 1, 6, "PHOSRunbyrunCalibrator", "CBSignal");
  // fit inv mass distributions

  for (int mod = 0; mod < 4; mod++) {
    mReMi[2 * mod]->Sumw2();
    mReMi[2 * mod + 1]->Sumw2();
    TH1D* tmp = (TH1D*)mReMi[2 * mod]->Clone("Ratio");
    tmp->Divide(mReMi[2 * mod + 1]);
    fRatio.SetParameters(1., 0.134, 0.005, 0.01, 0., 0.);
    tmp->Fit(&fRatio, "q", "", 0.07, 0.22);
    fBg.SetParameters(fRatio.GetParameter(3), fRatio.GetParameter(4), fRatio.GetParameter(5));
    mReMi[2 * mod + 1]->Multiply(&fBg);
    // mReMi[2*mod]->Add(mReMi[2*mod+1],-1.) ;
    fSignal.SetParameters(0.3 * mReMi[2 * mod]->Integral(65, 67, ""), 0.135, 0.005);
    mReMi[2 * mod]->Fit(&fSignal, "q", "", 0.07, 0.22);
    mRunByRun[2 * mod] = fSignal.GetParameter(1);
    mRunByRun[2 * mod + 1] = fSignal.GetParError(1);
    tmp->Write();
    mReMi[2 * mod]->Write();
    mReMi[2 * mod + 1]->Write();
    delete tmp;
    // Clear before next period?
    mReMi[2 * mod]->Reset();
    mReMi[2 * mod + 1]->Reset();
  }

  LOG(info) << "Wrote Run-by-run calibration histos";
}
double PHOSRunbyrunCalibrator::CBRatio(double* x, double* par)
{
  double m = par[1];
  double s = par[2];
  const double n = 4.1983;
  const double a = 1.5;
  const double A = TMath::Power((n / TMath::Abs(a)), n) * TMath::Exp(-a * a / 2);
  const double B = n / TMath::Abs(a) - TMath::Abs(a);
  double dx = (x[0] - m) / s;
  if (dx > -a) {
    return par[0] * exp(-dx * dx / 2.) +
           par[3] + par[4] * x[0] + par[5] * x[0] * x[0];
  } else {
    return par[0] * A * TMath::Power((B - dx), -n) +
           par[3] + par[4] * x[0] + par[5] * x[0] * x[0];
  }
}
double PHOSRunbyrunCalibrator::CBSignal(double* x, double* par)
{
  double m = par[1];
  double s = par[2];
  const double n = 4.1983;
  const double a = 1.5;
  const double A = TMath::Power((n / TMath::Abs(a)), n) * TMath::Exp(-a * a / 2);
  const double B = n / TMath::Abs(a) - TMath::Abs(a);
  double dx = (x[0] - m) / s;
  if (dx > -a) {
    return par[0] * exp(-dx * dx / 2.) + par[3];
  } else {
    return par[0] * A * TMath::Power((B - dx), -n) + par[3];
  }
}
double PHOSRunbyrunCalibrator::bg(double* x, double* par)
{
  return par[0] + par[1] * x[0] + par[2] * x[0] * x[0];
}

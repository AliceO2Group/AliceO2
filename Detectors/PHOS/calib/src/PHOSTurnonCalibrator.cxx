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

#include "PHOSCalibWorkflow/PHOSTurnonCalibrator.h"
#include "PHOSCalibWorkflow/TurnOnHistos.h"
#include "PHOSBase/Geometry.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/ControlService.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"

#include "TF1.h"
#include "TH1.h"
#include "TGraphAsymmErrors.h"

#include <fairlogger/Logger.h>
#include <fstream> // std::ifstream

using namespace o2::phos;

PHOSTurnonSlot::PHOSTurnonSlot(bool useCCDB) : mUseCCDB(useCCDB)
{
  mFiredTiles.reset();
  mNoisyTiles.reset();
  mTurnOnHistos = std::make_unique<TurnOnHistos>();
}
PHOSTurnonSlot::PHOSTurnonSlot(const PHOSTurnonSlot& other)
{
  mUseCCDB = other.mUseCCDB;
  mRunStartTime = other.mUseCCDB;
  mFiredTiles.reset();
  mNoisyTiles.reset();
  mTurnOnHistos = std::make_unique<TurnOnHistos>();
}

void PHOSTurnonSlot::print() const
{
  for (short ddl = 0; ddl < 14; ddl++) {
    const std::array<float, TurnOnHistos::Npt>& all = mTurnOnHistos->getTotSpectrum(ddl);
    const std::array<float, TurnOnHistos::Npt>& tr = mTurnOnHistos->getTrSpectrum(ddl);
    float sumAll = 0, sumTr = 0.;
    for (int i = 0; i < TurnOnHistos::Npt; i++) {
      sumAll += all[i];
      sumTr += tr[i];
    }
    LOG(info) << "DDL " << ddl << " total entries " << sumAll << " trigger clusters " << sumTr;
  }
}
void PHOSTurnonSlot::fill(const gsl::span<const Cell>& cells, const gsl::span<const TriggerRecord>& cellTR,
                          const gsl::span<const Cluster>& clusters, const gsl::span<const TriggerRecord>& cluTR)
{

  auto ctr = cellTR.begin();
  auto clutr = cluTR.begin();
  while (ctr != cellTR.end() && clutr != cluTR.end()) {
    //
    // TODO! select NOT PHOS triggered events
    // DataProcessingHeader::
    //
    if (ctr->getBCData() == clutr->getBCData()) {
      scanClusters(cells, *ctr, clusters, *clutr);
    } else {
      LOG(error) << "Different TrigRecords for cells:" << ctr->getBCData() << " and clusters:" << clutr->getBCData();
    }
    ctr++;
    clutr++;
  }
}
void PHOSTurnonSlot::clear()
{
  mFiredTiles.reset();
  mNoisyTiles.reset();
  mTurnOnHistos.reset();
}
void PHOSTurnonSlot::scanClusters(const gsl::span<const Cell>& cells, const TriggerRecord& celltr,
                                  const gsl::span<const Cluster>& clusters, const TriggerRecord& clutr)
{
  // First fill map of expected tiles from TRU cells
  mNoisyTiles.reset();
  int firstCellInEvent = celltr.getFirstEntry();
  int lastCellInEvent = firstCellInEvent + celltr.getNumberOfObjects();
  for (int i = firstCellInEvent; i < lastCellInEvent; i++) {
    const Cell& c = cells[i];
    if (c.getTRU()) {
      mNoisyTiles.set(c.getTRUId() - Geometry::getTotalNCells() - 1);
    }
  }

  // Copy to have good and noisy map
  mFiredTiles.reset();
  char mod;
  float x, z;
  short ddl;
  int firstCluInEvent = clutr.getFirstEntry();
  int lastCluInEvent = firstCluInEvent + clutr.getNumberOfObjects();
  for (int i = firstCluInEvent; i < lastCluInEvent; i++) {
    const Cluster& clu = clusters[i];
    if (clu.getEnergy() < 1.e-4) {
      continue;
    }
    mod = clu.module();
    clu.getLocalPosition(x, z);
    // TODO: do we need separate 2x2 and 4x4 spectra? Switch?
    //  short truId2x2 = Geometry::relPosToTruId(mod, x, z, 0);
    //  short truId4x4 = Geometry::relPosToTruId(mod, x, z, 1);
    mTurnOnHistos->fillTotSp(ddl, clu.getEnergy());
    //     if (clu.firedTrigger() & 1) { //Bit 1: 2x2, bit 2 4x4
    //       mTurnOnHistos->fillFiredSp(ddl, clu.getEnergy());
    //       //Fill trigger map
    //       mFiredTiles.set(truId);
    //     }
    // }
  }
  // Fill final good and noisy maps
  mTurnOnHistos->fillFiredMap(mFiredTiles);
  mNoisyTiles ^= mFiredTiles;
  mTurnOnHistos->fillNoisyMap(mFiredTiles);
}
//==============================================

void PHOSTurnonCalibrator::finalizeSlot(Slot& slot)
{
  // Extract results for the single slot
  // if not ready yet, prepare containers
  if (!mTurnOnHistos) {
    mTurnOnHistos.reset(new TurnOnHistos());
  }
  PHOSTurnonSlot* c = slot.getContainer();
  LOG(info) << "Finalize slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();
  // Add histos
  for (int mod = 0; mod < 8; mod++) {
    mTurnOnHistos->merge(c->getCollectedHistos());
  }
  c->clear();
}
PHOSTurnonCalibrator::Slot& PHOSTurnonCalibrator::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{

  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);
  slot.setContainer(std::make_unique<PHOSTurnonSlot>(mUseCCDB));
  return slot;
}

bool PHOSTurnonCalibrator::process(uint64_t tf, const gsl::span<const Cell>& cells, const gsl::span<const TriggerRecord>& cellTR,
                                   const gsl::span<const Cluster>& clusters, const gsl::span<const TriggerRecord>& cluTR)
{
  // process current TF
  auto& slotTF = getSlotForTF(tf);
  slotTF.getContainer()->setRunStartTime(tf);
  slotTF.getContainer()->fill(cells, cellTR, clusters, cluTR);

  return true;
}

void PHOSTurnonCalibrator::endOfStream()
{
  // Use stored histos to calculate maps and turn-on curves
  // return true of successful

  // extract TOC
  if (!mTriggerMap) {
    mTriggerMap.reset(new TriggerMap());
  }
  TF1* th = new TF1("aTh", "[0]/(TMath::Exp(([1]-x)/[2])+1.)+(1.-[0])/(TMath::Exp(([3]-x)/[2])+1.)", 0., 40.);
  std::array<std::array<float, 10>, TurnOnHistos::NDDL> params;
  for (int ddl = 0; ddl < TurnOnHistos::NDDL; ddl++) {
    TH1F hF("fired", "fired", 200, 0., 20.);
    TH1F hA("all", "all", 200, 0., 20.);
    const std::array<float, TurnOnHistos::Npt>& vf = mTurnOnHistos->getTrSpectrum(ddl);
    const std::array<float, TurnOnHistos::Npt>& va = mTurnOnHistos->getTotSpectrum(ddl);
    for (int i = 0; i < 200; i++) {
      hF.SetBinContent(i + 1, vf[i]);
      hA.SetBinContent(i + 1, va[i]);
    }
    hF.Sumw2();
    hA.Sumw2();

    TGraphAsymmErrors* gr = new TGraphAsymmErrors(&hF, &hA);
    th->SetParameters(0.9, 3.5, 0.3, 7.5, 0.6);
    gr->Fit(th, "Q", "", 2., 20.);
    gr->SetName(Form("DDL_%d", ddl));
    gr->SetTitle(Form("DDL %d", ddl));
    // TODO!!! Add TGraph with fit to list of objects to send to QC
    double* par = th->GetParameters();
    for (int i = 0; i < 10; i++) {
      params[ddl][i] = par[i];
    }
  }
  std::string_view versionName{"default"};
  mTriggerMap->addTurnOnCurvesParams(versionName, params);
  // TODO: calculate bad map
  // and fill object
  // mTriggerMap->addBad2x2Channel(short cellID) ;
}

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

/// \file  FastMultEst.h
/// \brief Fast multiplicity estimator for ITS
/// \author ruben.shahoyan@cern.ch

#include "ITSReconstruction/FastMultEst.h"
#include "ITSMFTBase/DPLAlpideParam.h"
#include "Framework/Logger.h"
#include <ctime>
#include <cstring>
#include <TRandom.h>

using namespace o2::its;

bool FastMultEst::sSeedSet = false;

///______________________________________________________
FastMultEst::FastMultEst()
{
  if (!sSeedSet && FastMultEstConfig::Instance().cutRandomFraction > 0.f) {
    sSeedSet = true;
    if (FastMultEstConfig::Instance().randomSeed > 0) {
      gRandom->SetSeed(FastMultEstConfig::Instance().randomSeed);
    } else if (FastMultEstConfig::Instance().randomSeed < 0) {
      gRandom->SetSeed(std::time(nullptr) % 0xffff);
    }
  }
}

///______________________________________________________
/// find multiplicity for given set of clusters
void FastMultEst::fillNClPerLayer(const gsl::span<const o2::itsmft::CompClusterExt>& clusters)
{
  int lr = FastMultEst::NLayers - 1, nchAcc = o2::itsmft::ChipMappingITS::getNChips() - o2::itsmft::ChipMappingITS::getNChipsPerLr(lr);
  std::memset(&nClPerLayer[0], 0, sizeof(int) * FastMultEst::NLayers);
  for (int i = clusters.size(); i--;) { // profit from clusters being ordered in chip increasing order
    while (clusters[i].getSensorID() < nchAcc) {
      assert(lr >= 0);
      nchAcc -= o2::itsmft::ChipMappingITS::getNChipsPerLr(--lr);
    }
    nClPerLayer[lr]++;
  }
}

///______________________________________________________
/// find multiplicity for given number of clusters per layer
float FastMultEst::processNoiseFree(const std::array<int, NLayers> ncl)
{
  // we assume that on the used layers the observed number of clusters is defined by the
  // the noise ~ nu * Nchips and contribution from the signal tracks Ntr*mAccCorr
  const auto& conf = FastMultEstConfig::Instance();

  float mat[3] = {0}, b[2] = {0};
  nLayersUsed = 0;
  for (int il = conf.firstLayer; il <= conf.lastLayer; il++) {
    if (ncl[il] > 0) {
      int nch = o2::itsmft::ChipMappingITS::getNChipsPerLr(il);
      float err2i = 1. / ncl[il];
      float m2n = nch * err2i;
      mat[0] += err2i * conf.accCorr[il] * conf.accCorr[il];
      mat[2] += nch * m2n;
      mat[1] += conf.accCorr[il] * m2n; // non-diagonal element
      b[0] += conf.accCorr[il];
      b[1] += nch;
      nLayersUsed++;
    }
  }
  mult = noisePerChip = chi2 = -1;
  float det = mat[0] * mat[2] - mat[1] * mat[1];
  if (nLayersUsed < 2 || std::abs(det) < 1e-15) {
    return -1;
  }
  float detI = 1. / det;
  mult = detI * (b[0] * mat[2] - b[1] * mat[1]);
  noisePerChip = detI * (b[1] * mat[0] - b[0] * mat[1]);
  cov[0] = mat[2] * detI;
  cov[2] = mat[0] * detI;
  cov[1] = -mat[1] * detI;
  chi2 = 0.;
  for (int il = conf.firstLayer; il <= conf.lastLayer; il++) {
    if (ncl[il] > 0) {
      int nch = o2::itsmft::ChipMappingITS::getNChipsPerLr(il);
      float diff = mult * conf.accCorr[il] + nch * noisePerChip - ncl[il];
      chi2 += diff * diff / ncl[il];
    }
  }
  chi2 = nLayersUsed > 2 ? chi2 / (nLayersUsed - 2) : 0.;
  return mult > 0 ? mult : 0;
}

///______________________________________________________
/// find multiplicity for given number of clusters per layer with mean noise imposed
float FastMultEst::processNoiseImposed(const std::array<int, NLayers> ncl)
{
  // we assume that on the used layers the observed number of clusters is defined by the
  // the noise ~ nu * Nchips and contribution from the signal tracks Ntr*conf.accCorr
  //
  const auto& conf = FastMultEstConfig::Instance();
  float accSum = 0., accWSum = 0., noiseSum = 0.;
  nLayersUsed = 0;
  for (int il = conf.firstLayer; il <= conf.lastLayer; il++) {
    if (ncl[il] > 0) {
      float err = 1. / ncl[il];
      accSum += conf.accCorr[il];
      accWSum += conf.accCorr[il] * conf.accCorr[il] * err;
      noiseSum += o2::itsmft::ChipMappingITS::getNChipsPerLr(il) * conf.accCorr[il] * err;
      nLayersUsed++;
    }
  }
  mult = 0;
  if (nLayersUsed) {
    mult = (accSum - noisePerChip * noiseSum) / accWSum;
  }
  return mult;
}

int FastMultEst::selectROFs(const gsl::span<const o2::itsmft::ROFRecord> rofs, const gsl::span<const o2::itsmft::CompClusterExt> clus,
                            const gsl::span<const o2::itsmft::PhysTrigger> trig, std::vector<bool>& sel)
{
  int nrof = rofs.size(), nsel = 0;
  const auto& multEstConf = FastMultEstConfig::Instance(); // parameters for mult estimation and cuts
  sel.clear();
  sel.resize(nrof, true); // by default select all
  lastRandomSeed = gRandom->GetSeed();
  if (multEstConf.isMultCutRequested()) {
    for (uint32_t irof = 0; irof < nrof; irof++) {
      nsel += sel[irof] = multEstConf.isPassingMultCut(process(rofs[irof].getROFData(clus)));
    }
  } else {
    nsel = nrof;
  }
  using IdNT = std::pair<int, int>;
  if (multEstConf.cutRandomFraction > 0.) {
    int ntrig = trig.size(), currTrig = 0;
    if (multEstConf.preferTriggered) {
      const auto& alpParams = o2::itsmft::DPLAlpideParam<o2::detectors::DetID::ITS>::Instance();
      std::vector<IdNT> nTrigROF;
      nTrigROF.reserve(nrof);
      for (uint32_t irof = 0; irof < nrof; irof++) {
        if (sel[irof]) {
          if (nsel && gRandom->Rndm() < multEstConf.cutRandomFraction) {
            nsel--;
          }
          auto irROF = rofs[irof].getBCData();
          while (currTrig < ntrig && trig[currTrig].ir < irROF) { // triggers are sorted, jump to 1st one not less than current ROF
            currTrig++;
          }
          auto& trof = nTrigROF.emplace_back(irof, 0);
          irROF += alpParams.roFrameLengthInBC;
          while (currTrig < ntrig && trig[currTrig].ir < irROF) {
            trof.second++;
            currTrig++;
          }
        }
      }
      if (nsel > 0) {
        sort(nTrigROF.begin(), nTrigROF.end(), [](const IdNT& a, const IdNT& b) { return a.second > b.second; }); // order in number of triggers
        auto last = nTrigROF.begin() + nsel;
        sort(nTrigROF.begin(), last, [](const IdNT& a, const IdNT& b) { return a.first < b.first; }); // order in ROF ID first nsel ROFs
      }
      for (int i = nsel; i < int(nTrigROF.size()); i++) { // reject ROFs in the tail
        sel[nTrigROF[i].first] = false;
      }
    } else { // dummy random rejection
      for (int irof = 0; irof < nrof; irof++) {
        if (sel[irof]) {
          float sr = gRandom->Rndm();
          if (gRandom->Rndm() < multEstConf.cutRandomFraction) {
            sel[irof] = false;
            nsel--;
          }
        }
      }
    }
  }
  LOGP(debug, "NSel = {} of {} rofs Seeds: before {} after {}", nsel, nrof, lastRandomSeed, gRandom->GetSeed());

  return nsel;
}

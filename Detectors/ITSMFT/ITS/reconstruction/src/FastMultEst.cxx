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
#include <cstring>
#include "Framework/Logger.h"

using namespace o2::its;

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
  // minimize the form sum_lr (noise_i - mu nchips_i)^2 / (mu nchips_i) + lambda_i * (noise_i + mult*acc_i - ncl_i)
  // whith noise_i being estimate of the noise clusters in nchips_i of layer i, mu is the mean noise per chip,
  // mult is the number of signal clusters on the ref. (1st) layer and the acc_i is the acceptance of layer i wrt 1st.
  // The lambda_i is hust a Lagrange multiplier.

  const auto& conf = FastMultEstConfig::Instance();
  float w2sum = 0., wnsum = 0., wsum = 0.;
  nLayersUsed = 0;
  for (int il = conf.firstLayer; il <= conf.lastLayer; il++) {
    if (ncl[il] > 0) {
      float nchInv = 1. / o2::itsmft::ChipMappingITS::getNChipsPerLr(il);
      w2sum += conf.accCorr[il] * conf.accCorr[il] * nchInv;
      wnsum += ncl[il] * nchInv * conf.accCorr[il];
      wsum += conf.accCorr[il];
      nLayersUsed++;
    }
  }
  mult = 0;
  chi2 = -1;
  noisePerChip = conf.imposeNoisePerChip;
  if (nLayersUsed < 1) {
    return -1;
  }
  auto w2sumI = 1. / w2sum;
  mult = (wnsum - noisePerChip * wsum) * w2sumI;
  cov[0] = wnsum * w2sumI;
  cov[2] = 0.;
  cov[1] = 0.;

  chi2 = 0.;
  for (int il = conf.firstLayer; il <= conf.lastLayer; il++) {
    if (ncl[il] > 0) {
      float noise = ncl[il] - mult * conf.accCorr[il], estNoise = o2::itsmft::ChipMappingITS::getNChipsPerLr(il) * noisePerChip;
      float diff = noise - estNoise;
      chi2 += diff * diff / estNoise;
    }
  }
  chi2 = nLayersUsed > 2 ? chi2 / (nLayersUsed - 2) : 0.;
  return mult > 0 ? mult : 0;
}

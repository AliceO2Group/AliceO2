// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUO2InterfaceQA.cxx
/// \author David Rohr

#include "GPUParam.h"
#include "GPUQA.h"
#include "GPUO2InterfaceConfiguration.h"
#include "GPUO2InterfaceQA.h"

using namespace o2::gpu;
using namespace o2::tpc;

GPUO2InterfaceQA::GPUO2InterfaceQA(const GPUO2InterfaceConfiguration* config)
{
  mParam.reset(new GPUParam);
  mParam->SetDefaults(&config->configEvent, &config->configReconstruction, &config->configProcessing, nullptr);
  mQA.reset(new GPUQA(nullptr, &config->configQA, mParam.get()));
}

GPUO2InterfaceQA::~GPUO2InterfaceQA() = default;

int GPUO2InterfaceQA::initializeForProcessing(int tasks)
{
  return mQA->InitQA(tasks);
}

void GPUO2InterfaceQA::runQA(const std::vector<o2::tpc::TrackTPC>* tracksExternal, const std::vector<o2::MCCompLabel>* tracksExtMC, const o2::tpc::ClusterNativeAccess* clNative)
{
  mQA->RunQA(false, tracksExternal, tracksExtMC, clNative);
}
int GPUO2InterfaceQA::postprocess(TObjArray& out)
{
  return mQA->DrawQAHistograms(&out);
}

int GPUO2InterfaceQA::postprocessExternal(std::vector<TH1F>& in1, std::vector<TH2F>& in2, std::vector<TH1D>& in3, TObjArray& out, int tasks)
{
  if (mQA->loadHistograms(in1, in2, in3, tasks)) {
    return 1;
  }
  return mQA->DrawQAHistograms(&out);
}

void GPUO2InterfaceQA::cleanup()
{
  mQA->DrawQAHistogramsCleanup();
}

void GPUO2InterfaceQA::getHists(const std::vector<TH1F>*& h1, const std::vector<TH2F>*& h2, const std::vector<TH1D>*& h3)
{
  h1 = &mQA->getHistograms1D();
  h2 = &mQA->getHistograms2D();
  h3 = &mQA->getHistograms1Dd();
}

void GPUO2InterfaceQA::resetHists()
{
  mQA->resetHists();
}

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

#include "GPUQA.h"
#include "GPUO2InterfaceQA.h"

using namespace o2::gpu;
using namespace o2::tpc;

GPUO2InterfaceQA::GPUO2InterfaceQA(const GPUSettingsQA* config) : mQA(new GPUQA(nullptr, config))
{
}

GPUO2InterfaceQA::~GPUO2InterfaceQA() = default;

int GPUO2InterfaceQA::postprocess(std::vector<TH1F>& in1, std::vector<TH2F>& in2, std::vector<TH1D>& in3, TObjArray& out)
{
  if (mQA->loadHistograms(in1, in2, in3)) {
    return 1;
  }
  return mQA->DrawQAHistograms(&out);
}

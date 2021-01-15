// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Tracking.cxx
/// \author David Rohr

#define _USE_MATH_DEFINES

#include <cmath>

//root includes
#include "TStyle.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TH1D.h"

//o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "TPCQC/Tracking.h"
#include "GPUO2InterfaceQA.h"
#include "GPUO2InterfaceConfiguration.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DataFormatsParameters/GRPObject.h"

ClassImp(o2::tpc::qc::Tracking);

using namespace o2::tpc::qc;
using namespace o2::gpu;

Tracking::Tracking() = default;
Tracking::~Tracking() = default;

static constexpr int QAMODE = 7; // Efficiency (1) | Res (2) | Pull (3), others are not supported from external input!

//______________________________________________________________________________
void Tracking::initialize(outputModes outputMode, bool postprocessOnly)
{
  mOutputMode = outputMode;
  mQAConfig = std::make_unique<GPUO2InterfaceConfiguration>();
  const auto grp = o2::parameters::GRPObject::loadFrom(o2::base::NameConf::getGRPFileName());
  if (grp) {
    mQAConfig->configEvent.solenoidBz = 5.00668f * grp->getL3Current() / 30000.;
    mQAConfig->configEvent.continuousMaxTimeBin = grp->isDetContinuousReadOut(o2::detectors::DetID::TPC) ? -1 : 0;
  } else {
    throw std::runtime_error("Failed to initialize run parameters from GRP");
  }
  mQAConfig->ReadConfigurableParam();
  mQAConfig->configQA.shipToQCAsCanvas = mOutputMode == outputLayout;
  mQA = std::make_unique<GPUO2InterfaceQA>(mQAConfig.get());
  if (!postprocessOnly) {
    mQA->initializeForProcessing(QAMODE);
  }
}

//______________________________________________________________________________
void Tracking::resetHistograms()
{
  mQA->resetHists();
}

//______________________________________________________________________________
void Tracking::processTracks(const std::vector<o2::tpc::TrackTPC>* tracks, const std::vector<o2::MCCompLabel>* tracksMC, const o2::tpc::ClusterNativeAccess* clNative, TObjArray* out)
{
  mQA->runQA(tracks, tracksMC, clNative);
  if (mOutputMode == outputPostprocessed || mOutputMode == outputLayout) {
    mQA->postprocess(*out);
  }
}

int Tracking::postprocess(std::vector<TH1F>& in1, std::vector<TH2F>& in2, std::vector<TH1D>& in3, TObjArray& out)
{
  return mQA->postprocessExternal(in1, in2, in3, out, QAMODE);
}

void Tracking::getHists(const std::vector<TH1F>*& h1, const std::vector<TH2F>*& h2, const std::vector<TH1D>*& h3) const
{
  mQA->getHists(h1, h2, h3);
}

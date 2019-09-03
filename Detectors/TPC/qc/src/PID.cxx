// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cmath>

#include "DataFormatsTPC/dEdxInfo.h"

#include "TPCQC/PID.h"

ClassImp(o2::tpc::qc::PID);

using namespace o2::tpc::qc;

PID::PID() : mHist1D{}
{
}

void PID::initializeHistograms()
{
  mHist1D.emplace_back("hMIP", "MIP position;MIP (ADC counts)", 100, 0, 100);
}

void PID::resetHistograms()
{
  for (auto& hist : mHist1D) {
    hist.Reset();
  }
}

bool PID::processTrack(o2::tpc::TrackTPC const& track)
{
  // ===| variables required for cutting and filling |===
  const auto p = track.getP();
  const auto dEdx = track.getdEdx().dEdxTotTPC;

  // ===| cuts |===
  // hard coded cuts. Should be more configural in future
  if (std::abs(p - 0.5) > 0.05) {
    return false;
  }

  if (dEdx > 70) {
    return false;
  }

  // ===| histogram filling |===
  mHist1D[0].Fill(dEdx);

  return true;
}

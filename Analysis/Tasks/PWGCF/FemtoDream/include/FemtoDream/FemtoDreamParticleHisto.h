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

/// \file FemtoDreamParticleHisto.h
/// \brief FemtoDreamParticleHisto - Histogram class for tracks
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ANALYSIS_TASKS_PWGCF_O2FEMTODREAM_INCLUDE_O2FEMTODREAM_FEMTODREAMPARTICLEHISTO_H_
#define ANALYSIS_TASKS_PWGCF_O2FEMTODREAM_INCLUDE_O2FEMTODREAM_FEMTODREAMPARTICLEHISTO_H_

#include "Framework/HistogramRegistry.h"
#include <Rtypes.h>

using namespace o2::framework;

namespace o2::analysis::femtoDream
{
class FemtoDreamParticleHisto
{
 public:
  virtual ~FemtoDreamParticleHisto() = default;

  void init(HistogramRegistry* registry)
  {
    if (registry) {
      mHistogramRegistry = registry;
      /// \todo how to do the naming for track - track combinations?
      mHistogramRegistry->add("Tracks/pThist", "; #it{p}_{T} (GeV/#it{c}); Entries", kTH1F, {{1000, 0, 10}});
      mHistogramRegistry->add("Tracks/etahist", "; #eta; Entries", kTH1F, {{1000, -1, 1}});
      mHistogramRegistry->add("Tracks/phihist", "; #phi; Entries", kTH1F, {{1000, 0, 2. * M_PI}});
      mHistogramRegistry->add("Tracks/dcaXYhist", "; #it{p}_{T} (GeV/#it{c}); DCA_{xy} (cm)", kTH2F, {{100, 0, 10}, {501, -3, 3}});
    }
  }

  template <typename T>
  void fillQA(T const& track)
  {
    if (mHistogramRegistry) {
      mHistogramRegistry->fill(HIST("TrackCuts/pThist"), track.pt());
      mHistogramRegistry->fill(HIST("TrackCuts/etahist"), track.eta());
      mHistogramRegistry->fill(HIST("TrackCuts/phihist"), track.phi());
      mHistogramRegistry->fill(HIST("TrackCuts/dcaXYhist"), track.pt(), track.tempFitVar());
    }
  }

 private:
  HistogramRegistry* mHistogramRegistry; ///< For QA output

  ClassDefNV(FemtoDreamParticleHisto, 1);
};
} // namespace o2::analysis::femtoDream

#endif /* ANALYSIS_TASKS_PWGCF_O2FEMTODREAM_INCLUDE_O2FEMTODREAM_FEMTODREAMPARTICLEHISTO_H_ */

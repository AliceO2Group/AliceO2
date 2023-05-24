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

#ifndef O2_MCH_EVALUATION_HISTOS_H__
#define O2_MCH_EVALUATION_HISTOS_H__

#include <array>
#include <list>
#include <vector>
class TH1;

namespace o2::mch
{
class TrackParam;

namespace eval
{
class ExtendedTrack;

void createHistosAtVertex(std::vector<TH1*>& histos, const char* extension);

void createHistosResiduals(std::vector<TH1*>& histos, const char* extension, double range);

void createHistosResidualsAtFirstCluster(std::vector<TH1*>& histos);

void fillComparisonsAtVertex(std::list<ExtendedTrack>& tracks1, std::list<ExtendedTrack>& tracks2, const std::array<std::vector<TH1*>, 5>& histos);

void fillHistosAtVertex(const std::list<ExtendedTrack>& tracks, const std::vector<TH1*>& histos);

void fillHistosDimuAtVertex(const ExtendedTrack& track1, const ExtendedTrack& track2, const std::vector<TH1*>& histos);

void fillHistosMuAtVertex(const ExtendedTrack& track, const std::vector<TH1*>& histos);

/** fill histograms of residuals at given param */
void fillResiduals(const TrackParam& param1, const TrackParam& param2, std::vector<TH1*>& histos);

/** fill histograms of cluster-cluster residuals*/
void fillResiduals(const ExtendedTrack& track1, const ExtendedTrack& track2, std::vector<TH1*>& histos);

/** fill histograms of cluster-track residuals */
void fillResiduals(const std::list<ExtendedTrack>& tracks, std::vector<TH1*>& histos, bool matched);
} // namespace eval
} // namespace o2::mch

#endif

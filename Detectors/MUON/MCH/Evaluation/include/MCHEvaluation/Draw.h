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

#ifndef O2_MCH_EVALUATION_DRAW_H__
#define O2_MCH_EVALUATION_DRAW_H__

#include <array>
#include <vector>

class TCanvas;
class TH1;

namespace o2::mch::eval
{

void drawAll(const char* filename);

TCanvas* autoCanvas(const char* title, const char* name,
                    const std::vector<TH1*>& histos,
                    int* nPadsx = nullptr,
                    int* nPadsy = nullptr);

void drawTrackResiduals(std::vector<TH1*>& histos, TCanvas* c = nullptr);
void drawClusterResiduals(const std::array<std::vector<TH1*>, 5>& histos, TCanvas* c = nullptr);
void drawClusterClusterResiduals(const std::vector<TH1*>& histos, const char* extension, TCanvas* c = nullptr);
void drawClusterTrackResiduals(const std::vector<TH1*>& histos1, const std::vector<TH1*>& histos2, const char* extension, TCanvas* c = nullptr);
void drawClusterTrackResidualsSigma(const std::vector<TH1*>& histos1, const std::vector<TH1*>& histos2, const char* extension, TCanvas* c1 = nullptr, TCanvas* c2 = nullptr);
void drawClusterTrackResidualsRatio(const std::vector<TH1*>& histos1, const std::vector<TH1*>& histos2, const char* extension, TCanvas* c = nullptr);

void drawHistosAtVertex(const std::array<std::vector<TH1*>, 2>& histos, TCanvas* c = nullptr);

void drawPlainHistosAtVertex(const std::array<std::vector<TH1*>, 2>& histos, TCanvas* c = nullptr);
void drawDiffHistosAtVertex(const std::array<std::vector<TH1*>, 2>& histos, TCanvas* c = nullptr);
void drawRatioHistosAtVertex(const std::array<std::vector<TH1*>, 2>& histos, TCanvas* c = nullptr);

void drawComparisonsAtVertex(const std::array<std::vector<TH1*>, 5> histos, TCanvas* c = nullptr);

} // namespace o2::mch::eval

#endif

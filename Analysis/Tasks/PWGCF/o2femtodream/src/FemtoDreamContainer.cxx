// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FemtoDreamContainer.cxx
/// \brief Implementation of the FemtoDreamContainer
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "o2femtodream/FemtoDreamContainer.h"
#include <string>

using namespace o2::analysis::femtoDream;

FemtoDreamContainer::FemtoDreamContainer()
  : mFemtoObs(Observable::kstar),
    mHistogramRegistry(nullptr),
    mMassOne(0.f),
    mMassTwo(0.f)
{
}

FemtoDreamContainer::FemtoDreamContainer(HistogramRegistry* registry, Observable obs)
  : mFemtoObs(obs),
    mHistogramRegistry(registry),
    mMassOne(0.f),
    mMassTwo(0.f)
{
  std::string femtoObs;
  if (mFemtoObs == Observable::kstar) {
    femtoObs = "#it{k}^{*} (GeV/#it{c})";
  }
  mHistogramRegistry->add("relPairDist", ("; " + femtoObs + "; Entries").c_str(), kTH1F, {{5000, 0, 5}});
}

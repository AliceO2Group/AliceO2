// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrivialClusterer.cxx
/// \brief Implementation of the ITS cluster finder
#include "MathUtils/Cartesian3D.h"
#include "ITSMFTBase/Digit.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSReconstruction/TrivialClusterer.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "FairLogger.h" // for LOG

using o2::itsmft::SegmentationAlpide;
using namespace o2::its;
using namespace o2::itsmft;

using Point3Df = Point3D<float>;

TrivialClusterer::TrivialClusterer() = default;

TrivialClusterer::~TrivialClusterer() = default;

void TrivialClusterer::process(const std::vector<Digit>* digits, std::vector<Cluster>* clusters)
{
  Float_t sigma2 = SegmentationAlpide::PitchRow * SegmentationAlpide::PitchRow / 12.;

  for (const auto& d : *digits) {
    Int_t ix = d.getRow(), iz = d.getColumn();
    Float_t x = 0., y = 0., z = 0.;
    SegmentationAlpide::detectorToLocal(ix, iz, x, z);
    Point3Df loc(x, 0.f, z);
    // inverse transform from local to tracking frame
    auto tra = mGeometry->getMatrixT2L(d.getChipIndex()) ^ (loc);

    int noc = clusters->size();
    clusters->emplace_back(d.getChipIndex(), tra, sigma2, sigma2, 0.);
    (*clusters)[noc].SetUniqueID(noc); // Save the index within the cluster array
    if (mClsLabels) {
      /*
      for (int i=0; i<Digit::maxLabels; i++) {
        Label lab = d.getLabel(i);
        if (lab.isEmpty()) break;
        mClsLabels->addElement(noc,lab);
      }
      */
    }
  }
}

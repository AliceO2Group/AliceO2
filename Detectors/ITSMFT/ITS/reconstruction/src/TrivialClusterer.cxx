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
#include "ITSMFTBase/SegmentationPixel.h"
#include "ITSReconstruction/TrivialClusterer.h"
#include "ITSMFTReconstruction/Cluster.h"

#include "FairLogger.h"   // for LOG
#include "TClonesArray.h" // for TClonesArray

using o2::ITSMFT::SegmentationPixel;
using o2::ITSMFT::Digit;
using namespace o2::ITS;
using namespace o2::ITSMFT;

using Point3Df = Point3D<float>;

TrivialClusterer::TrivialClusterer() = default;

TrivialClusterer::~TrivialClusterer() = default;

void
TrivialClusterer::process(const SegmentationPixel *seg, const TClonesArray* digits, TClonesArray* clusters)
{
  Float_t sigma2 = seg->cellSizeX() * seg->cellSizeX() / 12.;

  TClonesArray& clref = *clusters;
  for (TIter digP = TIter(digits).Begin(); digP != TIter::End(); ++digP) {
    Digit* dig = static_cast<Digit*>(*digP);
    Int_t ix = dig->getRow(), iz = dig->getColumn();
    Int_t lab = dig->getLabel(0);
    Float_t x = 0., y = 0., z = 0.;
    seg->detectorToLocal(ix, iz, x, z);
    Point3Df loc(x,0.f,z);
    // inverse transform from local to tracking frame
    auto tra = mGeometry->getMatrixT2L( dig->getChipIndex() )^(loc);
    Cluster c(dig->getChipIndex(), tra, sigma2, sigma2, 0.);
    c.setLabel(lab, 0);

    new (clref[clref.GetEntriesFast()]) Cluster(c);
  }
}

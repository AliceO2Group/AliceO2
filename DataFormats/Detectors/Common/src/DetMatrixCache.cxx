// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DetectorsCommonDataFormats/DetMatrixCache.h"
#include "MathUtils/Utils.h"
#include <TGeoMatrix.h>

using namespace o2::detectors;
using namespace o2::utils;

ClassImp(o2::detectors::MatrixCache<o2::Transform3D>);
ClassImp(o2::detectors::MatrixCache<o2::Rotation2D>);
ClassImp(o2::detectors::DetMatrixCache);


//_______________________________________________________
void DetMatrixCache::setSize(int s)
{
  // set the size of the matrix cache, can be done only once
  if (mSize!=0) {
    LOG(FATAL) << "Cache size (N sensors) was already set to " << mSize << FairLogger::endl;
  }
  mSize = s;
}

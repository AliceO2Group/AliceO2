// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  HitAnalysis.h
//  ALICEO2
//
//  Created by Markus Fasel on 28.07.15.
//
//

#ifndef __ALICEO2__HitAnalysis__
#define __ALICEO2__HitAnalysis__

#include <map>
#include "FairTask.h" // for FairTask, InitStatus
#include "Rtypes.h"   // for Bool_t, HitAnalysis::Class, ClassDef, etc
#include "ITSMFTSimulation/Hit.h"
#include <vector>

class TH1; // lines 16-16
namespace o2
{
namespace its
{
class GeometryTGeo;
}
} // namespace o2

class TH1;

namespace o2
{
namespace itsmft
{
class Chip;
}
} // namespace o2

namespace o2
{
namespace its
{

class HitAnalysis : public FairTask
{
 public:
  HitAnalysis();

  ~HitAnalysis() override;

  InitStatus Init() override;

  void Exec(Option_t* option) override;

  void FinishTask() override;

 protected:
  void ProcessHits();

 private:
  Bool_t mIsInitialized;                     ///< Check whether task is initialized
  const std::vector<o2::itsmft::Hit>* mHits; ///< Array with ITS hits, filled by the FairRootManager
  const GeometryTGeo* mGeometry;             ///<  geometry
  TH1* mLineSegment;                         ///< Histogram for line segment
  TH1* mLocalX0;                             ///< Histogram for Starting X position in local coordinates
  TH1* mLocalX1;                             ///< Histogram for Hit X position in local coordinates
  TH1* mLocalY0;                             ///< Histogram for Starting Y position in local coordinates
  TH1* mLocalY1;                             ///< Histogram for Hit Y position in local coordinates
  TH1* mLocalZ0;                             ///< Histogram for Starting Z position in local coordinates
  TH1* mLocalZ1;                             ///< Histogram for Hit Z position in local coordinates
  TH1* mHitCounter;                          ///< simple hit counter histogram

  ClassDefOverride(HitAnalysis, 1);
};
} // namespace its
} // namespace o2

#endif /* defined(__ALICEO2__HitAnalysis__) */

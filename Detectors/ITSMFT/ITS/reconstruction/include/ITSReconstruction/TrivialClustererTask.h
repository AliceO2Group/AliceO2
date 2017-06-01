// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrivialClustererTask.h
/// \brief Definition of the ITS cluster finder task

#ifndef ALICEO2_ITS_TRIVIALCLUSTERERTASK
#define ALICEO2_ITS_TRIVIALCLUSTERERTASK

#include "FairTask.h" 

#include "ITSBase/GeometryTGeo.h"
#include "ITSReconstruction/TrivialClusterer.h"

class TClonesArray;

namespace o2
{
namespace ITS
{
class TrivialClustererTask : public FairTask
{
 public:
  TrivialClustererTask();
  ~TrivialClustererTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;

 private:
  GeometryTGeo mGeometry; ///< ITS geometry
  TrivialClusterer mTrivialClusterer;   ///< Cluster finder

  TClonesArray* mDigitsArray;   ///< Array of digits
  TClonesArray* mClustersArray; ///< Array of clusters

  ClassDefOverride(TrivialClustererTask, 2)
};
}
}

#endif /* ALICEO2_ITS_TRIVIALCLUSTERERTASK */

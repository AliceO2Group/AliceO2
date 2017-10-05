// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClustererTask.h
/// \brief Definition of the ITS cluster finder task

#ifndef ALICEO2_ITS_CLUSTERERTASK
#define ALICEO2_ITS_CLUSTERERTASK

#include "FairTask.h" 

#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/PixelReader.h"
#include "ITSMFTReconstruction/Clusterer.h"

class TClonesArray;

namespace o2
{
namespace ITS
{
  
class ClustererTask : public FairTask
{
  using DigitPixelReader = o2::ITSMFT::DigitPixelReader;
  using Clusterer = o2::ITSMFT::Clusterer;
  
 public:
  ClustererTask();
  ~ClustererTask() override;

  InitStatus Init() override;
  void Exec(Option_t* option) override;

 private:

  const o2::ITSMFT::GeometryTGeo* mGeometry = nullptr;    ///< ITS OR MFT upgrade geometry
  DigitPixelReader mReader;  ///< Pixel reader
  Clusterer mClusterer;      ///< Cluster finder

  TClonesArray* mClustersArray = nullptr; ///< Array of clusters

  ClassDefOverride(ClustererTask, 1)
};
}
}

#endif /* ALICEO2_ITS_CLUSTERERTASK */

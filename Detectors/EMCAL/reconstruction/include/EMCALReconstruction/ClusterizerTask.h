// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  ClusterizerTask.h
/// \brief Definition of the EMCAL clusterizer task

#ifndef ALICEO2_EMCAL_CLUSTERIZERTASK
#define ALICEO2_EMCAL_CLUSTERIZERTASK

#include "DataFormatsEMCAL/Cluster.h"
#include "DataFormatsEMCAL/Digit.h"
#include "DataFormatsEMCAL/Cell.h"
#include "EMCALBase/Geometry.h"
#include "EMCALReconstruction/DigitReader.h"
#include "EMCALReconstruction/ClusterizerParameters.h"
#include "EMCALReconstruction/Clusterizer.h"

namespace o2
{

namespace emcal
{

/// \class ClusterizerTask
/// \brief Stand-alone task running EMCAL clusterization
/// \ingroup EMCALreconstruction
/// \author Rudiger Haake (Yale)
template <class InputType>
class ClusterizerTask
{

 public:
  ClusterizerTask(ClusterizerParameters* parameters);
  ~ClusterizerTask() = default;

  void init();
  void process(const std::string inputFileName, const std::string outputFileName);
  void setGeometry(Geometry* geometry) { mGeometry = geometry; }
  Geometry* getGeometry() { return mGeometry; }

 private:
  Clusterizer<InputType> mClusterizer;                                             ///< Clusterizer
  Geometry* mGeometry = nullptr;                                                   ///< Pointer to geometry object
  std::unique_ptr<DigitReader<InputType>> mInputReader;                            ///< Pointer to cell/digit reader
  std::vector<Cluster>* mClustersArray = nullptr;                                  ///< Array of clusters
  std::vector<ClusterIndex>* mClustersInputIndices = nullptr;                      ///< Array of cell/digit indices
  std::vector<o2::emcal::TriggerRecord>* mClusterTriggerRecordsClusters = nullptr; ///< Array of Clusters trigger recirds
  std::vector<o2::emcal::TriggerRecord>* mClusterTriggerRecordsIndices = nullptr;  ///< Array of cell/digit indices trigger records
  ClassDefNV(ClusterizerTask, 1)
};
} // namespace emcal
} // namespace o2

#endif /* ALICEO2_EMCAL_CLUSTERIZERTASK */

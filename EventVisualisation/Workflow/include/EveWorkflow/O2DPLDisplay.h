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

///
/// \file   EveWorkflowHelper.h
/// \author julian.myrcha@cern.ch

#ifndef ALICE_O2_EVENTVISUALISATION_WORKFLOW_O2DPLDISPLAY_H
#define ALICE_O2_EVENTVISUALISATION_WORKFLOW_O2DPLDISPLAY_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "EveWorkflow/DetectorData.h"
#include "Framework/Task.h"
#include <memory>

using GID = o2::dataformats::GlobalTrackID;

namespace o2::trd
{
class GeometryFlat;
}

namespace o2::globaltracking
{
struct DataRequest;
}

namespace o2::itsmft
{
class TopologyDictionary;
}

namespace o2::event_visualisation
{
class TPCFastTransform;

class O2DPLDisplaySpec : public o2::framework::Task
{
 public:
  static constexpr float mWorkflowVersion = 1.02; // helps recognizing version of workflow which produce data
  O2DPLDisplaySpec(bool useMC, o2::dataformats::GlobalTrackID::mask_t trkMask,
                   o2::dataformats::GlobalTrackID::mask_t clMask,
                   std::shared_ptr<o2::globaltracking::DataRequest> dataRequest, const std::string& jsonPath,
                   std::chrono::milliseconds timeInterval, int numberOfFiles, int numberOfTracks, bool eveHostNameMatch, int minITSTracks, int minTracks, bool filterITSROF, bool filterTime, const EveWorkflowHelper::TBracket& timeBracket)
    : mUseMC(useMC), mTrkMask(trkMask), mClMask(clMask), mDataRequest(dataRequest), mJsonPath(jsonPath), mTimeInterval(timeInterval), mNumberOfFiles(numberOfFiles), mNumberOfTracks(numberOfTracks), mEveHostNameMatch(eveHostNameMatch), mMinITSTracks(minITSTracks), mMinTracks(minTracks), mFilterITSROF(filterITSROF), mFilterTime(filterTime), mTimeBracket(timeBracket)
  {
    this->mTimeStamp = std::chrono::high_resolution_clock::now() - timeInterval; // first run meets condition
  }
  ~O2DPLDisplaySpec() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;
  void endOfStream(o2::framework::EndOfStreamContext& ec) final;
  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(o2::framework::ProcessingContext& pc);

  bool mUseMC = false;
  bool mEveHostNameMatch;                   // empty or correct hostname
  int mMinITSTracks;                        // minimum number of ITS tracks to produce a file
  int mMinTracks;                           // minimum number of all tracks to produce a file
  bool mNoEmptyOutput;                      // don't create files with no tracks/clusters
  bool mFilterITSROF;                       // don't display tracks outside ITS readout frame
  bool mFilterTime;                         // don't display tracks outside [min, max] range in TF time
  EveWorkflowHelper::TBracket mTimeBracket; // [min, max] range in TF time for the filter
  std::string mJsonPath;                    // folder where files are stored
  std::chrono::milliseconds mTimeInterval;  // minimal interval between files in milliseconds
  int mNumberOfFiles;                       // maximum number of files in folder - newer replaces older
  int mNumberOfTracks;                      // maximum number of track in single file (0 means no limit)
  std::chrono::time_point<std::chrono::high_resolution_clock> mTimeStamp;

  o2::dataformats::GlobalTrackID::mask_t mTrkMask;
  o2::dataformats::GlobalTrackID::mask_t mClMask;
  DetectorData mData;
  std::shared_ptr<o2::globaltracking::DataRequest> mDataRequest;
};

} // namespace o2::event_visualisation

#endif

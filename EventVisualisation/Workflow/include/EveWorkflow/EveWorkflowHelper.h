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

#ifndef ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVEWORKFLOWHELPER_H
#define ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVEWORKFLOWHELPER_H

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "EveWorkflow/EveConfiguration.h"
#include "EventVisualisationDataConverter/VisualisationEvent.h"

namespace o2::event_visualisation
{
using GID = o2::dataformats::GlobalTrackID;
using PNT = std::array<float, 3>;

struct TracksSet {
  std::vector<GID> trackGID;
  std::vector<float> trackTime;
};

class EveWorkflowHelper
{
  static constexpr std::array<std::pair<float, float>, GID::NSources> minmaxR{{
    {1., 40.},   // ITS
    {85., 240.}, // TPC
    {-1, -1},    // TRD
    {-1, -1},    // TOF
    {-1, -1},    // PHS
    {-1, -1},    // CPV
    {-1, -1},    // EMC
    {-1, -1},    // HMP
    {-1, -1},    // MFT
    {-1, -1},    // MCH
    {-1, -1},    // MID
    {-1, -1},    // ZDC
    {-1, -1},    // FT0
    {-1, -1},    // VF0
    {-1, -1},    // FDD
    {1., 240},   // ITSTPC
    {85., 430.}, // TPCTOF
    {85., 380.}, // TPCTRD
    {1., 380.},  // ITSTPCTRD
    {-1, -1},    // ITSTPCTOF,
    {-1, -1},    // TPCTRDTOF,
    {-1, -1},    // ITSTPCTRDTOF, // full barrel track
    {-1, -1},    // ITSAB,
  }};

 public:
  static std::vector<PNT> getTrackPoints(const o2::track::TrackPar& trc, float minR, float maxR, float maxStep);
  void selectTracks(const CalibObjectsConst* calib, GID::mask_t maskCl,
                    GID::mask_t maskTrk, GID::mask_t maskMatch);
  template <typename Functor>
  void addTrackToEvent(Functor source, GID gid, float trackTime, float z = 0.); // store track in mEvent
  void draw(std::string jsonPath, int numberOfFiles, int numberOfTracks = -1);
  void drawTPC(GID gid, float trackTime);
  void drawITS(GID gid, float trackTime);
  void drawITSTPC(GID gid, float trackTime);
  void drawITSTPCTOF(GID gid, float trackTime);
  void drawITSClusters(GID gid, float trackTime);
  void drawTPCClusters(GID gid, float trackTime);
  void drawPoint(o2::BaseCluster<float> pnt);
  void prepareITSClusters(std::string dictfile = "");
  o2::globaltracking::RecoContainer mRecoCont;
  o2::globaltracking::RecoContainer& getRecoContainer() { return mRecoCont; }
  TracksSet mTrackSet;
  o2::event_visualisation::VisualisationEvent mEvent;
  std::vector<o2::BaseCluster<float>> mITSClustersArray;
};
} // namespace o2::event_visualisation

#endif //ALICE_O2_EVENTVISUALISATION_WORKFLOW_EVEWORKFLOWHELPER_H

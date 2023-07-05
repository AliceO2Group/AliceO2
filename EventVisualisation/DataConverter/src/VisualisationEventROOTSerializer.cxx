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
/// \file   VisualisationEventROOTSerializer.cxx
/// \brief  ROOT serialization
/// \author julian.myrcha@cern.ch

#include "EventVisualisationDataConverter/VisualisationEventROOTSerializer.h"
#include <fairlogger/Logger.h>
#include <iostream>

#include <TFile.h>
#include <TTree.h>
#include <TKey.h>
#include <TParameter.h>
#include <TNtuple.h>

namespace o2
{
namespace event_visualisation
{
constexpr int ROOT_FILE_VERSION = 1;

void VisualisationEventROOTSerializer::save(const char* name, const std::string& value)
{
  TNamed obj(name, value);
  obj.Write();
}

std::string VisualisationEventROOTSerializer::readString(TFile& f, const char* name)
{
  TNamed* v = (TNamed*)f.Get(name);
  if (v == nullptr) {
    return "";
  }
  std::string result = v->GetTitle();
  free(v);
  return result;
}

void VisualisationEventROOTSerializer::save(const char* name, int value)
{
  TParameter<int> obj;
  obj.SetVal(value);
  obj.Write(name);
}

int VisualisationEventROOTSerializer::readInt(TFile& f, const char* name)
{
  TParameter<int>* v = (TParameter<int>*)f.Get(name);
  if (v == nullptr) {
    return 0;
  }
  int result = v->GetVal();
  free(v);
  return result;
}

void VisualisationEventROOTSerializer::toFile(const VisualisationEvent& event, std::string fileName)
{
  TFile f(fileName.c_str(), "recreate");

  save("runNumber", event.mRunNumber);
  save("runType", event.mRunType);
  save("clMask", event.mClMask);
  save("trkMask", event.mTrkMask);
  save("tfCounter", event.mTfCounter);
  save("firstTForbit", event.mFirstTForbit);
  save("primaryVertex", event.mPrimaryVertex);
  save("collisionTime", event.mCollisionTime);
  save("eveVersion", event.mEveVersion);
  save("workflowParameters", event.mWorkflowParameters);

  // clusters
  TNtuple xyz("xyz", "xyz", "x:y:z");
  long xyzPos = 0L;
  TTree clusters("clusters", "Clusters");
  long cluster_xyz;
  int cluster_source;
  float cluster_time;
  clusters.Branch("source", &cluster_source);
  clusters.Branch("time", &cluster_time);
  clusters.Branch("xyz", &cluster_xyz);
  for (auto cluster : event.getClustersSpan()) {
    cluster_source = cluster.getSource();
    cluster_time = cluster.Time();
    cluster_xyz = xyzPos;
    xyz.Fill(cluster.X(), cluster.Y(), cluster.Z());
    xyzPos++;
    clusters.Fill();
  }

  // Tracks
  long track_xyz;     // first track point
  int track_points;   // number of track points
  int track_clusters; // number of track clusters
  int track_source;
  std::string track_GID;
  float track_time;
  int track_charge;
  float track_theta;
  float track_phi;
  float track_eta;
  int track_PID;

  TTree tracks("tracks", "Tracks");
  tracks.Branch("xyz", &track_xyz);
  tracks.Branch("time", &track_time);
  tracks.Branch("charge", &track_charge);
  tracks.Branch("theta", &track_theta);
  tracks.Branch("phi", &track_phi);
  tracks.Branch("eta", &track_eta);
  tracks.Branch("PID", &track_PID);
  tracks.Branch("GID", &track_GID);
  tracks.Branch("source", &track_source);
  tracks.Branch("points", &track_points);
  tracks.Branch("clusters", &track_clusters);

  for (auto track : event.getTracksSpan()) {
    track_xyz = xyzPos;
    track_time = std::isnan(track.mTime) ? 0 : track.mTime;
    track_charge = track.mCharge;
    track_theta = std::isnan(track.mTheta) ? 0 : track.mTheta;
    track_phi = std::isnan(track.mPhi) ? 0 : track.mPhi;
    track_eta = std::isnan(track.mEta) ? 0 : track.mEta;
    track_PID = track.mPID;
    track_GID = track.mGID;
    track_source = track.mSource;

    xyz.Fill(track.mStartCoordinates[0], track.mStartCoordinates[1], track.mStartCoordinates[2]);
    xyzPos++;
    track_points = track.getPointCount();

    for (size_t i = 0; i < track.getPointCount(); i++) {
      xyz.Fill(track.mPolyX[i], track.mPolyY[i], track.mPolyZ[i]);
      xyzPos++;
    }
    track_clusters = track.getClusterCount();
    for (auto cluster : track.getClustersSpan()) {
      xyz.Fill(cluster.X(), cluster.Y(), cluster.Z());
      xyzPos++;
    }
    tracks.Fill();
  }

  // calorimeters
  TTree calo("calo", "Calorimeters");
  int calo_source;
  float calo_time;
  float calo_energy;
  float calo_eta;
  float calo_phi;
  std::string calo_GID;
  int calo_PID;

  calo.Branch("source", &calo_source);
  calo.Branch("time", &calo_time);
  calo.Branch("energy", &calo_energy);
  calo.Branch("eta", &calo_eta);
  calo.Branch("phi", &calo_phi);
  calo.Branch("GID", &calo_GID);
  calo.Branch("PID", &calo_PID);

  for (auto calorimeter : event.getCalorimetersSpan()) {
    calo_source = calorimeter.getSource();
    calo_time = calorimeter.getTime();
    calo_energy = calorimeter.getEnergy();
    calo_eta = calorimeter.getEta();
    calo_phi = calorimeter.getPhi();
    calo_GID = calorimeter.getGIDAsString();
    calo_PID = calorimeter.getPID();
    calo.Fill();
  }
  tracks.Write();
  clusters.Write();
  calo.Write();
  xyz.Write();
}

bool VisualisationEventROOTSerializer::fromFile(VisualisationEvent& event, std::string fileName)
{
  LOG(info) << "VisualisationEventROOTSerializer <- " << fileName;
  event.mTracks.clear();
  event.mClusters.clear();
  event.mCalo.clear();

  TFile f(fileName.c_str());

  event.setRunNumber(readInt(f, "runNumber"));
  event.setRunType(static_cast<parameters::GRPECS::RunType>(readInt(f, "runType")));
  event.setClMask(readInt(f, "clMask"));
  event.setTrkMask(readInt(f, "trkMask"));
  event.setTfCounter(readInt(f, "tfCounter"));
  event.setFirstTForbit(readInt(f, "firstTForbit"));
  event.mPrimaryVertex = readInt(f, "primaryVertex");

  event.setCollisionTime(readString(f, "collisionTime"));
  event.mEveVersion = readString(f, "eveVersion");
  event.mWorkflowParameters = readString(f, "workflowParameters");

  // xyz
  TNtuple* xyz = (TNtuple*)f.Get("xyz");
  if (xyz == nullptr) {
    return false;
  }

  // tracks
  TTree* tracks = (TTree*)f.Get("tracks");
  if (tracks == nullptr) {
    delete xyz;
    return false;
  }

  long track_xyz;     // first track point
  int track_points;   // number of track points
  int track_clusters; // number of track clusters
  int track_source;
  std::string* track_GID = nullptr;
  float track_time;
  int track_charge;
  float track_theta;
  float track_phi;
  float track_eta;
  int track_PID;

  tracks->SetBranchAddress("xyz", &track_xyz);
  tracks->SetBranchAddress("time", &track_time);
  tracks->SetBranchAddress("charge", &track_charge);
  tracks->SetBranchAddress("theta", &track_theta);
  tracks->SetBranchAddress("phi", &track_phi);
  tracks->SetBranchAddress("eta", &track_eta);
  tracks->SetBranchAddress("PID", &track_PID);
  tracks->SetBranchAddress("GID", &track_GID);
  tracks->SetBranchAddress("source", &track_source);
  tracks->SetBranchAddress("points", &track_points);
  tracks->SetBranchAddress("clusters", &track_clusters);

  Int_t tracksNoOfEntries = (Int_t)tracks->GetEntries();
  for (Int_t i = 0; i < tracksNoOfEntries; i++) {
    tracks->GetEntry(i);
    VisualisationTrack track;
    track.mTime = track_time;
    track.mCharge = track_charge;
    track.mTheta = track_theta;
    track.mPhi = track_phi;
    track.mEta = track_eta;
    track.mPID = track_PID;
    track.mGID = *track_GID;
    track.mSource = (o2::dataformats::GlobalTrackID::Source)track_source;
    xyz->GetEntry(track_xyz);
    track.addStartCoordinates(xyz->GetArgs());
    for (size_t i = 0; i < track_points; i++) {
      xyz->GetEntry(track_xyz + 1 + i);
      track.addPolyPoint(xyz->GetArgs());
    }
    for (size_t i = 0; i < track_clusters; i++) {
      xyz->GetEntry(track_xyz + 1 + track_points + i);
      VisualisationCluster cluster(xyz->GetArgs(), track.mTime);
      cluster.mSource = track.mSource;
      track.mClusters.emplace_back(cluster);
    }
    event.mTracks.emplace_back(track);
  }
  if (track_GID != nullptr) {
    delete track_GID;
    track_GID = nullptr;
  }

  TTree* clusters = (TTree*)f.Get("clusters");
  if (clusters == nullptr) {
    delete xyz;
    delete tracks;
    return false;
  }

  long cluster_xyz;
  int cluster_source;
  float cluster_time;
  clusters->SetBranchAddress("source", &cluster_source);
  clusters->SetBranchAddress("time", &cluster_time);
  clusters->SetBranchAddress("xyz", &cluster_xyz);
  Int_t clustersNoOfEntries = (Int_t)clusters->GetEntries();
  for (Int_t i = 0; i < clustersNoOfEntries; i++) {
    clusters->GetEntry(i);
    xyz->GetEntry(cluster_xyz);
    VisualisationCluster cluster(xyz->GetArgs(), cluster_time);
    cluster.mSource = (o2::dataformats::GlobalTrackID::Source)cluster_source;
    event.mClusters.emplace_back(cluster);
  }

  // calorimeters
  TTree* calo = (TTree*)f.Get("calo");
  if (calo == nullptr) {
    delete xyz;
    delete tracks;
    delete clusters;
    return false;
  }
  int calo_source;
  float calo_time;
  float calo_energy;
  float calo_eta;
  float calo_phi;
  std::string* calo_GID = nullptr;
  int calo_PID;

  calo->SetBranchAddress("source", &calo_source);
  calo->SetBranchAddress("time", &calo_time);
  calo->SetBranchAddress("energy", &calo_energy);
  calo->SetBranchAddress("eta", &calo_eta);
  calo->SetBranchAddress("phi", &calo_phi);
  calo->SetBranchAddress("GID", &calo_GID);
  calo->SetBranchAddress("PID", &calo_PID);

  Int_t nentries = (Int_t)calo->GetEntries();
  for (Int_t i = 0; i < nentries; i++) {
    calo->GetEntry(i);
    VisualisationCalo calorimeter;
    calorimeter.mSource = (o2::dataformats::GlobalTrackID::Source)calo_source;
    calorimeter.mTime = calo_time;
    calorimeter.mEnergy = calo_energy;
    calorimeter.mEta = calo_eta;
    calorimeter.mPhi = calo_phi;
    if (calo_GID) {
      calorimeter.mGID = *calo_GID;
    }
    calorimeter.mPID = calo_PID;
    event.mCalo.emplace_back(calorimeter);
  }
  if (calo_GID != nullptr) {
    delete calo_GID;
    calo_GID = nullptr;
  }
  delete calo;
  delete tracks;
  delete xyz;
  delete clusters;
  event.afterLoading();
  return true;
}

} // namespace event_visualisation
} // namespace o2

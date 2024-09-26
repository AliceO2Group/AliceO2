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
  auto* v = (TNamed*)f.Get(name);
  if (v == nullptr) {
    return "";
  }
  std::string result = v->GetTitle();
  free(v);
  return result;
}

void VisualisationEventROOTSerializer::saveInt(const char* name, int value)
{
  TParameter<int> obj;
  obj.SetVal(value);
  obj.Write(name);
}

void VisualisationEventROOTSerializer::saveUInt64(const char* name, uint64_t value)
{
  TParameter<long> obj;
  obj.SetVal((long)value);
  obj.Write(name);
}

int VisualisationEventROOTSerializer::readInt(TFile& f, const char* name)
{
  auto v = (TParameter<int>*)f.Get(name);
  if (v == nullptr) {
    return 0;
  }
  int result = v->GetVal();
  free(v);
  return result;
}

uint64_t VisualisationEventROOTSerializer::readUInt64(TFile& f, const char* name)
{
  auto v = (TParameter<uint64_t>*)f.Get(name);
  if (v == nullptr) {
    return 0;
  }
  uint64_t result = v->GetVal();
  free(v);
  return result;
}

bool VisualisationEventROOTSerializer::existUInt64(TFile& f, const char* name)
{
  auto v = (TParameter<uint64_t>*)f.Get(name);
  if (v == nullptr) {
    return false;
  }
  free(v);
  return true;
}

void VisualisationEventROOTSerializer::toFile(const VisualisationEvent& event, std::string fileName)
{
  TFile f(fileName.c_str(), "recreate");

  saveInt("runNumber", event.mRunNumber);
  saveInt("runType", event.mRunType);
  saveInt("clMask", event.mClMask);
  saveInt("trkMask", event.mTrkMask);
  saveInt("tfCounter", event.mTfCounter);
  saveInt("firstTForbit", event.mFirstTForbit);
  saveInt("primaryVertex", event.mPrimaryVertex);
  saveUInt64("creationTime", event.mCreationTime);
  std::string version = std::to_string(event.mEveVersion / 100.0);
  save("eveVersion", version); // obsolete
  saveInt("version", event.mEveVersion);
  // save("workflowParameters", event.mWorkflowParameters);

  // clusters
  TNtuple xyz("xyz", "xyz", "x:y:z");
  long xyzPos = 0L;

  TTree clusters("clusters", "Clusters");
  long cluster_xyz;
  unsigned cluster_bgid;
  float cluster_time;
  clusters.Branch("BGID", &cluster_bgid);
  clusters.Branch("time", &cluster_time);
  clusters.Branch("xyz", &cluster_xyz);
  for (auto cluster : event.getClustersSpan()) {
    cluster_time = cluster.Time();
    cluster_bgid = serialize(cluster.mBGID);
    cluster_xyz = xyzPos;
    xyz.Fill(cluster.X(), cluster.Y(), cluster.Z());
    xyzPos++;
    clusters.Fill();
  }

  // Tracks
  long track_xyz;     // first track point
  int track_points;   // number of track points
  int track_clusters; // number of track clusters
  unsigned track_BGID; // binary GID
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
  tracks.Branch("BGID", &track_BGID);
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
    track_BGID = serialize(track.mBGID);

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

  clusters.Write();

  // calorimeters
  TTree calo("calo", "Calorimeters");
  float calo_time;
  float calo_energy;
  float calo_eta;
  float calo_phi;
  int calo_PID;
  unsigned calo_BGID;

  calo.Branch("time", &calo_time);
  calo.Branch("energy", &calo_energy);
  calo.Branch("eta", &calo_eta);
  calo.Branch("phi", &calo_phi);
  calo.Branch("BGID", &calo_BGID);
  calo.Branch("PID", &calo_PID);

  for (const auto& calorimeter : event.getCalorimetersSpan()) {
    calo_time = calorimeter.getTime();
    calo_energy = calorimeter.getEnergy();
    calo_eta = calorimeter.getEta();
    calo_phi = calorimeter.getPhi();
    calo_BGID = serialize(calorimeter.mBGID);
    calo_PID = calorimeter.getPID();
    calo.Fill();
  }
  calo.Write();

  tracks.Write();

  xyz.Write();
}

bool VisualisationEventROOTSerializer::fromFile(VisualisationEvent& event, std::string fileName)
{
  // LOGF(info, "VisualisationEventROOTSerializer <- %s ", fileName);
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

  if (existUInt64(f, "creationTime")) {
    event.setCreationTime(readInt(f, "creationTime"));
  } else {
    auto collisionTime = readString(f, "collisionTime");
    event.mCreationTime = parseDateTime(collisionTime.c_str());
  }

  event.mEveVersion = 0;
  if (f.Get("version") != nullptr) {
    event.mEveVersion = readInt(f, "version");
  } else {
    std::string version = readString(f, "eveVersion");
    event.mEveVersion = (int)(100 * std::stof(version));
  }

  // xyz
  auto* xyz = (TNtuple*)f.Get("xyz");
  if (xyz == nullptr) {
    return false;
  }

  // tracks
  auto* tracks = (TTree*)f.Get("tracks");
  if (tracks == nullptr) {
    delete xyz;
    return false;
  }

  long track_xyz;     // first track point
  int track_points;   // number of track points
  int track_clusters; // number of track clusters
  int track_source;   // obsolete
  std::string* track_GID = nullptr;
  unsigned track_BGID;
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
  auto gid = tracks->GetBranch("GID"); // obsolete
  if (gid != nullptr) {
    tracks->SetBranchAddress("GID", &track_GID);
  }
  auto bgid = tracks->GetBranch("BGID");
  if (bgid != nullptr) {
    tracks->SetBranchAddress("BGID", &track_BGID);
  }
  auto source = tracks->GetBranch("source"); // obsolete
  if (source != nullptr) {
    tracks->SetBranchAddress("source", &track_source);
  }
  tracks->SetBranchAddress("points", &track_points);
  tracks->SetBranchAddress("clusters", &track_clusters);

  auto tracksNoOfEntries = (Int_t)tracks->GetEntries();
  for (Int_t i = 0; i < tracksNoOfEntries; i++) {
    tracks->GetEntry(i);
    VisualisationTrack track;
    track.mTime = track_time;
    track.mCharge = track_charge;
    track.mTheta = track_theta;
    track.mPhi = track_phi;
    track.mEta = track_eta;
    track.mPID = track_PID;
    if (bgid) {
      track.mBGID = deserialize(track_BGID);
    } else {
      track.mBGID = gidFromString(*track_GID);
    }
    xyz->GetEntry(track_xyz);
    track.addStartCoordinates(xyz->GetArgs());
    for (int p = 0; p < track_points; p++) {
      xyz->GetEntry(track_xyz + 1 + p);
      track.addPolyPoint(xyz->GetArgs());
    }
    for (size_t it = 0; it < track_clusters; it++) {
      xyz->GetEntry(track_xyz + 1 + track_points + it);
      VisualisationCluster cluster(xyz->GetArgs(), track.mTime, track.mBGID);
      track.mClusters.emplace_back(cluster);
    }
    event.mTracks.emplace_back(track);
  }

  if (gid != nullptr) {
    delete track_GID;
    track_GID = nullptr;
  }

  if (!readClusters(event, f, xyz)) {
    delete xyz;
    delete tracks;
    return false;
  }

  if (!readCalo(event, f)) {
    delete xyz;
    delete tracks;
    return false;
  }

  delete tracks;
  delete xyz;

  event.afterLoading();
  return true;
}

bool VisualisationEventROOTSerializer::readClusters(VisualisationEvent& event, TFile& f, TNtuple* xyz)
{
  auto* clusters = (TTree*)f.Get("clusters");
  if (clusters == nullptr) {
    return false;
  }

  long cluster_xyz;
  int cluster_source;
  unsigned cluster_BGID;

  float cluster_time;
  auto source = clusters->GetBranch("source"); // obsolete
  if (source != nullptr) {
    clusters->SetBranchAddress("source", &cluster_source);
  }

  auto bgid = clusters->GetBranch("BGID");
  if (bgid != nullptr) {
    clusters->SetBranchAddress("BGID", &cluster_BGID);
  }

  clusters->SetBranchAddress("time", &cluster_time);
  clusters->SetBranchAddress("xyz", &cluster_xyz);
  auto clustersNoOfEntries = (Int_t)clusters->GetEntries();
  for (Int_t i = 0; i < clustersNoOfEntries; i++) {
    clusters->GetEntry(i);
    xyz->GetEntry(cluster_xyz);
    dataformats::GlobalTrackID gid = 0;
    if (bgid) {
      gid = deserialize(cluster_BGID);
    } else if (source) {
      gid = deserialize(cluster_source, 0, 0);
    }
    VisualisationCluster cluster(xyz->GetArgs(), cluster_time, gid);
    event.mClusters.emplace_back(cluster);
  }
  delete clusters;
  return true;
}

bool VisualisationEventROOTSerializer::readCalo(VisualisationEvent& event, TFile& f)
{
  auto* calo = (TTree*)f.Get("calo");
  if (calo == nullptr) {
    return false;
  }

  int calo_source;
  float calo_time;
  float calo_energy;
  float calo_eta;
  float calo_phi;
  unsigned calo_BGID;
  int calo_PID;

  auto source = calo->GetBranch("source");
  if (source != nullptr) {
    calo->SetBranchAddress("source", &calo_source);
  }
  auto bgid = calo->GetBranch("BGID");
  if (bgid != nullptr) {
    calo->SetBranchAddress("BGID", &calo_BGID);
  }

  calo->SetBranchAddress("time", &calo_time);
  calo->SetBranchAddress("energy", &calo_energy);
  calo->SetBranchAddress("eta", &calo_eta);
  calo->SetBranchAddress("phi", &calo_phi);
  calo->SetBranchAddress("PID", &calo_PID);

  auto nentries = (Int_t)calo->GetEntries();
  for (Int_t i = 0; i < nentries; i++) {
    calo->GetEntry(i);
    VisualisationCalo calorimeter;
    calorimeter.mTime = calo_time;
    calorimeter.mEnergy = calo_energy;
    calorimeter.mEta = calo_eta;
    calorimeter.mPhi = calo_phi;
    if (bgid) {
      calorimeter.mBGID = deserialize(calo_BGID);
    } else {
      calorimeter.mBGID = deserialize(calo_source, 0, 0);
    }
    calorimeter.mPID = calo_PID;
    event.mCalo.emplace_back(calorimeter);
  }
  delete calo;
  return true;
}

} // namespace event_visualisation
} // namespace o2

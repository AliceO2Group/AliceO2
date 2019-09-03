// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Tracker.cxx
/// \brief Implementation of the track finding from MFT clusters
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 8, 2018

#include "MFTReconstruction/Tracker.h"
#include "DataFormatsITSMFT/Cluster.h"

#include <future>
#include <chrono>

using namespace o2::mft;
using namespace o2::itsmft;

Tracker::Layer Tracker::sLayers[constants::LayersNumber];

//_____________________________________________________________________________
Tracker::Tracker(Int_t n) : mNumOfThreads(n) {}

//_____________________________________________________________________________
void Tracker::process(const std::vector<Cluster>& clusters, std::vector<TrackMFT>& tracks)
{

  static int entry = 0;

  LOG(INFO);
  LOG(INFO) << "Tracker::process() entry " << entry++ << ", number of threads: " << mNumOfThreads;

  Int_t nClFrame = 0;
  Int_t numOfClustersLeft = clusters.size(); // total number of clusters
  if (numOfClustersLeft == 0) {
    LOG(WARNING) << "No clusters to process !";
    return;
  }

  auto start = std::chrono::system_clock::now();

  while (numOfClustersLeft > 0) {

    nClFrame = loadClusters(clusters);
    if (!nClFrame) {
      LOG(FATAL) << "Failed to select any cluster out of " << numOfClustersLeft << " check if cont/trig mode is correct";
    }
    numOfClustersLeft -= nClFrame;
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end - start;
    LOG(INFO) << "Loading clusters: " << nClFrame << " in single frame " << mROFrame << " : " << diff.count() << " s";

    start = end;

    processFrame(tracks);

    unloadClusters();
    end = std::chrono::system_clock::now();
    diff = end - start;
    LOG(INFO) << "Processing time for single frame " << mROFrame << " : " << diff.count() << " s";

    start = end;
    if (mContinuousMode) {
      mROFrame++; // expect incremented frame in following clusters
    }
  }
}

//_____________________________________________________________________________
void Tracker::processFrame(std::vector<TrackMFT>& tracks)
{

  LOG(INFO) << "Tracker::process(), number of threads: " << mNumOfThreads;

  std::vector<std::future<std::vector<TrackMFT>>> futures(mNumOfThreads);
  Int_t numOfClusters = sLayers[0].getNumberOfClusters();
  for (Int_t t = 0, first = 0; t < mNumOfThreads; t++) {
    Int_t rem = t < (numOfClusters % mNumOfThreads) ? 1 : 0;
    Int_t last = first + (numOfClusters / mNumOfThreads) + rem;
    futures[t] = std::async(std::launch::async, &Tracker::trackInThread, this, first, last);
    first = last;
  }

  std::vector<std::vector<TrackMFT>> trackArray(mNumOfThreads);
  for (Int_t t = 0; t < mNumOfThreads; t++) {
    trackArray[t] = futures[t].get();
  }
}

//_____________________________________________________________________________
std::vector<TrackMFT> Tracker::trackInThread(Int_t first, Int_t last)
{

  std::vector<TrackMFT> tracks;

  Layer& layer1 = sLayers[0];
  Layer& layer2 = sLayers[1];
  Int_t nClusters1 = layer1.getNumberOfClusters();
  Int_t nClusters2 = layer2.getNumberOfClusters();
  LOG(INFO) << "trackInThread first: " << first << " last: " << last;
  LOG(INFO) << "nCusters 1: " << nClusters1 << " 2: " << nClusters2;
  // std::this_thread::sleep_for(std::chrono::seconds(10));

  return tracks;
}

//_____________________________________________________________________________
void Tracker::setGeometry(o2::mft::GeometryTGeo* geom)
{

  /// attach geometry interface
  mGeom = geom;
  for (Int_t i = 0; i < constants::LayersNumber; i++) {
    sLayers[i].setGeometry(geom);
  }
}

//_____________________________________________________________________________
Int_t Tracker::loadClusters(const std::vector<Cluster>& clusters)
{

  Int_t numOfClusters = clusters.size();
  int nLoaded = 0;

  if (mContinuousMode) { // check the ROFrame in cont. mode
    for (auto& c : clusters) {
      if (c.getROFrame() != mROFrame) {
        continue;
      }
      nLoaded++;
      Int_t layer = mGeom->getLayer(c.getSensorID());
      if (!sLayers[layer].insertCluster(&c)) {
        continue;
      }
    }
  } else { // do not check the ROFrame in triggered mode
    for (auto& c : clusters) {
      nLoaded++;
      Int_t layer = mGeom->getLayer((Int_t)(c.getSensorID()));
      if (!sLayers[layer].insertCluster(&c)) {
        continue;
      }
    }
  }

  if (nLoaded) {
    std::vector<std::future<void>> fut;
    for (Int_t l = 0; l < constants::LayersNumber; l += mNumOfThreads) {
      for (Int_t t = 0; t < mNumOfThreads; t++) {
        if ((l + t) >= constants::LayersNumber)
          break;
        auto f = std::async(std::launch::async, &Tracker::Layer::init, sLayers + (l + t));
        fut.push_back(std::move(f));
      }
      for (Int_t t = 0; t < fut.size(); t++)
        fut[t].wait();
    }
  }

  return nLoaded;
}

//_____________________________________________________________________________
void Tracker::unloadClusters()
{

  for (Int_t i = 0; i < constants::LayersNumber; i++) {
    sLayers[i].unloadClusters();
  }
}

//_____________________________________________________________________________
Tracker::Layer::Layer() = default;

//_____________________________________________________________________________
void Tracker::Layer::init() {}

//_____________________________________________________________________________
void Tracker::Layer::unloadClusters() { mClusters.clear(); }

//_____________________________________________________________________________
Bool_t Tracker::Layer::insertCluster(const Cluster* c)
{

  mClusters.push_back(c);

  return kTRUE;
}

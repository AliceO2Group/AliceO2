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
#include "ITSMFTReconstruction/Cluster.h"

#include <future>
#include <chrono>

using namespace o2::MFT;
using namespace o2::ITSMFT;

Tracker::Layer Tracker::sLayers[Constants::sNLayers];

//_____________________________________________________________________________
Tracker::Tracker(Int_t n) : mNumOfThreads(n)
{

}

//_____________________________________________________________________________
void Tracker::process(const std::vector<Cluster> &clusters, std::vector<Track> &tracks)
{

  static int entry = 0;
  
  LOG(INFO) << FairLogger::endl;
  LOG(INFO) << "Tracker::process() entry " << entry++
            << ", number of threads: " << mNumOfThreads << FairLogger::endl;
  
  Int_t nClFrame = 0;
  Int_t numOfClustersLeft = clusters.size(); // total number of clusters
  if (numOfClustersLeft == 0) {
    LOG(WARNING) << "No clusters to process !" << FairLogger::endl;
    return;
  }

  auto start = std::chrono::system_clock::now();
  
  while (numOfClustersLeft > 0) {

    nClFrame = loadClusters(clusters);
    if (!nClFrame) {
      LOG(FATAL) << "Failed to select any cluster out of " <<  numOfClustersLeft
                 << " check if cont/trig mode is correct" <<  FairLogger::endl;
    }
    numOfClustersLeft -= nClFrame;    
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end-start;   
    LOG(INFO) << "Loading clusters: " << nClFrame   << " in single frame " << mROFrame << " : " << diff.count() << " s" << FairLogger::endl;

    start = end;

    processFrame(tracks);

    unloadClusters();
    end = std::chrono::system_clock::now();
    diff = end-start;
    LOG(INFO)<<"Processing time for single frame " << mROFrame << " : " << diff.count() <<" s" << FairLogger::endl;
    
    start = end;
    if (mContinuousMode) mROFrame++; // expect incremented frame in following clusters
    
  }
  
  
}

//_____________________________________________________________________________
void Tracker::processFrame(std::vector<Track> &tracks)
{

  LOG(INFO) << "Tracker::process(), number of threads: " << mNumOfThreads << FairLogger::endl;

  std::vector<std::future<std::vector<Track>>> futures(mNumOfThreads);
  Int_t numOfClusters = sLayers[0].getNumberOfClusters();
  for (Int_t t = 0, first = 0; t < mNumOfThreads; t++) {
    Int_t rem = t < (numOfClusters % mNumOfThreads) ? 1 : 0;
    Int_t last = first + (numOfClusters / mNumOfThreads) + rem;
    futures[t] = std::async(std::launch::async, &Tracker::trackInThread, this, first, last);
    first = last;
  }

  std::vector<std::vector<Track>> trackArray(mNumOfThreads);
  for (Int_t t = 0; t < mNumOfThreads; t++) {
    trackArray[t] = futures[t].get();
  }
  
}

//_____________________________________________________________________________
std::vector<Track> Tracker::trackInThread(Int_t first, Int_t last)
{

  std::vector<Track> tracks;

  Layer& layer1 = sLayers[0];
  Layer& layer2 = sLayers[1];
  Int_t nClusters1 = layer1.getNumberOfClusters();
  Int_t nClusters2 = layer2.getNumberOfClusters();
  LOG(INFO) << "trackInThread first: " << first << " last: " << last << FairLogger::endl;
  LOG(INFO) << "nCusters 1: " << nClusters1 << " 2: " << nClusters2 << FairLogger::endl;
  for (Int_t i = 0; i < 1000000; i++) {
    for (Int_t j = 0; j < 1000000; j++) {
      LOG(INFO) << i << " " << j << FairLogger::endl;
      for (Int_t n1 = first; n1 < last; n1++) {
	const Cluster* c1 = layer1.getCluster(n1);
	for (Int_t n2 = 0; n2 < nClusters2; n2++) {
	  const Cluster* c2 = layer2.getCluster(n2);
	}
      }
    }
  }
  //std::this_thread::sleep_for(std::chrono::seconds(10));
  
  return tracks;
  
}
  
//_____________________________________________________________________________
void Tracker::setGeometry(o2::MFT::GeometryTGeo* geom)
{
  
  /// attach geometry interface
  mGeom = geom;
  for (Int_t i = 0; i < Constants::sNLayers; i++) {
    sLayers[i].setGeometry(geom);
  }
 
}

//_____________________________________________________________________________
Int_t Tracker::loadClusters(const std::vector<Cluster> &clusters)
{

  Int_t numOfClusters = clusters.size();
  int nLoaded = 0;

  if (mContinuousMode) { // check the ROFrame in cont. mode
    for (auto &c : clusters) {
      if (c.getROFrame() != mROFrame) {
        continue;
      }
      nLoaded++;
      Int_t layer = mGeom->getLayer(c.getSensorID());
      if (!sLayers[layer].insertCluster(&c)) {
        continue;
      }
    }
  } else {  // do not check the ROFrame in triggered mode
    for (auto &c : clusters) {
      nLoaded++;
      Int_t layer = mGeom->getLayer((Int_t)(c.getSensorID()));
      if (!sLayers[layer].insertCluster(&c)) {
        continue;
      }
    }    
  }
  
  if (nLoaded) {
    std::vector<std::future<void>> fut;
    for (Int_t l = 0; l < Constants::sNLayers; l += mNumOfThreads) {
      for (Int_t t = 0; t < mNumOfThreads; t++) {
        if ((l+t) >= Constants::sNLayers) break;
        auto f = std::async(std::launch::async, &Tracker::Layer::init, sLayers+(l+t));
        fut.push_back(std::move(f));
      }
      for (Int_t t = 0; t < fut.size(); t++) fut[t].wait();
    }
  }
  
  return nLoaded;

}

//_____________________________________________________________________________
void Tracker::unloadClusters()
{
  
  for (Int_t i = 0; i < Constants::sNLayers; i++) {
    sLayers[i].unloadClusters();
  }
  
}

//_____________________________________________________________________________
Tracker::Layer::Layer()
{
  
}

//_____________________________________________________________________________
void Tracker::Layer::init()
{

}

//_____________________________________________________________________________
void Tracker::Layer::unloadClusters()
{

  mClusters.clear();

}

//_____________________________________________________________________________
Bool_t Tracker::Layer::insertCluster(const Cluster* c)
{

  mClusters.push_back(c);

  return kTRUE;

}

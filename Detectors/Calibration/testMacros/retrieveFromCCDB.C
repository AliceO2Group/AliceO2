#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <string>
#include <chrono>
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <unistd.h>
#include "Framework/Logger.h"
#include "CCDB/BasicCCDBManager.h"
#include <pthread.h>
#include <thread>
#endif
#include "populateCCDB.C"

using CcdbManager = o2::ccdb::CCDBManagerInstance;

std::vector<std::unique_ptr<CcdbManager>> ccdbManPool;
std::vector<bool> ccdbManPoolFlag;
std::decay_t<decltype(std::chrono::high_resolution_clock::now())> startTime{};

// macro to populate the CCDB emulating the rates that we expect for
// Run 3, as read (in terms of size and rate) from an external file

int getFreeCCDBManagerID();
void retrieve(const std::vector<CCDBObj>& objs, float procTimeSec, int ccdbID, std::atomic<int>& n, size_t tfID);

void retrieveFromCCDB(int maxTFs = 8, float procTimeSec = 10.,
                      const std::string& fname = "cdbSizeV0.txt",
                      const std::string& ccdbHost = "http://localhost:8080" /*"http://ccdb-test.cern.ch:8080"*/,
                      bool allowCaching = true)
{
  ccdbManPool.resize(1 + maxTFs);
  ccdbManPoolFlag.resize(ccdbManPool.size(), false); // nothig is used at the moment
  LOG(INFO) << "Caching is " << (allowCaching ? "ON" : "OFF");
  for (auto& mgr : ccdbManPool) {
    mgr = std::make_unique<CcdbManager>(ccdbHost.c_str());
    mgr->setCaching(allowCaching);
  }
  auto objs = readObjectsList(fname);
  if (objs.empty()) {
    return;
  }
  startTime = std::chrono::high_resolution_clock::now();

  std::atomic<int> nTFs{0};
  size_t tfID = 0;
  while (1) {
    if (nTFs < maxTFs) {
      int ccdbID = getFreeCCDBManagerID();
      if (ccdbID < 0) { // all slots are busy, wait
        continue;
      }
      std::thread th(retrieve, std::cref(objs), procTimeSec, ccdbID, std::ref(nTFs), tfID++);
      th.detach();
      LOG(INFO) << nTFs << " TFs currently in processing";
    } else {
      usleep(long(procTimeSec * (0.01e6)));
    }
  }
}

void retrieve(const std::vector<CCDBObj>& objs, float procTimeSec, int ccdbID, std::atomic<int>& n, size_t tfID)
{
  // function to retrieve the CCDB objects and wait some (TF processing) time
  // to avoid all treads starting to read at the same time, we randomize the time of reading within the allocated processing time
  auto mgr = ccdbManPool[ccdbID].get();
  n++;
  float tfrac = gRandom->Rndm();
  usleep(long(procTimeSec * tfrac * 1e6));
  auto ret_start = std::chrono::high_resolution_clock::now();
  for (auto& o : objs) {
    auto now = std::chrono::high_resolution_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto timeStamp = now_ms.time_since_epoch();
    std::vector<uint8_t>* ob = mgr->getForTimeStamp<std::vector<uint8_t>>(o.path, timeStamp.count());
    LOG(DEBUG) << "Retrieved object " << o.path << " of size " << ob->size() << " Bytes"
               << " for TF " << tfID;
    if (!mgr->isCachingEnabled()) { // we can delete object only when caching is disabled
      delete ob;
    }
  }
  auto ret_end = std::chrono::high_resolution_clock::now();
  usleep(long(procTimeSec * (1. - tfrac) * 1e6));
  std::chrono::duration<double, std::ratio<1, 1>> elapsedSeconds = std::chrono::high_resolution_clock::now() - startTime;
  std::chrono::duration<double, std::ratio<1, 1>> retTime = ret_end - ret_start;
  LOG(INFO) << "Finished TF " << tfID << " elapsed time " << elapsedSeconds.count() << " s., CCDB query time: " << retTime.count() << " s.";
  n--;
  ccdbManPoolFlag[ccdbID] = false; // release the manager
}

int getFreeCCDBManagerID()
{
  /// get ID of 1st CCDBManager not used by any thread
  for (unsigned i = 0; i < ccdbManPoolFlag.size(); i++) {
    if (!ccdbManPoolFlag[i]) {
      ccdbManPoolFlag[i] = true;
      return i;
    }
  }
  return -1;
}

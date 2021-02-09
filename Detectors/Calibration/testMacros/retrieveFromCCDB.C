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
#include <mutex>
#endif
#include "populateCCDB.C"

using CcdbManager = o2::ccdb::BasicCCDBManager;

// macro to populate the CCDB emulating the rates that we expect for
// Run 3, as read (in terms of size and rate) from an external file

void retrieve(const std::vector<CCDBObj>& objs, float procTimeSec, std::mutex& mtx, std::atomic<int>& n, size_t tfID);

void retrieveFromCCDB(int maxTFs = 8, float procTimeSec = 10.,
                      const std::string& fname = "cdbSizeV0.txt",
                      const std::string& ccdbHost = "http://localhost:8080" /*"http://ccdb-test.cern.ch:8080"*/,
                      bool allowCaching = true)
{
  auto& mgr = CcdbManager::instance();
  mgr.setURL(ccdbHost.c_str()); // or http://localhost:8080 for a local installation
  mgr.setCaching(allowCaching);
  auto objs = readObjectsList(fname);
  if (objs.empty()) {
    return;
  }
  std::mutex ccdb_mtx;
  std::atomic<int> nTFs{0};
  size_t tfID = 0;
  while (1) {
    if (nTFs < maxTFs) {
      std::thread th(retrieve, std::cref(objs), procTimeSec, std::ref(ccdb_mtx), std::ref(nTFs), tfID++);
      th.detach();
      LOG(INFO) << nTFs << " TFs currently in processing";
    } else {
      usleep(long(procTimeSec * (0.01e6)));
    }
  }
}

void retrieve(const std::vector<CCDBObj>& objs, float procTimeSec, std::mutex& mtx, std::atomic<int>& n, size_t tfID)
{
  // function to retrieve the CCDB objects and wait some (TF processing) time
  // to avoid all treads starting to read at the same time, we randomize the time of reading within the allocated processing time

  n++;
  auto& mgr = CcdbManager::instance();
  float tfrac = gRandom->Rndm();
  usleep(long(procTimeSec * tfrac * 1e6));
  for (auto& o : objs) {
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto timeStamp = now_ms.time_since_epoch();
    mtx.lock();
    std::vector<uint8_t>* ob = mgr.getForTimeStamp<std::vector<uint8_t>>(o.path, timeStamp.count());
    mtx.unlock();
    LOG(INFO) << "Retrieved object " << o.path << " of size " << ob->size() << " Bytes"
              << " for TF " << tfID;
    if (!mgr.isCachingEnabled()) { // we can delete object only when caching is disabled
      delete ob;
    }
  }
  usleep(long(procTimeSec * (1. - tfrac) * 1e6));
  n--;
}

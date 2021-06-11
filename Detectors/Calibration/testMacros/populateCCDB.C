#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <string>
#include <chrono>
#include <iostream>
#include <string>
#include <fstream>
#include <regex>
#include <unistd.h>
#include <thread>
#include <TRandom.h>
#include <TVectorF.h>
#include "Framework/Logger.h"
#include "CCDB/CcdbApi.h"

#endif

// macro to populate the CCDB emulating the rates that we expect for
// Run 3, as read (in terms of size and rate) from an external file

using DurSec = std::chrono::duration<double, std::ratio<1, 1>>;

struct CCDBObj {
  std::string path;
  float validity;
  size_t sz;
  size_t count = 0;
  std::decay_t<decltype(std::chrono::high_resolution_clock::now())> lastUpdate{};
  CCDBObj(const std::string& _path, size_t _sz, float _val) : path(_path), validity(_val), sz(_sz) {}
};

std::vector<CCDBObj> readObjectsList(const std::string& fname);
void pushObject(o2::ccdb::CcdbApi& api, const CCDBObj& obj);

void populateCCDB(const std::string& fname = "cdbSizeV0.txt", const std::string& ccdbHost = "http://localhost:8080" /*"http://ccdb-test.cern.ch:8080"*/)
{
  auto objs = readObjectsList(fname);
  if (objs.empty()) {
    return;
  }

  o2::ccdb::CcdbApi api;
  api.init(ccdbHost.c_str()); // or http://localhost:8080 for a local installation

  while (true) {
    auto timeLoopStart = std::chrono::high_resolution_clock::now();
    double minTLeft = 1e99;
    for (auto& o : objs) {
      DurSec elapsedSeconds = timeLoopStart - o.lastUpdate;
      if (elapsedSeconds.count() > o.validity || !o.count) {
        std::cout << "Storing entry: " << o.path << " copy " << o.count
                  << " after " << (o.count ? elapsedSeconds.count() : 0.) << "s\n";

        auto uploadStart = std::chrono::high_resolution_clock::now();
        std::thread th(pushObject, std::ref(api), std::cref(o));
        th.detach();
        //pushObject(api, o);
        auto uploadEnd = std::chrono::high_resolution_clock::now();
        DurSec uploadTime = uploadEnd - uploadStart;
        LOG(INFO) << "Took " << uploadTime.count() << " to load " << o.sz << " bytes object";
        o.count++;
        o.lastUpdate = timeLoopStart;
        if (minTLeft < 0.9 * o.validity) {
          minTLeft = o.validity;
        }
      }
    }
    usleep(minTLeft * 0.9 * 1e6);
  }
}

std::vector<CCDBObj> readObjectsList(const std::string& fname)
{
  std::vector<CCDBObj> objs;
  std::ifstream inFile(fname);
  if (!inFile.is_open()) {
    LOG(ERROR) << "Failed to open input file " << fname;
    return objs;
  }
  std::string str;
  while (std::getline(inFile, str)) {
    str = std::regex_replace(str, std::regex("^\\s+|\\s+$|\\s+\r\n$|\\s+\r$|\\s+\n$"), "$1");
    if (str[0] == '#' || str.empty()) {
      continue;
    }
    std::stringstream ss(str);
    std::string path;
    float sz = 0.f, sec = 0.f;
    ss >> path;
    ss >> sz;
    ss >> sec;
    if (sz == 0 || sec == 0) {
      LOG(ERROR) << "Invalid data for " << path;
      objs.clear();
      break;
    }
    LOG(INFO) << "Account " << path << " size= " << sz << " validity= " << sec;
    objs.emplace_back(path, sz, sec);
  }
  return objs;
}

void pushObject(o2::ccdb::CcdbApi& api, const CCDBObj& obj)
{
  std::vector<uint8_t> buff(obj.sz);
  for (auto& c : buff) {
    c = gRandom->Integer(0xff);
  }
  std::map<std::string, std::string> metadata; // can be empty
  metadata["responsible"] = "nobody";
  metadata["custom"] = "whatever";
  auto now = std::chrono::system_clock::now();
  auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto timeStamp = now_ms.time_since_epoch();
  api.storeAsTFileAny(&buff, obj.path, metadata, timeStamp.count(), 1670700184549); // one year validity time
}

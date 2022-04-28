#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "CCDB/CcdbApi.h"
#include "Framework/Logger.h"
#include <array>
#endif

// create and upload orbit reset object to CCDB
constexpr Long64_t DummyTime = -1;

void CreateCTPOrbitResetObject(const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080", long t = DummyTime, long tmin = 0, long tmax = -1)
{

  if (t == DummyTime) {
    t = std::chrono::time_point_cast<std::chrono::microseconds>(std::chrono::system_clock::now()).time_since_epoch().count();
  }
  std::vector<Long64_t> rt{t};
  const std::string objName{"CTP/Calib/OrbitReset"};
  o2::ccdb::CcdbApi api;
  api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
  map<string, string> metadata; // can be empty
  metadata["comment"] = "CTP Orbit reset";
  api.storeAsTFileAny(&rt, objName, metadata, tmin, tmax);
  LOGP(info, "Uploaded CTP Oribt reset time {} to {}", t, objName);
}

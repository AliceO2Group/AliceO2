#if !defined(__CLING__) || defined(__ROOTCLING__)
//#define ENABLE_UPGRADES
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "CCDB/CcdbApi.h"
#include "Framework/Logger.h"
#include <vector>
#include <fmt/format.h>
#endif

using DetID = o2::detectors::DetID;

// upload dummy alignment objects to CCDB
void UploadDummyAlignment(const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080", long tmin = 0, long tmax = -1)
{
  DetID::mask_t dets = DetID::FullMask & (~DetID::getMask(DetID::CTP));
  LOG(INFO) << "Mask = " << dets;
  o2::ccdb::CcdbApi api;
  api.init(ccdbHost.c_str()); // or http://localhost:8080 for a local installation
  std::vector<o2::detectors::AlignParam> params;

  for (auto id = DetID::First; id <= DetID::Last; id++) {
    if (!dets[id]) {
      continue;
    }
    map<string, string> metadata; // can be empty
    DetID det(id);
    metadata["comment"] = fmt::format("Empty alignment object for {}", det.getName());
    api.storeAsTFileAny(&params, o2::base::NameConf::getAlignmentPath(det), metadata, tmin, tmax);
    LOG(INFO) << "Uploaded dummy alignment for " << det.getName();
  }
}

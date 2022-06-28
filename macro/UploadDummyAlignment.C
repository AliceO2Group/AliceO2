#if !defined(__CLING__) || defined(__ROOTCLING__)
//#define ENABLE_UPGRADES
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "DetectorsCommonDataFormats/AlignParam.h"
#include "CCDB/CcdbApi.h"
#include "Framework/Logger.h"
#include <vector>
#include <fmt/format.h>
#endif

using DetID = o2::detectors::DetID;

// upload dummy alignment objects to CCDB
void UploadDummyAlignment(const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080", long tmin = 1, long tmax = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP, DetID::mask_t msk = DetID::FullMask)
{
  DetID::mask_t dets = msk & DetID::FullMask;
  LOG(info) << "Mask = " << dets;
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
    metadata["default"] = "true"; // tag default objects
    metadata["Created"] = "2";    // tag default objects
    api.storeAsTFileAny(&params, o2::base::DetectorNameConf::getAlignmentPath(det), metadata, tmin, tmax);
    LOG(info) << "Uploaded dummy alignment for " << det.getName();
  }
}

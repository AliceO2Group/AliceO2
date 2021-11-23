#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "DetectorsCommonDataFormats/NameConf.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "Framework/Logger.h"
#include <fmt/format.h>
#include "DetectorsBase/MatLayerCylSet.h"
#include "CommonUtils/StringUtils.h"
#endif

// upload material LUT to CCDB
bool UploadMatBudLUT(const std::string& matLUTFile, long tmin = 0, long tmax = -1, const std::string& url = "GLO/Param/MatLUT", const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080")
{
  o2::base::MatLayerCylSet* lut = nullptr;
  if (o2::utils::Str::pathExists(matLUTFile)) {
    lut = o2::base::MatLayerCylSet::loadFromFile(matLUTFile);
  } else {
    LOG(error) << "Material LUT " << matLUTFile << " file is absent";
    return false;
  }

  o2::ccdb::CcdbApi api;
  api.init(ccdbHost.c_str());   // or http://localhost:8080 for a local installation
  map<string, string> metadata; // can be empty
  metadata["comment"] = "Material lookup table";
  api.storeAsTFileAny(lut, url, metadata, tmin, tmax);
  return true;
}

auto fetchMatBudLUT(const std::string& url = "GLO/Param/MatLUT", const std::string& ccdbHost = "http://ccdb-test.cern.ch:8080")
{
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL(ccdbHost);
  auto lut = o2::base::MatLayerCylSet::rectifyPtrFromFile(mgr.get<o2::base::MatLayerCylSet>(url));
  if (!lut) {
    LOG(error) << "Failed to fetch material LUT from " << ccdbHost << "/" << url;
  }
  return lut;
}

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

#include "TPCDistortionMapFile.h"
#include "Framework/InputRecord.h"
#include "Framework/Logger.h"
#include "Framework/ProcessingContext.h"
#include "TPCCalibration/CorrMapParam.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSpaceCharge/SpaceCharge.h"
#include <Compression.h>
#include <TFile.h>
#include <TNamed.h>
#include <TStopwatch.h>
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#include <filesystem>
#include <memory>
#include <fmt/format.h>

using namespace o2::tpc;
using SC = o2::tpc::SpaceCharge<float>;

namespace
{
constexpr const char* ObjectName = "ccdb_object";
constexpr const char* KeyName = "tpc_distortion_key";
constexpr int KeyFormat = 1; ///< version of the file content

/// lock on a file, shared or exclusive, released at destruction
class FileLock
{
 public:
  FileLock(const std::string& path, int operation)
  {
    mFD = ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
    if (mFD >= 0 && ::flock(mFD, operation) != 0) {
      ::close(mFD);
      mFD = -1;
    }
  }
  ~FileLock()
  {
    if (mFD >= 0) {
      ::flock(mFD, LOCK_UN);
      ::close(mFD);
    }
  }
  bool ok() const { return mFD >= 0; }
  FileLock(const FileLock&) = delete;
  FileLock& operator=(const FileLock&) = delete;

 private:
  int mFD = -1;
};

std::string getETag(o2::framework::ProcessingContext& pc, const char* binding)
{
  const auto& headers = pc.inputs().get<o2::framework::CCDBMetadataExtractor>(binding);
  const auto it = headers.find("ETag");
  return it == headers.end() ? std::string{} : it->second;
}

std::unique_ptr<SC> readMap(const std::string& path, const std::string& key, bool warn)
{
  if (!std::filesystem::exists(path)) {
    return nullptr;
  }
  TFile file(path.c_str(), "READ");
  std::unique_ptr<TNamed> storedKey(file.IsZombie() ? nullptr : file.Get<TNamed>(KeyName));
  if (!storedKey || key != storedKey->GetTitle()) {
    if (warn) {
      LOGP(warning, "TPC: {} was written for other distortion inputs and will be replaced", path);
    }
    return nullptr;
  }
  return std::unique_ptr<SC>(file.Get<SC>(ObjectName));
}

bool writeMap(const std::string& path, const std::string& key, const SC& sc)
{
  // written under another name and renamed, so that readers never see a partial file
  const auto tmpPath = fmt::format("{}.{}.tmp", path, ::getpid());
  bool ok = false;
  {
    TFile file(tmpPath.c_str(), "RECREATE", "", ROOT::RCompressionSetting::EDefaults::kUseAnalysis);
    if (!file.IsZombie()) {
      TNamed named(KeyName, key.c_str());
      ok = file.WriteObjectAny(&sc, "o2::tpc::SpaceCharge<float>", ObjectName) > 0 && file.WriteObject(&named, KeyName) > 0;
      file.Close();
      ok = ok && !file.TestBit(TFile::kWriteError);
    }
  }
  std::error_code ec;
  if (ok) {
    std::filesystem::rename(tmpPath, path, ec);
  } else {
    std::filesystem::remove(tmpPath, ec);
  }
  return ok && !ec;
}
} // namespace

void TPCDistortionMapFile::prepare(o2::framework::ProcessingContext& pc, Digitizer& digitizer, float meanLumi, float meanLumiDerivative,
                                   const std::function<void()>& compute)
{
  const auto etagDist = getETag(pc, "tpcdistortions");
  const auto etagDeriv = getETag(pc, "tpcdistortionsderiv");
  if (etagDist.empty() || etagDeriv.empty()) {
    LOGP(warning, "TPC: no ETag for the distortion maps, the scaled distortions are computed and not shared through {}", mPath);
    compute();
    return;
  }
  const auto key = fmt::format("format {} dist {} deriv {} lumiInst {} meanLumi {} meanLumiDerivative {}", KeyFormat, etagDist, etagDeriv,
                               CorrMapParam::Instance().lumiInst, meanLumi, meanLumiDerivative);
  if (key == mKey) {
    return; // the digitizer holds them already
  }

  const auto lockPath = mPath + ".lock";
  TStopwatch timer;
  auto read = [&](bool warn) {
    auto sc = readMap(mPath, key, warn);
    if (sc) {
      digitizer.setScaledSCDistortions(std::move(sc));
      LOGP(info, "TPC: scaled distortions read from {} in {:.1f} s", mPath, timer.RealTime());
      mKey = key;
    }
    return mKey == key;
  };
  {
    // digitizers that find the file read it concurrently
    FileLock lock(lockPath, LOCK_SH);
    if (!lock.ok()) {
      LOGP(warning, "TPC: cannot lock {}, the scaled distortions are computed and not shared", lockPath);
      compute();
      return;
    }
    if (read(false)) {
      return;
    }
  }
  // one digitizer computes and writes, the others wait and then read
  FileLock lock(lockPath, LOCK_EX);
  if (!lock.ok()) {
    LOGP(warning, "TPC: cannot lock {}, the scaled distortions are computed and not shared", lockPath);
    compute();
    return;
  }
  if (!read(true)) {
    compute();
    timer.Start();
    if (writeMap(mPath, key, *digitizer.getSCDistortions())) {
      LOGP(info, "TPC: scaled distortions written to {} in {:.1f} s", mPath, timer.RealTime());
      mKey = key;
    } else {
      LOGP(warning, "TPC: cannot write the scaled distortions to {}", mPath);
    }
  }
}

// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "AODInputFile.h"
#include "NativeAOD.h"
#include "AODNative/Errors.h"
#include "Framework/Logger.h"
#include <TDirectory.h>
#include <TFile.h>
#include <TGrid.h>
#include <TGridResult.h>
#include <TKey.h>
#include <TMap.h>
#include <TObjString.h>
#include <TUrl.h>
#include <fmt/format.h>
#include <chrono>
#if __has_include(<TJAlienFile.h>)
#include <TJAlienFile.h>
#endif

namespace o2::framework
{

std::shared_ptr<RootAODInputFile> RootAODInputFile::open(std::string const& name, RootObjectReadingFactory& factory, TFile* external)
{
  TFile* file = external ? external : TFile::Open(name.c_str());
  if (!file || file->IsZombie()) {
    if (file && !external) {
      delete file;
    }
    return nullptr;
  }
  auto filesystem = std::make_shared<TFileFileSystem>(file, 50 * 1024 * 1024, factory, external == nullptr);
  return std::shared_ptr<RootAODInputFile>(new RootAODInputFile(file, std::move(filesystem)));
}

RootAODInputFile::RootAODInputFile(TFile* file, std::shared_ptr<TFileFileSystem> filesystem)
  : mFile(file), mFilesystem(std::move(filesystem))
{
}

std::string RootAODInputFile::name() const { return mFile->GetName(); }
uint64_t RootAODInputFile::size() const { return mFile->GetSize(); }
uint64_t RootAODInputFile::bytesRead() const { return mFile->GetBytesRead(); }
uint64_t RootAODInputFile::readCalls() const { return mFile->GetReadCalls(); }

std::vector<AODInputFile::Key> RootAODInputFile::topLevelKeys()
{
  std::vector<Key> keys;
  for (auto* obj : *mFile->GetListOfKeys()) {
    auto* key = (TKey*)obj;
    keys.push_back({key->GetName(), key->GetClassName()});
  }
  return keys;
}

std::vector<std::string> RootAODInputFile::keysIn(std::string const& dir)
{
  std::vector<std::string> names;
  auto* d = (TDirectory*)mFile->GetObjectChecked(dir.c_str(), TClass::GetClass("TDirectory"));
  if (!d) {
    return names;
  }
  for (auto* key : *d->GetListOfKeys()) {
    names.emplace_back(key->GetName());
  }
  return names;
}

std::map<std::string, std::string> RootAODInputFile::readStringMap(char const* key)
{
  std::map<std::string, std::string> result;
  std::unique_ptr<TMap> map{(TMap*)mFile->Get(key)};
  if (!map) {
    return result;
  }
  map->SetOwnerKeyValue(true, true);
  std::unique_ptr<TIterator> it{map->MakeIterator()};
  while (auto* key = it->Next()) {
    auto* value = (TObjString*)map->GetValue(key);
    result[key->GetName()] = value ? value->String().Data() : "";
  }
  return result;
}

std::map<std::string, std::string> RootAODInputFile::parentFiles() { return readStringMap("parentFiles"); }
std::map<std::string, std::string> RootAODInputFile::metaData() { return readStringMap("metaData"); }

std::string RootAODInputFile::monitoringDetails() const
{
#if __has_include(<TJAlienFile.h>)
  if (auto* alienFile = dynamic_cast<TJAlienFile*>(mFile)) {
    return fmt::format(",se={},open_time={:.1f}", alienFile->GetSE(), alienFile->GetElapsed());
  }
#endif
  return "";
}

std::vector<GridReplica> gridReplicas(std::string const& lfn)
{
  TUrl url(lfn.c_str());
  if (url.GetAnchor() && url.GetAnchor()[0] != '\0') {
    return {};
  }
  if (!gGrid) {
    TGrid::Connect("alien://");
  }
  if (!gGrid) {
    return {};
  }
  // As TJAlienFile::Open does for reading.
  std::unique_ptr<TGridResult> result(gGrid->Command(fmt::format("access read {}", url.GetFile()).c_str()));
  if (!result) {
    return {};
  }
  std::vector<GridReplica> replicas;
  for (auto* object : *result) {
    auto* map = dynamic_cast<TMap*>(object);
    if (!map) {
      continue;
    }
    auto value = [map](char const* key) -> std::string {
      auto* v = map->GetValue(key);
      return v ? v->GetName() : "";
    };
    auto pfn = value("url");
    auto envelope = value("envelope");
    if (pfn.empty() || envelope.empty() || pfn.find('#') != std::string::npos) {
      continue;
    }
    replicas.push_back({pfn + "?&authz=" + envelope, value("se")});
  }
  return replicas;
}

std::shared_ptr<AODInputFile> openNativeAOD(std::string const& filename, RootObjectReadingFactory& factory, o2::aod::native::Options const& options)
{
  if (!filename.starts_with("alien://")) {
    return NativeAODInputFile::open(filename, factory, options);
  }
  auto start = std::chrono::steady_clock::now();
  auto replicas = gridReplicas(filename);
  if (replicas.empty()) {
    throw o2::aod::native::IoError(filename, "no replica the native reader can read");
  }
  std::string errors;
  for (auto const& replica : replicas) {
    try {
      auto file = NativeAODInputFile::open(replica.url, factory, options, filename);
      file->setGridInfo(replica.se, std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count());
      return file;
    } catch (o2::aod::native::IoError const& e) {
      LOGP(info, "Could not open replica of \"{}\" on {}: {}", filename, replica.se, e.what());
      errors += fmt::format(" {}: {};", replica.se, e.what());
    }
  }
  throw o2::aod::native::IoError(filename, "no replica could be opened:" + errors);
}

} // namespace o2::framework

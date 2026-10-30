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

#include "NativeAOD.h"
#include "AODNative/Errors.h"
#include "Framework/RuntimeError.h"
#include <algorithm>
#include <cstdio>

namespace o2::framework
{

using namespace o2::aod::native;

NativeAODFileSystem::NativeAODFileSystem(std::shared_ptr<RootFile> file, Options options, RootObjectReadingFactory& factory)
  : mFile(std::move(file)), mOptions(options), mObjectFactory(factory)
{
}

namespace
{
// "DF_<n>/<tree>", with or without a leading slash. Empty strings if it is not that.
std::pair<std::string, std::string> splitTreePath(std::string const& path)
{
  size_t start = path.starts_with("/") ? 1 : 0;
  auto slash = path.find('/', start);
  if (slash == std::string::npos) {
    return {};
  }
  return {path.substr(start, slash - start), path.substr(slash + 1)};
}
} // namespace

NativeTreeHandle* NativeAODFileSystem::getTreeHandle(std::string const& path)
{
  auto [df, treeName] = splitTreePath(path);
  if (df.empty()) {
    return nullptr;
  }
  std::lock_guard lock{mMutex};
  return findTree(df, treeName);
}

std::shared_ptr<ReadPlan> NativeAODFileSystem::planFor(std::string const& path)
{
  auto [df, treeName] = splitTreePath(path);
  if (df.empty() || mFile->source().canMap()) {
    return nullptr;
  }
  std::lock_guard lock{mMutex};
  if (df != mPlanDF) {
    auto previous = std::move(mPlanTrees);
    mPlanTrees.clear();
    mPlanDF = df;
    // Also when the directory is already known, e.g. the first DF, read at open.
    adoptLookahead(df);
    directory(df);
    if (mAdoptedDF == df) {
      mPlan = std::move(mAdoptedPlan);
      mPlanTrees = std::move(mAdoptedTrees);
    } else {
      mPlan = std::make_shared<ReadPlan>(mFile->sourcePtr(), mOptions, mPool);
    }
    mAdoptedDF.clear();
    auto trees = previous;
    if (std::ranges::find(trees, treeName) == trees.end()) {
      trees.push_back(treeName);
    }
    if (!previous.empty()) {
      mFile->setTreesToPrefetch(trees);
    }
    for (auto& name : previous) {
      addToPlan(name);
    }
    // Only once a whole DF has shown which trees the reader asks for.
    if (!previous.empty() && mFile->source().isRemote() && mOptions.prefetchDepth > 0) {
      startLookahead(df, std::move(trees));
    }
  }
  addToPlan(treeName);
  return mPlan;
}

void NativeAODFileSystem::adoptLookahead(std::string const& df)
{
  if (!mLookahead.valid() || mLookaheadDF != df) {
    return;
  }
  auto lookahead = mLookahead.get();
  if (!lookahead) {
    return;
  }
  mDirectories.emplace(df, std::move(lookahead->keys));
  mAdoptedDF = df;
  mAdoptedPlan = lookahead->plan;
  mAdoptedTrees.clear();
  for (auto& [name, tree] : lookahead->trees) {
    // The plan already has the baskets of these trees.
    mTrees.emplace(df + "/" + name, std::make_unique<NativeTreeHandle>(NativeTreeHandle{std::move(tree), mFile->sourcePtr(), mOptions}));
    mAdoptedTrees.push_back(name);
  }
}

std::vector<KeyInfo> const& NativeAODFileSystem::directory(std::string const& df)
{
  if (auto it = mDirectories.find(df); it != mDirectories.end()) {
    return it->second;
  }
  adoptLookahead(df);
  if (auto it = mDirectories.find(df); it != mDirectories.end()) {
    return it->second;
  }
  return mDirectories.emplace(df, mFile->keysOf(df)).first->second;
}

std::vector<KeyInfo> NativeAODFileSystem::keysOf(std::string const& df)
{
  std::lock_guard lock{mMutex};
  return directory(df);
}

void NativeAODFileSystem::setReadOrder(std::vector<std::string> dfs)
{
  std::lock_guard lock{mMutex};
  mDataFrames = std::move(dfs);
}

void NativeAODFileSystem::startLookahead(std::string const& afterDF, std::vector<std::string> trees)
{
  if (mDataFrames.empty()) {
    mDataFrames = mFile->dataFrames();
  }
  if (!mLookaheadFile) {
    mLookaheadFile = std::make_shared<RootFile>(*mFile);
  }
  auto it = std::ranges::find(mDataFrames, afterDF);
  if (it == mDataFrames.end() || ++it == mDataFrames.end()) {
    return;
  }
  mLookaheadDF = *it;
  // Errors are left to the normal path, which reports them when the DF is read.
  mLookahead = std::async(std::launch::async, [file = mLookaheadFile, df = *it, trees = std::move(trees), options = mOptions, pool = mPool]() -> std::optional<Lookahead> {
    try {
      Lookahead lookahead{df, {}, {}, std::make_shared<ReadPlan>(file->sourcePtr(), options, pool)};
      file->setTreesToPrefetch(trees);
      lookahead.keys = file->keysOf(df);
      for (auto const& name : trees) {
        for (auto const& key : lookahead.keys) {
          if (key.name == name && key.className == "TTree") {
            lookahead.trees.emplace_back(name, file->readTree(key));
            lookahead.plan->add(lookahead.trees.back().second);
          }
        }
      }
      lookahead.plan->fetch();
      return lookahead;
    } catch (std::exception const&) {
      return std::nullopt;
    }
  });
}

void NativeAODFileSystem::addToPlan(std::string const& treeName)
{
  if (std::ranges::find(mPlanTrees, treeName) != mPlanTrees.end()) {
    return;
  }
  NativeTreeHandle* handle = nullptr;
  try {
    handle = findTree(mPlanDF, treeName);
  } catch (FormatError const&) {
    // Reported when the tree itself is asked for.
  }
  if (handle) {
    mPlan->add(handle->tree);
    mPlanTrees.push_back(treeName);
  }
}

NativeTreeHandle* NativeAODFileSystem::findTree(std::string const& df, std::string const& treeName)
{
  auto path = df + "/" + treeName;
  if (auto it = mTrees.find(path); it != mTrees.end()) {
    return it->second.get();
  }
  if (!mDirectories.contains(df)) {
    bool exists = false;
    for (auto& k : mFile->topKeys()) {
      exists |= k.name == df;
    }
    if (!exists) {
      return nullptr;
    }
  }
  for (auto& key : directory(df)) {
    if (key.name == treeName && key.className == "TTree") {
      auto handle = std::make_unique<NativeTreeHandle>(NativeTreeHandle{mFile->readTree(key), mFile->sourcePtr(), mOptions});
      return mTrees.emplace(path, std::move(handle)).first->second.get();
    }
  }
  return nullptr;
}

std::shared_ptr<RootObjectHandler> NativeAODFileSystem::GetObjectHandler(arrow::dataset::FileSource source)
{
  for (auto& capability : mObjectFactory.capabilities) {
    auto objectPath = capability.lfn2objectPath(source.path());
    void* handle = capability.getHandle(shared_from_this(), objectPath);
    if (!handle) {
      continue;
    }
    return std::make_shared<RootObjectHandler>(handle, capability.factory().format());
  }
  throw runtime_error_f("Unable to get handler for object %s", source.path().c_str());
}

bool NativeAODFileSystem::CheckSupport(arrow::dataset::FileSource source)
{
  for (auto& capability : mObjectFactory.capabilities) {
    if (capability.getHandle(shared_from_this(), capability.lfn2objectPath(source.path()))) {
      return true;
    }
  }
  return false;
}

arrow::Result<arrow::fs::FileInfo> NativeAODFileSystem::GetFileInfo(const std::string& path)
{
  arrow::fs::FileInfo result;
  result.set_path(path);
  result.set_type(arrow::fs::FileType::NotFound);
  for (auto& k : mFile->topKeys()) {
    if (k.name == path) {
      result.set_type(k.className == "TTree" ? arrow::fs::FileType::File : arrow::fs::FileType::Directory);
      return result;
    }
  }
  if (getTreeHandle(path)) {
    result.set_type(arrow::fs::FileType::File);
  }
  return result;
}

std::shared_ptr<NativeAODInputFile> NativeAODInputFile::open(std::string const& url, RootObjectReadingFactory& factory, Options const& options, std::string const& name)
{
  auto result = std::shared_ptr<NativeAODInputFile>(new NativeAODInputFile);
  result->mName = name.empty() ? url : name;
  result->mFile = std::make_shared<RootFile>(ByteSource::open(url, options));
  result->mFilesystem = std::make_shared<NativeAODFileSystem>(result->mFile, options, factory);
  result->mFilesystem->setName(result->mName);
  result->mParentFiles = result->mFile->readStringMap("parentFiles");
  result->mMetaData = result->mFile->readStringMap("metaData");
  // Parse the trees of the first DF now: a layout or version we do not support
  // is then found while the file can still be read with ROOT instead.
  auto dfs = result->mFile->dataFrames();
  if (!dfs.empty()) {
    for (auto& key : result->mFile->keysOf(dfs.front())) {
      if (key.className == "TTree") {
        result->mFilesystem->getTreeHandle(dfs.front() + "/" + key.name);
      }
    }
  }
  return result;
}

void NativeAODInputFile::setGridInfo(std::string se, double openTime)
{
  mSE = std::move(se);
  mOpenTime = openTime;
}

std::string NativeAODInputFile::monitoringDetails() const
{
  if (mSE.empty()) {
    return "";
  }
  char openTime[32];
  std::snprintf(openTime, sizeof(openTime), "%.1f", mOpenTime);
  return ",se=" + mSE + ",open_time=" + openTime;
}

uint64_t NativeAODInputFile::size() const { return mFile->source().size(); }
uint64_t NativeAODInputFile::bytesRead() const { return mFile->source().bytesRead(); }
uint64_t NativeAODInputFile::readCalls() const { return mFile->source().readCalls(); }

std::vector<AODInputFile::Key> NativeAODInputFile::topLevelKeys()
{
  std::vector<Key> keys;
  for (auto& k : mFile->topKeys()) {
    keys.push_back({k.name, k.className});
  }
  return keys;
}

std::vector<std::string> NativeAODInputFile::keysIn(std::string const& dir)
{
  std::vector<std::string> names;
  for (auto& k : mFile->topKeys()) {
    if (k.name == dir && (k.className == "TDirectoryFile" || k.className == "TDirectory")) {
      for (auto& key : mFilesystem->keysOf(dir)) {
        names.push_back(key.name);
      }
      break;
    }
  }
  return names;
}

} // namespace o2::framework

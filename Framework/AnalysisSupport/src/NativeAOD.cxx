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

namespace
{
Options withBudget(Options options, RootFile& file)
{
  if (options.bufferBudget == Options::kAutoBufferBudget) {
    options.bufferBudget = (file.source().isRemote() ? 64 : 16) * size_t{1024 * 1024};
  }
  return options;
}
} // namespace

NativeAODFileSystem::NativeAODFileSystem(std::shared_ptr<RootFile> file, Options options, RootObjectReadingFactory& factory)
  : mFile(std::move(file)), mOptions(withBudget(options, *mFile)), mObjectFactory(factory), mPool(std::make_shared<ReadBufferPool>(mOptions.bufferBudget))
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
  if (df.empty()) {
    return nullptr;
  }
  std::lock_guard lock{mMutex};
  if (mFile->source().canMap()) {
    releaseMappedDF(df);
    return nullptr;
  }
  if (df != mPlanDF) {
    auto previous = std::move(mPlanTrees);
    mPlanTrees.clear();
    mPlanOf.clear();
    mPlanBytes = 0;
    mPlanDF = df;
    // Also when the directory is already known, e.g. the first DF, read at open.
    adoptLookahead(df);
    directory(df);
    if (mAdoptedDF == df) {
      mPlan = std::move(mAdoptedPlan);
      for (auto& name : mAdoptedTrees) {
        mPlanOf[name] = mPlan;
        if (auto* handle = findTree(df, name)) {
          mPlanBytes += ReadPlan::bytesOf(handle->tree);
        }
      }
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
  // Let go of the plan once its tree is handed out: the plan, and its buffers,
  // then go away when the reads of its trees are done.
  auto it = mPlanOf.find(treeName);
  if (it == mPlanOf.end()) {
    return nullptr;
  }
  auto plan = std::move(it->second);
  mPlanOf.erase(it);
  return plan;
}

void NativeAODFileSystem::releaseMappedDF(std::string const& df)
{
  if (df == mMappedDF) {
    return;
  }
  // The pages of a DF which is done stay mapped and count in the RSS of the
  // device until the kernel reclaims them: give them back. Rereading them is
  // still correct, as they come back from the page cache.
  uint64_t begin = UINT64_MAX;
  uint64_t end = 0;
  for (auto it = mTrees.lower_bound(mMappedDF + "/"); it != mTrees.end() && it->first.starts_with(mMappedDF + "/"); ++it) {
    for (auto const& branch : it->second->tree.branches) {
      for (auto const& basket : branch.baskets) {
        if (!basket.embedded && basket.nbytes > 0) {
          begin = std::min(begin, basket.seek);
          end = std::max(end, basket.seek + basket.nbytes);
        }
      }
    }
  }
  if (!mMappedDF.empty() && begin < end) {
    mFile->source().dontNeed(begin, end - begin);
  }
  mMappedDF = df;
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
      // Only the trees which fit in the budget; the others are read when asked for.
      uint64_t bytes = 0;
      for (auto const& name : trees) {
        for (auto const& key : lookahead.keys) {
          if (key.name == name && key.className == "TTree") {
            auto tree = file->readTree(key);
            auto treeBytes = ReadPlan::bytesOf(tree);
            if (options.bufferBudget > 0 && bytes > 0 && bytes + treeBytes > options.bufferBudget) {
              lookahead.plan->fetch();
              return lookahead;
            }
            bytes += treeBytes;
            lookahead.trees.emplace_back(name, std::move(tree));
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
  } catch (std::exception const&) {
    // Reported when the tree itself is asked for.
  }
  if (handle) {
    auto bytes = ReadPlan::bytesOf(handle->tree);
    if (mOptions.bufferBudget > 0 && mPlanBytes > 0 && mPlanBytes + bytes > mOptions.bufferBudget) {
      mPlan = std::make_shared<ReadPlan>(mFile->sourcePtr(), mOptions, mPool);
      mPlanBytes = 0;
    }
    mPlan->add(handle->tree);
    mPlanBytes += bytes;
    mPlanOf[treeName] = mPlan;
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
    if (capability.name != "native-ttree") {
      continue;
    }
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
    if (capability.name == "native-ttree" && capability.getHandle(shared_from_this(), capability.lfn2objectPath(source.path()))) {
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
  return open(ByteSource::open(url, options), factory, options, name.empty() ? url : name);
}

std::shared_ptr<NativeAODInputFile> NativeAODInputFile::open(std::unique_ptr<ByteSource> source, RootObjectReadingFactory& factory, Options const& options, std::string const& name)
{
  auto result = std::shared_ptr<NativeAODInputFile>(new NativeAODInputFile);
  result->mName = name;
  result->mFile = std::make_shared<RootFile>(std::move(source));
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

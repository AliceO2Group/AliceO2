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

#ifndef O2_FRAMEWORK_NATIVEAOD_H_
#define O2_FRAMEWORK_NATIVEAOD_H_

// Glue between DataInputDirector and the ROOT-free AO2D reader (O2::AODNative).
// No ROOT API is used here: everything goes through RootFile and ByteSource.

#include "AODInputFile.h"
#include "AODNative/ByteSource.h"
#include "AODNative/Container.h"
#include "AODNative/ReadPlan.h"
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace o2::framework
{

/// What a native tree reading handle points to: the parsed tree and where to
/// read its baskets from. Owned by the NativeAODFileSystem which created it.
struct NativeTreeHandle {
  o2::aod::native::TreeInfo tree;
  std::shared_ptr<o2::aod::native::ByteSource> source;
  o2::aod::native::Options options;
};

/// A filesystem over an AO2D read without ROOT, with the same path convention
/// as TFileFileSystem: "DF_<n>/<tree>".
class NativeAODFileSystem : public VirtualRootFileSystemBase
{
 public:
  NativeAODFileSystem(std::shared_ptr<o2::aod::native::RootFile> file, o2::aod::native::Options options, RootObjectReadingFactory& factory);

  std::string type_name() const override { return "native-aod"; }
  arrow::Result<arrow::fs::FileInfo> GetFileInfo(const std::string& path) override;
  std::shared_ptr<RootObjectHandler> GetObjectHandler(arrow::dataset::FileSource source) override;
  bool CheckSupport(arrow::dataset::FileSource source) override;

  /// The tree at @a path, parsed on first use. nullptr if there is no such
  /// tree. Throws o2::aod::native::FormatError if it cannot be parsed.
  NativeTreeHandle* getTreeHandle(std::string const& path);
  /// The read plan of the DF of the tree at @a path, with that tree in it;
  /// nullptr when the file is mapped and nothing needs reading. A new DF gets a
  /// new plan, with the trees read in the previous DF already in it: a reader
  /// asks for the same tables in every DF, so the first read of a DF fetches
  /// all of them together. For a remote file, the next DF is prepared in the
  /// background meanwhile (keys, TTrees and data of the same trees), so that
  /// its round trips overlap with the decoding of this one.
  std::shared_ptr<o2::aod::native::ReadPlan> planFor(std::string const& path);
  o2::aod::native::RootFile& file() { return *mFile; }
  /// The name the file was asked for by (an LFN for a grid file), which is
  /// not the URL it is read from.
  std::string const& name() const { return mName; }
  void setName(std::string name) { mName = std::move(name); }
  /// The keys of DF @a df, read once (or prepared ahead) for all users.
  std::vector<o2::aod::native::KeyInfo> keysOf(std::string const& df);
  /// The order the DFs are read in, which tells which DF to prepare next.
  /// Without it, RootFile::dataFrames() is assumed.
  void setReadOrder(std::vector<std::string> dfs);

 private:
  /// A DF prepared ahead: its keys, the TTrees learned so far and a read plan
  /// with their baskets already fetched.
  struct Lookahead {
    std::string df;
    std::vector<o2::aod::native::KeyInfo> keys;
    std::vector<std::pair<std::string, o2::aod::native::TreeInfo>> trees;
    std::shared_ptr<o2::aod::native::ReadPlan> plan;
  };
  NativeTreeHandle* findTree(std::string const& df, std::string const& treeName);
  /// Keys of @a df, taking them from the lookahead when it prepared this DF.
  std::vector<o2::aod::native::KeyInfo> const& directory(std::string const& df);
  void addToPlan(std::string const& treeName);
  /// Takes the keys, trees and plan the lookahead prepared for @a df.
  void adoptLookahead(std::string const& df);
  void startLookahead(std::string const& afterDF, std::vector<std::string> trees);

  std::shared_ptr<o2::aod::native::RootFile> mFile;
  std::string mName;
  o2::aod::native::Options mOptions;
  RootObjectReadingFactory& mObjectFactory;
  std::mutex mMutex;
  std::map<std::string, std::vector<o2::aod::native::KeyInfo>> mDirectories;
  std::map<std::string, std::unique_ptr<NativeTreeHandle>> mTrees;
  std::shared_ptr<o2::aod::native::ReadBufferPool> mPool = std::make_shared<o2::aod::native::ReadBufferPool>();
  std::shared_ptr<o2::aod::native::ReadPlan> mPlan;
  std::string mPlanDF;
  std::vector<std::string> mPlanTrees;
  // Used only by the lookahead task, so that mFile is never shared across threads.
  std::shared_ptr<o2::aod::native::RootFile> mLookaheadFile;
  std::vector<std::string> mDataFrames;
  std::future<std::optional<Lookahead>> mLookahead;
  std::string mLookaheadDF;
  // The plan and trees adopted for a DF, until planFor() starts that DF.
  std::string mAdoptedDF;
  std::shared_ptr<o2::aod::native::ReadPlan> mAdoptedPlan;
  std::vector<std::string> mAdoptedTrees;
};

/// An AO2D read with the ROOT-free reader.
class NativeAODInputFile : public AODInputFile
{
 public:
  /// Opens @a url and parses its header, keys, maps and the trees of its first
  /// DF, so that layouts and versions we cannot read are found here, when the
  /// file can still go to ROOT. Throws o2::aod::native::FormatError.
  /// @a name is what the file is called (an LFN for a grid file); @a url by default.
  static std::shared_ptr<NativeAODInputFile> open(std::string const& url, RootObjectReadingFactory& factory, o2::aod::native::Options const& options, std::string const& name = {});
  /// The storage element and the seconds the grid lookup and open took, for monitoring.
  void setGridInfo(std::string se, double openTime);
  std::string monitoringDetails() const override;

  std::string name() const override { return mName; }
  uint64_t size() const override;
  uint64_t bytesRead() const override;
  uint64_t readCalls() const override;
  std::vector<Key> topLevelKeys() override;
  std::vector<std::string> keysIn(std::string const& dir) override;
  std::map<std::string, std::string> parentFiles() override { return mParentFiles; }
  std::map<std::string, std::string> metaData() override { return mMetaData; }
  std::shared_ptr<VirtualRootFileSystemBase> filesystem() override { return mFilesystem; }
  void setReadOrder(std::vector<std::string> const& dfs) override { mFilesystem->setReadOrder(dfs); }

 private:
  NativeAODInputFile() = default;
  std::string mName;
  std::string mSE;
  double mOpenTime = 0;
  std::shared_ptr<o2::aod::native::RootFile> mFile;
  std::shared_ptr<NativeAODFileSystem> mFilesystem;
  std::map<std::string, std::string> mParentFiles;
  std::map<std::string, std::string> mMetaData;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_NATIVEAOD_H_

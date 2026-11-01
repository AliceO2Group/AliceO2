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

#ifndef O2_FRAMEWORK_AODINPUTFILE_H_
#define O2_FRAMEWORK_AODINPUTFILE_H_

#include "Framework/RootArrowFilesystem.h"
#include "AODNative/ByteSource.h"
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

class TFile;

namespace o2::framework
{

/// An AO2D file opened for reading, as DataInputDirector needs it: the list of
/// top-level keys, the file-level maps and an arrow filesystem to read the
/// tables from. This hides which library actually reads the file.
struct AODInputFile {
  struct Key {
    std::string name;
    std::string className;
  };

  virtual ~AODInputFile() = default;
  virtual std::string name() const = 0;
  virtual uint64_t size() const = 0;
  virtual uint64_t bytesRead() const = 0;
  virtual uint64_t readCalls() const = 0;
  /// Top-level keys, in the order the file lists them, all cycles.
  virtual std::vector<Key> topLevelKeys() = 0;
  /// Names of the keys inside the top-level directory @a dir, empty if there is none.
  virtual std::vector<std::string> keysIn(std::string const& dir) = 0;
  /// DF_<n> to parent file path. Empty if the file has no parentFiles.
  virtual std::map<std::string, std::string> parentFiles() = 0;
  virtual std::map<std::string, std::string> metaData() = 0;
  virtual std::shared_ptr<VirtualRootFileSystemBase> filesystem() = 0;
  /// Extra key=value pairs for the file monitoring metrics, starting with a comma.
  virtual std::string monitoringDetails() const { return ""; }
  /// The DF_<n> names in the order they will be read, for readers which read ahead.
  virtual void setReadOrder(std::vector<std::string> const& dfs) {}
};

/// A replica of an alien:// file: a root:// URL carrying its access envelope,
/// and the storage element it is on.
struct GridReplica {
  std::string url;
  std::string se;
};

/// Asks the AliEn catalogue (through gGrid, connecting if needed) for the
/// replicas to read @a lfn from, in the order it proposes. Empty if there are
/// none, or for a file inside an archive (alien://...#member), which only ROOT reads.
std::vector<GridReplica> gridReplicas(std::string const& lfn);

/// Opens @a filename with the native reader. A grid file is read from the
/// first of its replicas which can be opened, as TJAlienFile does; format
/// problems are not retried on other replicas. Throws o2::aod::native::FormatError.
std::shared_ptr<AODInputFile> openNativeAOD(std::string const& filename, RootObjectReadingFactory& factory, o2::aod::native::Options const& options);

/// An AO2D read with ROOT.
class RootAODInputFile : public AODInputFile
{
 public:
  /// Opens @a name with TFile::Open, or uses @a external (not owned) when given.
  /// Returns nullptr if the file cannot be opened.
  static std::shared_ptr<RootAODInputFile> open(std::string const& name, RootObjectReadingFactory& factory, TFile* external = nullptr);

  std::string name() const override;
  uint64_t size() const override;
  uint64_t bytesRead() const override;
  uint64_t readCalls() const override;
  std::vector<Key> topLevelKeys() override;
  std::vector<std::string> keysIn(std::string const& dir) override;
  std::map<std::string, std::string> parentFiles() override;
  std::map<std::string, std::string> metaData() override;
  std::shared_ptr<VirtualRootFileSystemBase> filesystem() override { return mFilesystem; }
  std::string monitoringDetails() const override;

 private:
  RootAODInputFile(TFile* file, std::shared_ptr<TFileFileSystem> filesystem);
  std::map<std::string, std::string> readStringMap(char const* key);

  TFile* mFile;
  std::shared_ptr<TFileFileSystem> mFilesystem;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_AODINPUTFILE_H_

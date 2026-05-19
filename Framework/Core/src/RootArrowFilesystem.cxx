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
#include "Framework/RootArrowFilesystem.h"
#include "Framework/RuntimeError.h"
#include <Rtypes.h>
#include <arrow/array/array_nested.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <memory>
#include <TFile.h>
#include <TBufferFile.h>
#include <TDirectoryFile.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/dataset/file_base.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <fmt/format.h>
#include <TKey.h>

template class
  std::shared_ptr<arrow::Array>;

namespace o2::framework
{
using arrow::Status;

TFileFileSystem::TFileFileSystem(TDirectoryFile* f, size_t readahead, RootObjectReadingFactory& factory, bool ownsFile)
  : VirtualRootFileSystemBase(),
    mFile(f),
    mObjectFactory(factory),
    mOwnsFile(ownsFile)
{
  ((TFile*)mFile)->SetReadaheadSize(50 * 1024 * 1024);
}

TFileFileSystem::~TFileFileSystem()
{
  if (mOwnsFile) {
    mFile->Close();
    delete mFile;
  }
}

std::shared_ptr<RootObjectHandler> TFileFileSystem::GetObjectHandler(arrow::dataset::FileSource source)
{
  // We use a plugin to create the actual objects inside the
  // file, so that we can support TTree and RNTuple at the same time
  // without having to depend on both.
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

bool TFileFileSystem::CheckSupport(arrow::dataset::FileSource source)
{
  // We use a plugin to create the actual objects inside the
  // file, so that we can support TTree and RNTuple at the same time
  // without having to depend on both.
  for (auto& capability : mObjectFactory.capabilities) {
    auto objectPath = capability.lfn2objectPath(source.path());

    void* handle = capability.getHandle(shared_from_this(), objectPath);
    if (handle) {
      return true;
    }
  }
  return false;
}

std::shared_ptr<VirtualRootFileSystemBase> TFileFileSystem::GetSubFilesystem(arrow::dataset::FileSource source)
{
  auto directory = (TDirectoryFile*)mFile->GetObjectChecked(source.path().c_str(), TClass::GetClass<TDirectory>());
  if (directory) {
    return std::shared_ptr<VirtualRootFileSystemBase>(new TFileFileSystem(directory, 50 * 1024 * 1024, mObjectFactory));
  }
  throw runtime_error_f("Unsupported file layout");
}

arrow::Result<arrow::fs::FileInfo> TFileFileSystem::GetFileInfo(const std::string& path)
{
  arrow::fs::FileInfo result;
  result.set_type(arrow::fs::FileType::NotFound);
  result.set_path(path);
  arrow::dataset::FileSource source(path, shared_from_this());

  auto fs = GetSubFilesystem(source);

  // For now we only support single trees.
  if (std::dynamic_pointer_cast<TFileFileSystem>(fs)) {
    result.set_type(arrow::fs::FileType::Directory);
    return result;
  }
  // Everything else is a file, if it was created.
  if (fs.get()) {
    result.set_type(arrow::fs::FileType::File);
  }
  return result;
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> TFileFileSystem::OpenOutputStream(
  const std::string& path,
  const std::shared_ptr<const arrow::KeyValueMetadata>& metadata)
{
  if (path == "/") {
    return std::make_shared<TDirectoryFileOutputStream>(this->GetFile());
  }

  auto* dir = dynamic_cast<TDirectoryFile*>(this->GetFile()->Get(path.c_str()));
  if (!dir) {
    return arrow::Status::Invalid(fmt::format("Unable to open directory {} in file {} ", path.c_str(), GetFile()->GetName()));
  }
  auto stream = std::make_shared<TDirectoryFileOutputStream>(dir);
  return stream;
}

arrow::Result<arrow::fs::FileInfo> VirtualRootFileSystemBase::GetFileInfo(std::string const&)
{
  arrow::fs::FileInfo result;
  result.set_type(arrow::fs::FileType::NotFound);
  return result;
}

arrow::Result<arrow::fs::FileInfoVector> VirtualRootFileSystemBase::GetFileInfo(const arrow::fs::FileSelector& select)
{
  arrow::fs::FileInfoVector results;
  auto selected = this->GetFileInfo(select.base_dir);
  if (selected.ok()) {
    results.emplace_back(*selected);
  }
  return results;
}

arrow::Status VirtualRootFileSystemBase::CreateDir(const std::string& path, bool recursive)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::DeleteDir(const std::string& path)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::CopyFile(const std::string& src, const std::string& dest)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::Move(const std::string& src, const std::string& dest)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::DeleteDirContents(const std::string& path, bool missing_dir_ok)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::DeleteRootDirContents()
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Status VirtualRootFileSystemBase::DeleteFile(const std::string& path)
{
  return arrow::Status::NotImplemented("Read only filesystem");
}

arrow::Result<std::shared_ptr<arrow::io::InputStream>> VirtualRootFileSystemBase::OpenInputStream(const std::string& path)
{
  return arrow::Status::NotImplemented("Non streamable filesystem");
}

arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> VirtualRootFileSystemBase::OpenInputFile(const std::string& path)
{
  return arrow::Status::NotImplemented("No random access file system");
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> VirtualRootFileSystemBase::OpenOutputStream(
  const std::string& path,
  const std::shared_ptr<const arrow::KeyValueMetadata>& metadata)
{
  return arrow::Status::NotImplemented("Non streamable filesystem");
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> VirtualRootFileSystemBase::OpenAppendStream(
  const std::string& path,
  const std::shared_ptr<const arrow::KeyValueMetadata>& metadata)
{
  return arrow::Status::NotImplemented("No random access file system");
}

// An arrow outputstream which allows to write to a ttree
TDirectoryFileOutputStream::TDirectoryFileOutputStream(TDirectoryFile* f)
  : mDirectory(f)
{
}

arrow::Status TDirectoryFileOutputStream::Close()
{
  mDirectory->GetFile()->Close();
  return arrow::Status::OK();
}

arrow::Result<int64_t> TDirectoryFileOutputStream::Tell() const
{
  return arrow::Result<int64_t>(arrow::Status::NotImplemented("Cannot move"));
}

arrow::Status TDirectoryFileOutputStream::Write(const void* data, int64_t nbytes)
{
  return arrow::Status::NotImplemented("Cannot write raw bytes to a TTree");
}

bool TDirectoryFileOutputStream::closed() const
{
  return mDirectory->GetFile()->IsOpen() == false;
}

TBufferFileFS::TBufferFileFS(TBufferFile* f, RootObjectReadingFactory& factory)
  : VirtualRootFileSystemBase(),
    mBuffer(f),
    mFilesystem(nullptr),
    mObjectFactory(factory)
{
}

arrow::Result<arrow::fs::FileInfo> TBufferFileFS::GetFileInfo(const std::string& path)
{
  arrow::fs::FileInfo result;
  result.set_type(arrow::fs::FileType::NotFound);
  result.set_path(path);
  arrow::dataset::FileSource source(path, shared_from_this());

  // Only once to avoid rereading the streamed tree.
  if (!mFilesystem.get()) {
    return result;
  }

  auto info = mFilesystem->GetFileInfo(path);
  if (!info.ok()) {
    return result;
  }

  result.set_type(info->type());
  return result;
}

bool TBufferFileFS::CheckSupport(arrow::dataset::FileSource source)
{
  // We use a plugin to create the actual objects inside the
  // file, so that we can support TTree and RNTuple at the same time
  // without having to depend on both.
  for (auto& capability : mObjectFactory.capabilities) {
    auto objectPath = capability.lfn2objectPath(source.path());

    mBuffer->SetBufferOffset(0);
    mBuffer->InitMap();
    TClass* serializedClass = mBuffer->ReadClass();
    mBuffer->SetBufferOffset(0);
    mBuffer->ResetMap();
    mBuffer->Reset();
    if (!serializedClass) {
      continue;
    }

    bool supports = capability.checkSupport(serializedClass->GetName());
    if (supports) {
      return true;
    }
  }
  return false;
}

std::shared_ptr<RootObjectHandler> TBufferFileFS::GetObjectHandler(arrow::dataset::FileSource source)
{
  // We use a plugin to create the actual objects inside the
  // file, so that we can support TTree and RNTuple at the same time
  // without having to depend on both.
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

RootObjectHandler::~RootObjectHandler() noexcept(false)
{
  if (payload) {
    throw runtime_error_f("Payload not owned");
  }
}

} // namespace o2::framework

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

// Reading of AO2D trees with the ROOT-free reader (O2::AODNative). A drop-in
// sibling of the TTree plugin: same schemas, same placeholder sizes and the
// same deferred write protocol, so that the decoded columns land directly in
// the shared memory DPL ships downstream. No ROOT API is used in this file.

#include "Framework/RootArrowFilesystem.h"
#include "Framework/Plugins.h"
#include "Framework/RuntimeError.h"
#include "Framework/Signpost.h"
#include "AODNative/ArrowTable.h"
#include "AODNative/Errors.h"
#include "DataInputDirector.h"
#include "NativeAOD.h"
#include <arrow/buffer.h>
#include <arrow/dataset/file_base.h>
#include <arrow/io/interfaces.h>
#include <arrow/memory_pool.h>
#include <arrow/record_batch.h>
#include <arrow/status.h>
#include <memory>

O2_DECLARE_DYNAMIC_LOG(native_aod);

namespace o2::framework
{

using o2::aod::native::BranchDecoder;
using o2::aod::native::ColumnOp;

// A fragment which holds a natively read tree, and the read operations
// planned for it.
class NativeAODFileFragment : public arrow::dataset::FileFragment
{
 public:
  NativeAODFileFragment(arrow::dataset::FileSource source,
                        std::shared_ptr<arrow::dataset::FileFormat> format,
                        arrow::compute::Expression partition_expression,
                        std::shared_ptr<arrow::Schema> physical_schema)
    : FileFragment(source, format, std::move(partition_expression), physical_schema)
  {
    auto rootFS = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(this->source().filesystem());
    if (rootFS.get() == nullptr) {
      throw runtime_error_f("Unknown filesystem %s when reading %s.",
                            source.filesystem()->type_name().c_str(), source.path().c_str());
    }
    auto objectHandler = rootFS->GetObjectHandler(source);
    if (!objectHandler->format->Equals(*format)) {
      throw runtime_error_f("Cannot read source %s with format %s to populate a NativeAODFileFragment.",
                            source.path().c_str(), objectHandler->format->type_name().c_str());
    }
    // The handle is owned by the filesystem, which the source keeps alive.
    mHandle = objectHandler->GetObjectAsOwner<NativeTreeHandle>().release();
    // Unless the file is mapped, the baskets come from the read plan of the DF.
    if (auto nativeFS = std::dynamic_pointer_cast<NativeAODFileSystem>(rootFS)) {
      if (auto plan = nativeFS->planFor(source.path())) {
        mSource = std::make_unique<o2::aod::native::PlannedByteSource>(mHandle->source, std::move(plan));
      }
    }
    mDecoder = std::make_unique<BranchDecoder>(mSource ? *mSource : *mHandle->source, mHandle->options);
  }

  NativeTreeHandle* handle() { return mHandle; }
  BranchDecoder& decoder() { return *mDecoder; }
  std::vector<ColumnOp>& ops() { return mOps; }

 private:
  NativeTreeHandle* mHandle = nullptr;
  std::unique_ptr<o2::aod::native::ByteSource> mSource;
  std::unique_ptr<BranchDecoder> mDecoder;
  std::vector<ColumnOp> mOps;
};

/// An OutputStream which does the reading of the input buffers directly
/// on writing, if needed. Each deferred operation is encoded in the source
/// buffer by an incremental number which can be used to lookup in the
/// fragment's ops the operation to perform.
class NativeAODDeferredReadOutputStream : public arrow::io::OutputStream
{
 public:
  NativeAODDeferredReadOutputStream(std::shared_ptr<NativeAODFileFragment> fragment,
                                    const std::shared_ptr<arrow::ResizableBuffer>& buffer)
    : mFragment(std::move(fragment)),
      buffer_(buffer),
      is_open_(true),
      capacity_(buffer->size()),
      position_(0),
      mutable_data_(buffer->mutable_data())
  {
  }

  // By the time we call the destructor, the contents
  // of the buffer are already moved to fairmq
  // for being sent.
  ~NativeAODDeferredReadOutputStream() override = default;

  arrow::Status Close() override
  {
    if (is_open_) {
      is_open_ = false;
      if (position_ < capacity_) {
        RETURN_NOT_OK(buffer_->Resize(position_, false));
      }
    }
    return arrow::Status::OK();
  }

  [[nodiscard]] bool closed() const override { return !is_open_; }
  [[nodiscard]] arrow::Result<int64_t> Tell() const override { return position_; }

  arrow::Status Write(const void* data, int64_t nbytes) override
  {
    if (ARROW_PREDICT_FALSE(!is_open_)) {
      return arrow::Status::IOError("OutputStream is closed");
    }
    if (ARROW_PREDICT_TRUE(nbytes == 0)) {
      return arrow::Status::OK();
    }
    if (ARROW_PREDICT_FALSE(position_ + nbytes >= capacity_)) {
      RETURN_NOT_OK(Reserve(nbytes));
    }
    auto& ops = mFragment->ops();
    // This is a real address which needs to be copied. Do it!
    auto ref = (int64_t)data;
    if (ref < 0 || ref >= (int64_t)ops.size()) {
      memcpy(mutable_data_ + position_, data, nbytes);
      position_ += nbytes;
      return arrow::Status::OK();
    }
    auto& op = ops[ref];
    if ((int64_t)op.byteSize != nbytes) {
      return arrow::Status::Invalid("Deferred read of ", nbytes, " bytes for an operation of ", op.byteSize);
    }
    // Decoding happens here, straight into the message buffer.
    o2::aod::native::executeColumnOp(mFragment->decoder(), op, mutable_data_ + position_);
    position_ += nbytes;
    return arrow::Status::OK();
  }

  using OutputStream::Write;

 private:
  arrow::Status Reserve(int64_t nbytes)
  {
    int64_t new_capacity = position_ + nbytes;
    if (new_capacity > capacity_) {
      RETURN_NOT_OK(buffer_->Resize(new_capacity));
      capacity_ = new_capacity;
      mutable_data_ = buffer_->mutable_data();
    }
    return arrow::Status::OK();
  }

  std::shared_ptr<NativeAODFileFragment> mFragment;
  std::shared_ptr<arrow::ResizableBuffer> buffer_;
  bool is_open_;
  int64_t capacity_;
  int64_t position_;
  uint8_t* mutable_data_;
};

class NativeAODFileWriteOptions : public arrow::dataset::FileWriteOptions
{
 public:
  NativeAODFileWriteOptions(std::shared_ptr<arrow::dataset::FileFormat> format)
    : FileWriteOptions(format)
  {
  }
};

class NativeAODFileFormat : public arrow::dataset::FileFormat
{
  size_t& mTotCompressedSize;
  size_t& mTotUncompressedSize;

 public:
  NativeAODFileFormat(size_t& totalCompressedSize, size_t& totalUncompressedSize)
    : FileFormat({}),
      mTotCompressedSize(totalCompressedSize),
      mTotUncompressedSize(totalUncompressedSize)
  {
  }

  std::string type_name() const override { return "native-ttree"; }

  bool Equals(const FileFormat& other) const override { return other.type_name() == this->type_name(); }

  arrow::Result<bool> IsSupported(const arrow::dataset::FileSource& source) const override
  {
    auto fs = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(source.filesystem());
    if (!fs) {
      return false;
    }
    return fs->CheckSupport(source);
  }

  arrow::Result<std::shared_ptr<arrow::Schema>> Inspect(const arrow::dataset::FileSource& source) const override
  {
    auto fs = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(source.filesystem());
    if (!fs.get()) {
      throw runtime_error_f("Unknown filesystem %s\n", source.filesystem()->type_name().c_str());
    }
    auto objectHandler = fs->GetObjectHandler(source);
    if (!objectHandler->format->Equals(*this)) {
      throw runtime_error_f("Unknown filesystem %s\n", source.filesystem()->type_name().c_str());
    }
    // The handle is owned by the filesystem.
    auto* handle = objectHandler->GetObjectAsOwner<NativeTreeHandle>().release();
    try {
      return o2::aod::native::physicalSchema(handle->tree);
    } catch (o2::aod::native::FormatError const& e) {
      return arrow::Status::IOError(e.what());
    }
  }

  arrow::Result<std::shared_ptr<arrow::dataset::FileFragment>> MakeFragment(
    arrow::dataset::FileSource source, arrow::compute::Expression partition_expression,
    std::shared_ptr<arrow::Schema> physical_schema) override
  {
    return std::make_shared<NativeAODFileFragment>(source, std::dynamic_pointer_cast<arrow::dataset::FileFormat>(shared_from_this()),
                                                   std::move(partition_expression), physical_schema);
  }

  arrow::Result<std::shared_ptr<arrow::dataset::FileWriter>> MakeWriter(std::shared_ptr<arrow::io::OutputStream>, std::shared_ptr<arrow::Schema>,
                                                                        std::shared_ptr<arrow::dataset::FileWriteOptions>, arrow::fs::FileLocator) const override
  {
    return arrow::Status::NotImplemented("The native AO2D reader cannot write files");
  }

  std::shared_ptr<arrow::dataset::FileWriteOptions> DefaultWriteOptions() override
  {
    return std::make_shared<NativeAODFileWriteOptions>(shared_from_this());
  }

  arrow::Result<arrow::RecordBatchGenerator> ScanBatchesAsync(
    const std::shared_ptr<arrow::dataset::ScanOptions>& options,
    const std::shared_ptr<arrow::dataset::FileFragment>& fragment) const override
  {
    assert(options->dataset_schema != nullptr);
    auto datasetSchema = options->dataset_schema;
    auto nativeFragment = std::dynamic_pointer_cast<NativeAODFileFragment>(fragment);
    if (nativeFragment.get() == nullptr) {
      return {arrow::Status::NotImplemented("Not a native AO2D fragment")};
    }

    auto generator = [pool = options->pool, nativeFragment, datasetSchema, &totalCompressedSize = mTotCompressedSize,
                      &totalUncompressedSize = mTotUncompressedSize]() -> arrow::Future<std::shared_ptr<arrow::RecordBatch>> {
      auto* handle = nativeFragment->handle();
      auto const& tree = handle->tree;
      O2_SIGNPOST_ID_FROM_POINTER(tid, native_aod, handle);
      O2_SIGNPOST_START(native_aod, tid, "Generator", "Creating batch for tree %{public}s", tree.name.c_str());
      if (datasetSchema->num_fields() > (int)tree.branches.size()) {
        throw runtime_error_f("One TTree must have all the fields requested in a table");
      }
      // Plan the reads. VLA offsets are decoded now, because they size the
      // values; everything else is decoded later, in the deferred Write().
      int64_t rows = 0;
      auto& ops = nativeFragment->ops();
      ops = o2::aod::native::planColumns(tree, datasetSchema, nativeFragment->decoder(), pool, rows);
      // The pointer to each buffer is an incremental number, indexing ops.
      std::vector<std::shared_ptr<arrow::Buffer>> placeholders;
      for (size_t i = 0; i < ops.size(); ++i) {
        placeholders.push_back(std::make_shared<arrow::Buffer>((uint8_t*)i, ops[i].byteSize));
        O2_SIGNPOST_EVENT_EMIT(native_aod, tid, "Op", "Created op for branch %{public}s with %lli entries, size of the buffer %zu.",
                               ops[i].branch->name.c_str(), (long long)ops[i].branch->entries, ops[i].byteSize);
      }
      auto batch = o2::aod::native::makeBatch(datasetSchema, ops, placeholders, rows);
      totalCompressedSize += tree.zipBytes;
      totalUncompressedSize += tree.totBytes;
      O2_SIGNPOST_END(native_aod, tid, "Generator", "Done creating batch compressed:%zu uncompressed:%zu", totalCompressedSize, totalUncompressedSize);
      return batch;
    };
    return generator;
  }
};

struct NativeAODPluginContext {
  size_t totalCompressedSize = 0;
  size_t totalUncompressedSize = 0;
  std::shared_ptr<NativeAODFileFormat> format = nullptr;
};

RootArrowFactory* createNativeAODFactory()
{
  auto context = new NativeAODPluginContext;
  context->format = std::make_shared<NativeAODFileFormat>(context->totalCompressedSize, context->totalUncompressedSize);
  return new RootArrowFactory{
    .options = [context]() { return context->format->DefaultWriteOptions(); },
    .format = [context]() { return context->format; },
    .deferredOutputStreamer = [](std::shared_ptr<arrow::dataset::FileFragment> fragment, const std::shared_ptr<arrow::ResizableBuffer>& buffer) -> std::shared_ptr<arrow::io::OutputStream> {
      auto nativeFragment = std::dynamic_pointer_cast<NativeAODFileFragment>(fragment);
      return std::make_shared<NativeAODDeferredReadOutputStream>(nativeFragment, buffer);
    }};
}

struct NativeAODObjectReadingImplementation : public RootArrowFactoryPlugin {
  RootArrowFactory* create() override { return createNativeAODFactory(); }
};

// The capability lives here rather than in Framework/Core, because recognising
// a native handle needs NativeAODFileSystem. Being first in the capability
// list, it wins for files opened natively; for any other filesystem it returns
// no handle, and DPL falls through to the RNTuple and TTree capabilities.
struct NativeAODObjectReadingCapability : o2::framework::RootObjectReadingCapabilityPlugin {
  RootObjectReadingCapability* create() override
  {
    return new RootObjectReadingCapability{
      .name = "native-ttree",
      .lfn2objectPath = [](std::string s) { return s; },
      .getHandle = [](std::shared_ptr<arrow::fs::FileSystem> fs, std::string const& path) -> void* {
        auto nativeFS = std::dynamic_pointer_cast<NativeAODFileSystem>(fs);
        if (!nativeFS) {
          return nullptr;
        }
        try {
          return nativeFS->getTreeHandle(path);
        } catch (o2::aod::native::FormatError const& e) {
          // A DF which cannot be parsed: --aod-skip-invalid-reads skips it.
          throw AODReadError(AODReadError::Kind::InvalidTimeframe, e.what());
        }
      },
      .checkSupport = [](char const* className) -> bool { return std::string_view(className) == "TTree"; },
      .accountBytes = [](void* handle, size_t& compressed, size_t& uncompressed) {
        auto* h = (NativeTreeHandle*)handle;
        compressed += h->tree.zipBytes;
        uncompressed += h->tree.totBytes; },
      .factory = []() -> RootArrowFactory& {
        static RootArrowFactory* factory = createNativeAODFactory();
        return *factory;
      }};
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(NativeAODObjectReadingImplementation, RootObjectReadingImplementation);
DEFINE_DPL_PLUGIN_INSTANCE(NativeAODObjectReadingCapability, RootObjectReadingCapability);
DEFINE_DPL_PLUGINS_END
} // namespace o2::framework

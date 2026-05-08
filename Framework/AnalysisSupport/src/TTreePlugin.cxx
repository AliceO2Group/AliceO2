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
#include "Framework/Plugins.h"
#include "Framework/Signpost.h"
#include "Framework/BigEndian.h"
#include <TBufferFile.h>
#include <TBufferIO.h>
#include <arrow/buffer.h>
#include <arrow/dataset/file_base.h>
#include <arrow/extension_type.h>
#include <arrow/memory_pool.h>
#include <arrow/status.h>
#include <arrow/type.h>
#include <arrow/type_fwd.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/array/array_nested.h>
#include <arrow/array/array_primitive.h>
#include <arrow/array/builder_nested.h>
#include <arrow/array/builder_primitive.h>
#include <arrow/array/util.h>
#include <arrow/record_batch.h>
#include <TTree.h>
#include <TBranch.h>
#include <TFile.h>
#include <TLeaf.h>
#include <unistd.h>
#include <cstdint>
#include <memory>
#include <stdexcept>

O2_DECLARE_DYNAMIC_LOG(root_arrow_fs);

namespace o2::framework
{

enum struct ReadOpKind {
  Unknown,
  Offsets,
  Values,
  Booleans,
  VLA
};

struct ReadOps {
  TBranch* branch = nullptr;
  std::shared_ptr<arrow::Buffer> targetBuffer = nullptr;
  int64_t rootBranchEntries = 0;
  size_t typeSize = 0;
  size_t listSize = 0;
  // If this is an offset reading op, keep track of the actual
  // range for the offsets, not only how many VLAs are there.
  int64_t offsetCount = 0;
  ReadOpKind kind = ReadOpKind::Unknown;
};

/// An OutputStream which does the reading of the input buffers directly
/// on writing, if needed. Each deferred operation is encoded in the source
/// buffer by an incremental number which can be used to lookup in the @a ops
/// vector the operation to perform.
class TTreeDeferredReadOutputStream : public arrow::io::OutputStream
{
 public:
  explicit TTreeDeferredReadOutputStream(std::vector<ReadOps>& ops,
                                         const std::shared_ptr<arrow::ResizableBuffer>& buffer);

  /// \brief Create in-memory output stream with indicated capacity using a
  /// memory pool
  /// \param[in] initial_capacity the initial allocated internal capacity of
  /// the OutputStream
  /// \param[in,out] pool a MemoryPool to use for allocations
  /// \return the created stream
  static arrow::Result<std::shared_ptr<TTreeDeferredReadOutputStream>> Create(
    std::vector<ReadOps>& ops,
    int64_t initial_capacity = 4096,
    arrow::MemoryPool* pool = arrow::default_memory_pool());

  // By the time we call the destructor, the contents
  // of the buffer are already moved to fairmq
  // for being sent.
  ~TTreeDeferredReadOutputStream() override = default;

  // Implement the OutputStream interface

  /// Close the stream, preserving the buffer (retrieve it with Finish()).
  arrow::Status Close() override;
  [[nodiscard]] bool closed() const override;
  [[nodiscard]] arrow::Result<int64_t> Tell() const override;
  arrow::Status Write(const void* data, int64_t nbytes) override;

  /// \cond FALSE
  using OutputStream::Write;
  /// \endcond

  /// Close the stream and return the buffer
  arrow::Result<std::shared_ptr<arrow::Buffer>> Finish();

  /// \brief Initialize state of OutputStream with newly allocated memory and
  /// set position to 0
  /// \param[in] initial_capacity the starting allocated capacity
  /// \param[in,out] pool the memory pool to use for allocations
  /// \return Status
  arrow::Status Reset(std::vector<ReadOps> ops,
                      int64_t initial_capacity, arrow::MemoryPool* pool);

  [[nodiscard]] int64_t capacity() const { return capacity_; }

 private:
  TTreeDeferredReadOutputStream();
  std::vector<ReadOps> ops_;

  // Ensures there is sufficient space available to write nbytes
  arrow::Status Reserve(int64_t nbytes);

  std::shared_ptr<arrow::ResizableBuffer> buffer_;
  bool is_open_;
  int64_t capacity_;
  int64_t position_;
  uint8_t* mutable_data_;
};

static constexpr int64_t kBufferMinimumSize = 256;

TTreeDeferredReadOutputStream::TTreeDeferredReadOutputStream()
  : is_open_(false), capacity_(0), position_(0), mutable_data_(nullptr) {}

TTreeDeferredReadOutputStream::TTreeDeferredReadOutputStream(std::vector<ReadOps>& ops,
                                                             const std::shared_ptr<arrow::ResizableBuffer>& buffer)
  : ops_(ops),
    buffer_(buffer),
    is_open_(true),
    capacity_(buffer->size()),
    position_(0),
    mutable_data_(buffer->mutable_data()) {}

arrow::Result<std::shared_ptr<TTreeDeferredReadOutputStream>> TTreeDeferredReadOutputStream::Create(
  std::vector<ReadOps>& ops,
  int64_t initial_capacity, arrow::MemoryPool* pool)
{
  // ctor is private, so cannot use make_shared
  auto ptr = std::shared_ptr<TTreeDeferredReadOutputStream>(new TTreeDeferredReadOutputStream);
  RETURN_NOT_OK(ptr->Reset(ops, initial_capacity, pool));
  return ptr;
}

arrow::Status TTreeDeferredReadOutputStream::Reset(std::vector<ReadOps> ops,
                                                   int64_t initial_capacity, arrow::MemoryPool* pool)
{
  ARROW_ASSIGN_OR_RAISE(buffer_, AllocateResizableBuffer(initial_capacity, pool));
  ops_ = ops;
  is_open_ = true;
  capacity_ = initial_capacity;
  position_ = 0;
  mutable_data_ = buffer_->mutable_data();
  return arrow::Status::OK();
}

arrow::Status TTreeDeferredReadOutputStream::Close()
{
  if (is_open_) {
    is_open_ = false;
    if (position_ < capacity_) {
      RETURN_NOT_OK(buffer_->Resize(position_, false));
    }
  }
  return arrow::Status::OK();
}

bool TTreeDeferredReadOutputStream::closed() const { return !is_open_; }

arrow::Result<std::shared_ptr<arrow::Buffer>> TTreeDeferredReadOutputStream::Finish()
{
  RETURN_NOT_OK(Close());
  buffer_->ZeroPadding();
  is_open_ = false;
  return std::move(buffer_);
}

arrow::Result<int64_t> TTreeDeferredReadOutputStream::Tell() const { return position_; }

auto readValues = [](uint8_t* target, ReadOps& op, TBufferFile& rootBuffer) {
  int readEntries = 0;
  rootBuffer.Reset();
  while (readEntries < op.rootBranchEntries) {
    auto readLast = op.branch->GetBulkRead().GetEntriesSerialized(readEntries, rootBuffer);
    if (readLast < 0) {
      throw runtime_error_f("Error while reading branch %s starting from %zu.", op.branch->GetName(), readEntries);
    }
    int size = readLast * op.listSize;
    readEntries += readLast;
    bigEndianCopy(target, rootBuffer.GetCurrent(), size, op.typeSize);
    target += (ptrdiff_t)(size * op.typeSize);
  }
};

auto readBoolValues = [](uint8_t* target, ReadOps& op, TBufferFile& rootBuffer) {
  int readEntries = 0;
  rootBuffer.Reset();
  // Set to 0
  memset(target, 0, op.targetBuffer->size());
  int readLast = 0;
  while (readEntries < op.rootBranchEntries) {
    auto beginValue = readEntries;
    readLast = op.branch->GetBulkRead().GetBulkEntries(readEntries, rootBuffer);
    int size = readLast * op.listSize;
    readEntries += readLast;
    for (int i = beginValue; i < beginValue + size; ++i) {
      auto value = static_cast<uint8_t>(rootBuffer.GetCurrent()[i - beginValue] << (i % 8));
      target[i / 8] |= value;
    }
  }
};

auto readVLAValues = [](uint8_t* target, ReadOps& op, ReadOps const& offsetOp, TBufferFile& rootBuffer) {
  int readEntries = 0;
  auto* tPtrOffset = reinterpret_cast<const int*>(offsetOp.targetBuffer->data());
  std::span<int const> const offsets{tPtrOffset, tPtrOffset + offsetOp.rootBranchEntries + 1};

  rootBuffer.Reset();
  while (readEntries < op.rootBranchEntries) {
    auto readLast = op.branch->GetBulkRead().GetEntriesSerialized(readEntries, rootBuffer);
    int size = offsets[readEntries + readLast] - offsets[readEntries];
    readEntries += readLast;
    bigEndianCopy(target, rootBuffer.GetCurrent(), size, op.typeSize);
    target += (ptrdiff_t)(size * op.typeSize);
  }
};

TBufferFile& rootBuffer()
{
  // FIXME: we will need more than one once we have multithreaded reading.
  static TBufferFile rootBuffer{TBuffer::EMode::kWrite, 4 * 1024 * 1024};
  return rootBuffer;
}

arrow::Status TTreeDeferredReadOutputStream::Write(const void* data, int64_t nbytes)
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
  // This is a real address which needs to be copied. Do it!
  auto ref = (int64_t)data;
  if (ref >= ops_.size()) {
    memcpy(mutable_data_ + position_, data, nbytes);
    position_ += nbytes;
    return arrow::Status::OK();
  }
  auto& op = ops_[ref];

  switch (op.kind) {
    // Offsets need to be read in advance because we need to know
    // how many elements are there in total (since TTree does not allow discovering such informantion)
    case ReadOpKind::Offsets:
      break;
    case ReadOpKind::Values:
      readValues(mutable_data_ + position_, op, rootBuffer());
      break;
    case ReadOpKind::VLA:
      readVLAValues(mutable_data_ + position_, op, ops_[ref - 1], rootBuffer());
      break;
    case ReadOpKind::Booleans:
      readBoolValues(mutable_data_ + position_, op, rootBuffer());
      break;
    case ReadOpKind::Unknown:
      throw runtime_error("Unknown Op");
  }
  op.branch->SetStatus(false);
  op.branch->DropBaskets("all");
  op.branch->Reset();
  op.branch->GetTransientBuffer(0)->Expand(0);

  position_ += nbytes;
  return arrow::Status::OK();
}

arrow::Status TTreeDeferredReadOutputStream::Reserve(int64_t nbytes)
{
  // Always overallocate by doubling.  It seems that it is a better growth
  // strategy, at least for memory_benchmark.cc.
  // This may be because it helps match the allocator's allocation buckets
  // more exactly.  Or perhaps it hits a sweet spot in jemalloc.
  int64_t new_capacity = std::max(kBufferMinimumSize, capacity_);
  new_capacity = position_ + nbytes;
  if (new_capacity > capacity_) {
    RETURN_NOT_OK(buffer_->Resize(new_capacity));
    capacity_ = new_capacity;
    mutable_data_ = buffer_->mutable_data();
  }
  return arrow::Status::OK();
}

class TTreeFileWriteOptions : public arrow::dataset::FileWriteOptions
{
 public:
  TTreeFileWriteOptions(std::shared_ptr<arrow::dataset::FileFormat> format)
    : FileWriteOptions(format)
  {
  }
};

// A filesystem which allows me to get a TTree
class TTreeFileSystem : public VirtualRootFileSystemBase
{
 public:
  ~TTreeFileSystem() override;

  arrow::Result<std::shared_ptr<arrow::io::OutputStream>> OpenOutputStream(
    const std::string& path,
    const std::shared_ptr<const arrow::KeyValueMetadata>& metadata) override;

  virtual std::unique_ptr<TTree>& GetTree(arrow::dataset::FileSource source) = 0;
};

class TTreeFileFormat : public arrow::dataset::FileFormat
{
  size_t& mTotCompressedSize;
  size_t& mTotUncompressedSize;

 public:
  TTreeFileFormat(size_t& totalCompressedSize, size_t& totalUncompressedSize)
    : FileFormat({}),
      mTotCompressedSize(totalCompressedSize),
      mTotUncompressedSize(totalUncompressedSize)
  {
  }

  ~TTreeFileFormat() override = default;

  std::string type_name() const override
  {
    return "ttree";
  }

  bool Equals(const FileFormat& other) const override
  {
    return other.type_name() == this->type_name();
  }

  arrow::Result<bool> IsSupported(const arrow::dataset::FileSource& source) const override
  {
    auto fs = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(source.filesystem());
    if (!fs) {
      return false;
    }
    return fs->CheckSupport(source);
  }

  arrow::Result<std::shared_ptr<arrow::Schema>> Inspect(const arrow::dataset::FileSource& source) const override;
  /// \brief Create a FileFragment for a FileSource.
  arrow::Result<std::shared_ptr<arrow::dataset::FileFragment>> MakeFragment(
    arrow::dataset::FileSource source, arrow::compute::Expression partition_expression,
    std::shared_ptr<arrow::Schema> physical_schema) override;

  arrow::Result<std::shared_ptr<arrow::dataset::FileWriter>> MakeWriter(std::shared_ptr<arrow::io::OutputStream> destination, std::shared_ptr<arrow::Schema> schema, std::shared_ptr<arrow::dataset::FileWriteOptions> options, arrow::fs::FileLocator destination_locator) const override;

  std::shared_ptr<arrow::dataset::FileWriteOptions> DefaultWriteOptions() override;

  arrow::Result<arrow::RecordBatchGenerator> ScanBatchesAsync(
    const std::shared_ptr<arrow::dataset::ScanOptions>& options,
    const std::shared_ptr<arrow::dataset::FileFragment>& fragment) const override;
};

class SingleTreeFileSystem : public TTreeFileSystem
{
 public:
  SingleTreeFileSystem(TTree* tree)
    : TTreeFileSystem(),
      mTree(tree)
  {
  }

  arrow::Result<arrow::fs::FileInfo> GetFileInfo(std::string const& path) override;

  std::string type_name() const override
  {
    return "ttree";
  }

  std::shared_ptr<RootObjectHandler> GetObjectHandler(arrow::dataset::FileSource source) override
  {
    return std::make_shared<RootObjectHandler>((void*)mTree.get(), std::make_shared<TTreeFileFormat>(mTotCompressedSize, mTotUncompressedSize));
  }

  std::unique_ptr<TTree>& GetTree(arrow::dataset::FileSource) override
  {
    // Simply return the only TTree we have
    return mTree;
  }

 private:
  size_t mTotUncompressedSize;
  size_t mTotCompressedSize;
  std::unique_ptr<TTree> mTree;
};

arrow::Result<arrow::fs::FileInfo> SingleTreeFileSystem::GetFileInfo(std::string const& path)
{
  arrow::dataset::FileSource source(path, shared_from_this());
  arrow::fs::FileInfo result;
  result.set_path(path);
  result.set_type(arrow::fs::FileType::File);
  return result;
}

// A fragment which holds a tree
class TTreeFileFragment : public arrow::dataset::FileFragment
{
 public:
  TTreeFileFragment(arrow::dataset::FileSource source,
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
      throw runtime_error_f("Cannot read source %s with format %s to pupulate a TTreeFileFragment.",
                            source.path().c_str(), objectHandler->format->type_name().c_str());
    };
    mTree = objectHandler->GetObjectAsOwner<TTree>();
  }

  TTree* GetTree()
  {
    return mTree.get();
  }

  std::vector<ReadOps>& ops()
  {
    return mOps;
  }

  /// The pointer to each allocation is an incremental number, indexing a collection to track
  /// the size of each allocation.
  std::shared_ptr<arrow::Buffer> GetPlaceholderForOp(size_t size)
  {
    return std::make_shared<arrow::Buffer>((uint8_t*)(mOps.size() - 1), size);
  }

 private:
  std::unique_ptr<TTree> mTree;
  std::vector<ReadOps> mOps;
};

// An arrow outputstream which allows to write to a TTree. Eventually
// with a prefix for the branches.
class TTreeOutputStream : public arrow::io::OutputStream
{
 public:
  // Using a pointer means that the tree itself is owned by another
  // class
  TTreeOutputStream(TTree*, std::string branchPrefix);

  arrow::Status Close() override;

  arrow::Result<int64_t> Tell() const override;

  arrow::Status Write(const void* data, int64_t nbytes) override;

  bool closed() const override;

  TBranch* CreateBranch(char const* branchName, char const* sizeBranch);

  TTree* GetTree()
  {
    return mTree;
  }

 private:
  TTree* mTree;
  std::string mBranchPrefix;
};

// An arrow outputstream which allows to write to a ttree
// @a branch prefix is to be used to identify a set of branches which all belong to
// the same table.
TTreeOutputStream::TTreeOutputStream(TTree* f, std::string branchPrefix)
  : mTree(f),
    mBranchPrefix(std::move(branchPrefix))
{
}

arrow::Status TTreeOutputStream::Close()
{
  if (mTree->GetCurrentFile() == nullptr) {
    return arrow::Status::Invalid("Cannot close a tree not attached to a file");
  }
  mTree->GetCurrentFile()->Close();
  return arrow::Status::OK();
}

arrow::Result<int64_t> TTreeOutputStream::Tell() const
{
  return arrow::Result<int64_t>(arrow::Status::NotImplemented("Cannot move"));
}

arrow::Status TTreeOutputStream::Write(const void* data, int64_t nbytes)
{
  return arrow::Status::NotImplemented("Cannot write raw bytes to a TTree");
}

bool TTreeOutputStream::closed() const
{
  // A standalone tree is never closed.
  if (mTree->GetCurrentFile() == nullptr) {
    return false;
  }
  return mTree->GetCurrentFile()->IsOpen() == false;
}

TBranch* TTreeOutputStream::CreateBranch(char const* branchName, char const* sizeBranch)
{
  if (mBranchPrefix.empty() == true) {
    return mTree->Branch(branchName, (char*)nullptr, sizeBranch);
  }
  return mTree->Branch((mBranchPrefix + "/" + branchName).c_str(), (char*)nullptr, (mBranchPrefix + sizeBranch).c_str());
}

struct TTreePluginContext {
  size_t totalCompressedSize = 0;
  size_t totalUncompressedSize = 0;
  std::shared_ptr<o2::framework::TTreeFileFormat> format = nullptr;
};

struct TTreeObjectReadingImplementation : public RootArrowFactoryPlugin {
  RootArrowFactory* create() override
  {
    auto context = new TTreePluginContext;
    context->format = std::make_shared<o2::framework::TTreeFileFormat>(context->totalCompressedSize, context->totalUncompressedSize);
    return new RootArrowFactory{
      .options = [context]() { return context->format->DefaultWriteOptions(); },
      .format = [context]() { return context->format; },
      .deferredOutputStreamer = [](std::shared_ptr<arrow::dataset::FileFragment> fragment, const std::shared_ptr<arrow::ResizableBuffer>& buffer) -> std::shared_ptr<arrow::io::OutputStream> {
        auto treeFragment = std::dynamic_pointer_cast<TTreeFileFragment>(fragment);
        return std::make_shared<TTreeDeferredReadOutputStream>(treeFragment->ops(), buffer);
      }};
  }
};

struct BranchFieldMapping {
  int mainBranchIdx;
  int vlaIdx;
  int datasetFieldIdx;
};

auto readOffsets = [](ReadOps& op, TBufferFile& rootBuffer) {
  uint32_t offset = 0;
  std::span<int> offsets;
  int readEntries = 0;
  int count = 0;
  auto* tPtrOffset = reinterpret_cast<int*>(op.targetBuffer->mutable_data());
  offsets = std::span<int>{tPtrOffset, tPtrOffset + op.rootBranchEntries + 1};

  // read sizes first
  rootBuffer.Reset();
  while (readEntries < op.rootBranchEntries) {
    auto readLast = op.branch->GetBulkRead().GetEntriesSerialized(readEntries, rootBuffer);
    if (readLast == -1) {
      throw runtime_error_f("Unable to read from branch %s.", op.branch->GetName());
    }
    readEntries += readLast;
    for (auto i = 0; i < readLast; ++i) {
      offsets[count++] = (int)offset;
      uint32_t raw = reinterpret_cast<uint32_t*>(rootBuffer.GetCurrent())[i];
      offset += (std::endian::native == std::endian::little) ? __builtin_bswap32(raw) : raw;
    }
  }
  offsets[count] = (int)offset;
  op.offsetCount = offset;
};

arrow::Result<arrow::RecordBatchGenerator> TTreeFileFormat::ScanBatchesAsync(
  const std::shared_ptr<arrow::dataset::ScanOptions>& options,
  const std::shared_ptr<arrow::dataset::FileFragment>& fragment) const
{
  assert(options->dataset_schema != nullptr);
  // This is the schema we want to read
  auto dataset_schema = options->dataset_schema;
  auto treeFragment = std::dynamic_pointer_cast<TTreeFileFragment>(fragment);
  if (treeFragment.get() == nullptr) {
    return {arrow::Status::NotImplemented("Not a ttree fragment")};
  }

  auto generator = [pool = options->pool, treeFragment, dataset_schema, &totalCompressedSize = mTotCompressedSize,
                    &totalUncompressedSize = mTotUncompressedSize]() -> arrow::Future<std::shared_ptr<arrow::RecordBatch>> {
    O2_SIGNPOST_ID_FROM_POINTER(tid, root_arrow_fs, treeFragment->GetTree());
    O2_SIGNPOST_START(root_arrow_fs, tid, "Generator", "Creating batch for tree %{public}s", treeFragment->GetTree()->GetName());
    std::vector<std::shared_ptr<arrow::Array>> columns;
    std::vector<std::shared_ptr<arrow::Field>> fields = dataset_schema->fields();
    auto physical_schema = *treeFragment->ReadPhysicalSchema();

    if (dataset_schema->num_fields() > physical_schema->num_fields()) {
      throw runtime_error_f("One TTree must have all the fields requested in a table");
    }

    // Register physical fields into the cache
    std::vector<BranchFieldMapping> mappings;

    // We need to count the number of readops to avoid moving the vector.
    int opsCount = 0;
    for (int fi = 0; fi < dataset_schema->num_fields(); ++fi) {
      auto dataset_field = dataset_schema->field(fi);
      // This is needed because for now the dataset_field
      // is actually the schema of the ttree
      O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Generator", "Processing dataset field %{public}s.", dataset_field->name().c_str());
      int physicalFieldIdx = physical_schema->GetFieldIndex(dataset_field->name());

      if (physicalFieldIdx < 0) {
        throw runtime_error_f("Cannot find physical field associated to %s. Possible fields: %s",
                              dataset_field->name().c_str(), physical_schema->ToString().c_str());
      }
      if (physicalFieldIdx > 0 && physical_schema->field(physicalFieldIdx - 1)->name().ends_with("_size")) {
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Generator", "Field %{public}s has sizes in %{public}s.", dataset_field->name().c_str(),
                               physical_schema->field(physicalFieldIdx - 1)->name().c_str());
        mappings.push_back({physicalFieldIdx, physicalFieldIdx - 1, fi});
        opsCount += 2;
      } else {
        if (physicalFieldIdx > 0) {
          O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Generator", "Field %{public}s previous field is %{public}s.", dataset_field->name().c_str(),
                                 physical_schema->field(physicalFieldIdx - 1)->name().c_str());
        }
        mappings.push_back({physicalFieldIdx, -1, fi});
        opsCount++;
      }
    }

    auto* tree = treeFragment->GetTree();
    auto branches = tree->GetListOfBranches();
    size_t totalTreeSize = 0;
    std::vector<TBranch*> selectedBranches;
    for (auto& mapping : mappings) {
      selectedBranches.push_back((TBranch*)branches->At(mapping.mainBranchIdx));
      O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Generator", "Adding branch %{public}s to stream.", selectedBranches.back()->GetName());
      totalTreeSize += selectedBranches.back()->GetTotalSize();
      if (mapping.vlaIdx != -1) {
        selectedBranches.push_back((TBranch*)branches->At(mapping.vlaIdx));
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Generator", "Adding branch %{public}s to stream.", selectedBranches.back()->GetName());
        totalTreeSize += selectedBranches.back()->GetTotalSize();
      }
    }

    size_t cacheSize = std::max(std::min(totalTreeSize, 25000000UL), 1000000UL);
    O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Generator", "Resizing cache to %zu.", cacheSize);
    tree->SetCacheSize(cacheSize);
    for (auto* branch : selectedBranches) {
      tree->AddBranchToCache(branch, false);
    }
    tree->StopCacheLearningPhase();

    // Intermediate buffer to bulk read. Two for now
    std::vector<ReadOps>& ops = treeFragment->ops();
    ops.clear();
    ops.reserve(opsCount);
    for (size_t mi = 0; mi < mappings.size(); ++mi) {
      BranchFieldMapping mapping = mappings[mi];
      // The field actually on disk
      auto datasetField = dataset_schema->field(mapping.datasetFieldIdx);
      auto physicalField = physical_schema->field(mapping.mainBranchIdx);

      if (mapping.vlaIdx != -1) {
        auto* branch = (TBranch*)branches->At(mapping.vlaIdx);
        ops.emplace_back(ReadOps{
          .branch = branch,
          .rootBranchEntries = branch->GetEntries(),
          .typeSize = 4,
          .listSize = 1,
          .kind = ReadOpKind::Offsets,
        });
        auto& op = ops.back();
        ARROW_ASSIGN_OR_RAISE(op.targetBuffer, arrow::AllocateBuffer((op.rootBranchEntries + 1) * op.typeSize, pool));
        // Offsets need to be read immediately to know how many values are there
        readOffsets(op, rootBuffer());
      }
      ops.push_back({});
      auto& valueOp = ops.back();
      valueOp.branch = (TBranch*)branches->At(mapping.mainBranchIdx);
      valueOp.rootBranchEntries = valueOp.branch->GetEntries();
      // In case this is a vla, we set the offsetCount as totalEntries
      // In case we read booleans we need a special coversion from bytes to bits.
      auto listType = std::dynamic_pointer_cast<arrow::FixedSizeListType>(datasetField->type());
      valueOp.typeSize = physicalField->type()->byte_width();
      // Notice how we are not (yet) allocating buffers at this point. We merely
      // create placeholders to subsequently fill.
      if ((datasetField->type() == arrow::boolean())) {
        valueOp.kind = ReadOpKind::Booleans;
        valueOp.listSize = 1;
        valueOp.targetBuffer = treeFragment->GetPlaceholderForOp((valueOp.rootBranchEntries + 7) / 8);
      } else if (listType && datasetField->type()->field(0)->type() == arrow::boolean()) {
        valueOp.typeSize = physicalField->type()->field(0)->type()->byte_width();
        valueOp.listSize = listType->list_size();
        valueOp.kind = ReadOpKind::Booleans;
        valueOp.targetBuffer = treeFragment->GetPlaceholderForOp((valueOp.rootBranchEntries * valueOp.listSize) / 8 + 1);
      } else if (mapping.vlaIdx != -1) {
        valueOp.typeSize = physicalField->type()->field(0)->type()->byte_width();
        valueOp.listSize = -1;
        // -1 is the current one, -2 is the one with for the offsets
        valueOp.kind = ReadOpKind::VLA;
        valueOp.targetBuffer = treeFragment->GetPlaceholderForOp(ops[ops.size() - 2].offsetCount * valueOp.typeSize);
      } else if (listType) {
        valueOp.kind = ReadOpKind::Values;
        valueOp.listSize = listType->list_size();
        valueOp.typeSize = physicalField->type()->field(0)->type()->byte_width();
        valueOp.targetBuffer = treeFragment->GetPlaceholderForOp(valueOp.rootBranchEntries * valueOp.typeSize * valueOp.listSize);
      } else {
        valueOp.typeSize = physicalField->type()->byte_width();
        valueOp.kind = ReadOpKind::Values;
        valueOp.listSize = 1;
        valueOp.targetBuffer = treeFragment->GetPlaceholderForOp(valueOp.rootBranchEntries * valueOp.typeSize);
      }
      arrow::Status status;
      std::shared_ptr<arrow::Array> array;

      if (listType) {
        auto vdata = std::make_shared<arrow::ArrayData>(datasetField->type()->field(0)->type(), valueOp.rootBranchEntries * valueOp.listSize,
                                                        std::vector<std::shared_ptr<arrow::Buffer>>{nullptr, valueOp.targetBuffer});
        array = std::make_shared<arrow::FixedSizeListArray>(datasetField->type(), valueOp.rootBranchEntries, arrow::MakeArray(vdata));
        // This is a vla, there is also an offset op
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Op", "Created op for branch %{public}s with %lli entries, size of the buffer %lli.",
                               valueOp.branch->GetName(),
                               valueOp.rootBranchEntries,
                               valueOp.targetBuffer->size());
      } else if (mapping.vlaIdx != -1) {
        auto& offsetOp = ops[ops.size() - 2];
        auto vdata = std::make_shared<arrow::ArrayData>(datasetField->type()->field(0)->type(), offsetOp.offsetCount,
                                                        std::vector<std::shared_ptr<arrow::Buffer>>{nullptr, valueOp.targetBuffer});
        // We have pushed an offset op if this was the case.
        array = std::make_shared<arrow::ListArray>(datasetField->type(), offsetOp.rootBranchEntries, offsetOp.targetBuffer, arrow::MakeArray(vdata));
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Op", "Created op for branch %{public}s with %lli entries, size of the buffer %lli.",
                               offsetOp.branch->GetName(), offsetOp.rootBranchEntries, offsetOp.targetBuffer->size());
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Op", "Created op for branch %{public}s with %lli entries, size of the buffer %lli.",
                               valueOp.branch->GetName(),
                               offsetOp.offsetCount,
                               valueOp.targetBuffer->size());
      } else {
        auto data = std::make_shared<arrow::ArrayData>(datasetField->type(), valueOp.rootBranchEntries,
                                                       std::vector<std::shared_ptr<arrow::Buffer>>{nullptr, valueOp.targetBuffer});
        array = arrow::MakeArray(data);
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, tid, "Op", "Created op for branch %{public}s with %lli entries, size of the buffer %lli.",
                               valueOp.branch->GetName(),
                               valueOp.rootBranchEntries,
                               valueOp.targetBuffer->size());
      }

      columns.push_back(array);
    }

    // Do the actual filling of the buffers. This happens after we have created the whole structure
    // so that we can read directly in shared memory.
    int64_t rows = -1;
    for (size_t i = 0; i < ops.size(); ++i) {
      auto& op = ops[i];
      if (rows == -1 && op.kind != ReadOpKind::VLA) {
        rows = op.rootBranchEntries;
      }
      if (rows == -1 && op.kind == ReadOpKind::VLA) {
        auto& offsetOp = ops[i - 1];
        rows = offsetOp.rootBranchEntries;
      }
      if (op.kind != ReadOpKind::VLA && rows != op.rootBranchEntries) {
        throw runtime_error_f("Unmatching number of rows for branch %s. Expected %lli, found %lli", op.branch->GetName(), rows, op.rootBranchEntries);
      }
      if (op.kind == ReadOpKind::VLA && rows != ops[i - 1].rootBranchEntries) {
        throw runtime_error_f("Unmatching number of rows for branch %s. Expected %lli, found %lli", op.branch->GetName(), rows, ops[i - 1].offsetCount);
      }
    }

    auto batch = arrow::RecordBatch::Make(dataset_schema, rows, columns);
    totalCompressedSize += tree->GetZipBytes();
    totalUncompressedSize += tree->GetTotBytes();
    O2_SIGNPOST_END(root_arrow_fs, tid, "Generator", "Done creating batch compressed:%zu uncompressed:%zu", totalCompressedSize, totalUncompressedSize);
    return batch;
  };
  return generator;
}

char const* rootSuffixFromArrow(arrow::Type::type id)
{
  switch (id) {
    case arrow::Type::BOOL:
      return "/O";
    case arrow::Type::UINT8:
      return "/b";
    case arrow::Type::UINT16:
      return "/s";
    case arrow::Type::UINT32:
      return "/i";
    case arrow::Type::UINT64:
      return "/l";
    case arrow::Type::INT8:
      return "/B";
    case arrow::Type::INT16:
      return "/S";
    case arrow::Type::INT32:
      return "/I";
    case arrow::Type::INT64:
      return "/L";
    case arrow::Type::FLOAT:
      return "/F";
    case arrow::Type::DOUBLE:
      return "/D";
    default:
      throw runtime_error("Unsupported arrow column type");
  }
}

arrow::Result<std::shared_ptr<arrow::io::OutputStream>> TTreeFileSystem::OpenOutputStream(
  const std::string& path,
  const std::shared_ptr<const arrow::KeyValueMetadata>& metadata)
{
  arrow::dataset::FileSource source{path, shared_from_this()};
  auto prefix = metadata->Get("branch_prefix");
  if (prefix.ok()) {
    return std::make_shared<TTreeOutputStream>(GetTree(source).get(), *prefix);
  }
  return std::make_shared<TTreeOutputStream>(GetTree(source).get(), "");
}

namespace
{
struct BranchInfo {
  std::string name;
  TBranch* ptr;
  bool mVLA;
};
} // namespace

auto arrowTypeFromROOT(EDataType type, int size)
{
  auto typeGenerator = [](std::shared_ptr<arrow::DataType> const& type, int size) -> std::shared_ptr<arrow::DataType> {
    switch (size) {
      case -1:
        return arrow::list(type);
      case 1:
        return std::move(type);
      default:
        return arrow::fixed_size_list(type, size);
    }
  };

  switch (type) {
    case EDataType::kBool_t:
      return typeGenerator(arrow::boolean(), size);
    case EDataType::kUChar_t:
      return typeGenerator(arrow::uint8(), size);
    case EDataType::kUShort_t:
      return typeGenerator(arrow::uint16(), size);
    case EDataType::kUInt_t:
      return typeGenerator(arrow::uint32(), size);
    case EDataType::kULong64_t:
      return typeGenerator(arrow::uint64(), size);
    case EDataType::kChar_t:
      return typeGenerator(arrow::int8(), size);
    case EDataType::kShort_t:
      return typeGenerator(arrow::int16(), size);
    case EDataType::kInt_t:
      return typeGenerator(arrow::int32(), size);
    case EDataType::kLong64_t:
      return typeGenerator(arrow::int64(), size);
    case EDataType::kFloat_t:
      return typeGenerator(arrow::float32(), size);
    case EDataType::kDouble_t:
      return typeGenerator(arrow::float64(), size);
    default:
      throw o2::framework::runtime_error_f("Unsupported branch type: %d", static_cast<int>(type));
  }
}

// This is a datatype for branches which implies
struct RootTransientIndexType : arrow::ExtensionType {
};

arrow::Result<std::shared_ptr<arrow::Schema>> TTreeFileFormat::Inspect(const arrow::dataset::FileSource& source) const
{
  auto fs = std::dynamic_pointer_cast<VirtualRootFileSystemBase>(source.filesystem());

  if (!fs.get()) {
    throw runtime_error_f("Unknown filesystem %s\n", source.filesystem()->type_name().c_str());
  }
  auto objectHandler = fs->GetObjectHandler(source);

  if (!objectHandler->format->Equals(*this)) {
    throw runtime_error_f("Unknown filesystem %s\n", source.filesystem()->type_name().c_str());
  }

  // Notice that we abuse of the API here and do not release the TTree,
  // so that it's still managed by ROOT.
  auto tree = objectHandler->GetObjectAsOwner<TTree>().release();

  auto branches = tree->GetListOfBranches();
  auto n = branches->GetEntries();

  std::vector<std::shared_ptr<arrow::Field>> fields;

  bool prevIsSize = false;
  for (auto i = 0; i < n; ++i) {
    auto branch = static_cast<TBranch*>(branches->At(i));
    std::string name = branch->GetName();
    if (prevIsSize && fields.back()->name() != name + "_size") {
      throw runtime_error_f("Unexpected layout for VLA container %s.", branch->GetName());
    }

    if (name.ends_with("_size")) {
      fields.emplace_back(std::make_shared<arrow::Field>(name, arrow::int32()));
      prevIsSize = true;
    } else {
      static TClass* cls;
      EDataType type;
      branch->GetExpectedType(cls, type);

      if (prevIsSize) {
        fields.emplace_back(std::make_shared<arrow::Field>(name, arrowTypeFromROOT(type, -1)));
      } else {
        auto listSize = static_cast<TLeaf*>(branch->GetListOfLeaves()->At(0))->GetLenStatic();
        fields.emplace_back(std::make_shared<arrow::Field>(name, arrowTypeFromROOT(type, listSize)));
      }
      prevIsSize = false;
    }
  }

  if (fields.back()->name().ends_with("_size")) {
    throw runtime_error_f("Missing values for VLA indices %s.", fields.back()->name().c_str());
  }
  return std::make_shared<arrow::Schema>(fields);
}

/// \brief Create a FileFragment for a FileSource.
arrow::Result<std::shared_ptr<arrow::dataset::FileFragment>> TTreeFileFormat::MakeFragment(
  arrow::dataset::FileSource source, arrow::compute::Expression partition_expression,
  std::shared_ptr<arrow::Schema> physical_schema)
{

  return std::make_shared<TTreeFileFragment>(source, std::dynamic_pointer_cast<arrow::dataset::FileFormat>(shared_from_this()),
                                             std::move(partition_expression),
                                             physical_schema);
}

class TTreeFileWriter : public arrow::dataset::FileWriter
{
  std::vector<TBranch*> branches;
  std::vector<TBranch*> sizesBranches;
  std::vector<std::shared_ptr<arrow::Array>> valueArrays;
  std::vector<std::shared_ptr<arrow::Array>> sizeArrays;
  std::vector<std::shared_ptr<arrow::DataType>> valueTypes;

  std::vector<int64_t> valuesIdealBasketSize;
  std::vector<int64_t> sizeIdealBasketSize;

  std::vector<int64_t> typeSizes;
  std::vector<int64_t> listSizes;
  bool firstBasket = true;

  // This is to create a batsket size according to the first batch.
  void finaliseBasketSize(std::shared_ptr<arrow::RecordBatch> firstBatch)
  {
    O2_SIGNPOST_ID_FROM_POINTER(sid, root_arrow_fs, this);
    O2_SIGNPOST_START(root_arrow_fs, sid, "finaliseBasketSize", "First batch with %lli rows received and %zu columns",
                      firstBatch->num_rows(), firstBatch->columns().size());
    for (size_t i = 0; i < branches.size(); i++) {
      auto* branch = branches[i];
      auto* sizeBranch = sizesBranches[i];

      int valueSize = valueTypes[i]->byte_width();
      if (listSizes[i] == 1) {
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, sid, "finaliseBasketSize", "Branch %s exists and uses %d bytes per entry for %lli entries.",
                               branch->GetName(), valueSize, firstBatch->num_rows());
        assert(sizeBranch == nullptr);
        branch->SetBasketSize(1024 + firstBatch->num_rows() * valueSize);
      } else if (listSizes[i] == -1) {
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, sid, "finaliseBasketSize", "Branch %s exists and uses %d bytes per entry.",
                               branch->GetName(), valueSize);
        // This should probably lookup the
        auto column = firstBatch->GetColumnByName(schema_->field(i)->name());
        auto list = std::static_pointer_cast<arrow::ListArray>(column);
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, sid, "finaliseBasketSize", "Branch %s needed. Associated size branch %s and there are %lli entries of size %d in that list.",
                               branch->GetName(), sizeBranch->GetName(), list->length(), valueSize);
        branch->SetBasketSize(1024 + firstBatch->num_rows() * valueSize * list->length());
        sizeBranch->SetBasketSize(1024 + firstBatch->num_rows() * 4);
      } else {
        O2_SIGNPOST_EVENT_EMIT(root_arrow_fs, sid, "finaliseBasketSize", "Branch %s needed. There are %lli entries per array of size %d in that list.",
                               branch->GetName(), listSizes[i], valueSize);
        assert(sizeBranch == nullptr);
        branch->SetBasketSize(1024 + firstBatch->num_rows() * valueSize * listSizes[i]);
      }

      auto field = firstBatch->schema()->field(i);
      if (field->name().starts_with("fIndexArray")) {
        // One int per array to keep track of the size
        int idealBasketSize = 4 * firstBatch->num_rows() + 1024 + field->type()->byte_width() * firstBatch->num_rows(); // minimal additional size needed, otherwise we get 2 baskets
        int basketSize = std::max(32000, idealBasketSize);                                                              // keep a minimum value
        sizeBranch->SetBasketSize(basketSize);
        branch->SetBasketSize(basketSize);
      }
    }
    O2_SIGNPOST_END(root_arrow_fs, sid, "finaliseBasketSize", "Done");
  }

 public:
  // Create the TTree based on the physical_schema, not the one in the batch.
  // The write method will have to reconcile the two schemas.
  TTreeFileWriter(std::shared_ptr<arrow::Schema> schema, std::shared_ptr<arrow::dataset::FileWriteOptions> options,
                  std::shared_ptr<arrow::io::OutputStream> destination,
                  arrow::fs::FileLocator destination_locator)
    : FileWriter(schema, options, destination, destination_locator)
  {
    // Batches have the same number of entries for each column.
    auto directoryStream = std::dynamic_pointer_cast<TDirectoryFileOutputStream>(destination_);
    auto treeStream = std::dynamic_pointer_cast<TTreeOutputStream>(destination_);

    if (directoryStream.get()) {
      TDirectoryFile* dir = directoryStream->GetDirectory();
      dir->cd();
      auto* tree = new TTree(destination_locator_.path.c_str(), "");
      treeStream = std::make_shared<TTreeOutputStream>(tree, "");
    } else if (treeStream.get()) {
      // We already have a tree stream, let's derive a new one
      // with the destination_locator_.path as prefix for the branches
      // This way we can multiplex multiple tables in the same tree.
      auto* tree = treeStream->GetTree();
      treeStream = std::make_shared<TTreeOutputStream>(tree, destination_locator_.path);
    } else {
      // I could simply set a prefix here to merge to an already existing tree.
      throw std::runtime_error("Unsupported backend.");
    }

    for (auto i = 0u; i < schema->fields().size(); ++i) {
      auto& field = schema->field(i);
      listSizes.push_back(1);

      int valuesIdealBasketSize = 0;
      // Construct all the needed branches.
      switch (field->type()->id()) {
        case arrow::Type::FIXED_SIZE_LIST: {
          listSizes.back() = std::static_pointer_cast<arrow::FixedSizeListType>(field->type())->list_size();
          valuesIdealBasketSize = 1024 + valueTypes.back()->byte_width() * listSizes.back();
          valueTypes.push_back(field->type()->field(0)->type());
          sizesBranches.push_back(nullptr);
          std::string leafList = fmt::format("{}[{}]{}", field->name(), listSizes.back(), rootSuffixFromArrow(valueTypes.back()->id()));
          branches.push_back(treeStream->CreateBranch(field->name().c_str(), leafList.c_str()));
        } break;
        case arrow::Type::LIST: {
          valueTypes.push_back(field->type()->field(0)->type());
          std::string leafList = fmt::format("{}[{}_size]{}", field->name(), field->name(), rootSuffixFromArrow(valueTypes.back()->id()));
          listSizes.back() = -1; // VLA, we need to calculate it on the fly;
          std::string sizeLeafList = field->name() + "_size/I";
          sizesBranches.push_back(treeStream->CreateBranch((field->name() + "_size").c_str(), sizeLeafList.c_str()));
          branches.push_back(treeStream->CreateBranch(field->name().c_str(), leafList.c_str()));
          // Notice that this could be replaced by a better guess of the
          // average size of the list elements, but this is not trivial.
        } break;
        default: {
          valueTypes.push_back(field->type());
          std::string leafList = field->name() + rootSuffixFromArrow(valueTypes.back()->id());
          sizesBranches.push_back(nullptr);
          branches.push_back(treeStream->CreateBranch(field->name().c_str(), leafList.c_str()));
        } break;
      }
    }
    // We create the branches from the schema
  }

  arrow::Status Write(const std::shared_ptr<arrow::RecordBatch>& batch) override
  {
    if (firstBasket) {
      firstBasket = false;
      finaliseBasketSize(batch);
    }

    // Support writing empty tables
    if (batch->columns().empty() || batch->num_rows() == 0) {
      return arrow::Status::OK();
    }

    // Batches have the same number of entries for each column.
    auto directoryStream = std::dynamic_pointer_cast<TDirectoryFileOutputStream>(destination_);
    TTree* tree = nullptr;
    if (directoryStream.get()) {
      TDirectoryFile* dir = directoryStream->GetDirectory();
      tree = (TTree*)dir->Get(destination_locator_.path.c_str());
    }
    auto treeStream = std::dynamic_pointer_cast<TTreeOutputStream>(destination_);

    if (!tree) {
      // I could simply set a prefix here to merge to an already existing tree.
      throw std::runtime_error("Unsupported backend.");
    }

    for (auto i = 0u; i < batch->columns().size(); ++i) {
      auto column = batch->column(i);
      auto& field = batch->schema()->field(i);

      valueArrays.push_back(nullptr);

      switch (field->type()->id()) {
        case arrow::Type::FIXED_SIZE_LIST: {
          auto list = std::static_pointer_cast<arrow::FixedSizeListArray>(column);
          if (list->list_type()->field(0)->type()->id() == arrow::Type::BOOL) {
            int64_t length = list->length() * list->list_type()->list_size();
            arrow::UInt8Builder builder;
            auto ok = builder.Reserve(length);
            // I need to build an array of uint8_t for the conversion to ROOT which uses
            // bytes for boolans.
            auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(list->values());
            for (int64_t i = 0; i < length; ++i) {
              if (boolArray->IsValid(i)) {
                // Expand each boolean value (true/false) to uint8 (1/0)
                uint8_t value = boolArray->Value(i) ? 1 : 0;
                auto ok = builder.Append(value);
              } else {
                // Append null for invalid entries
                auto ok = builder.AppendNull();
              }
            }
            valueArrays.back() = *builder.Finish();
          } else {
            valueArrays.back() = list->values();
          }
        } break;
        case arrow::Type::LIST: {
          auto list = std::static_pointer_cast<arrow::ListArray>(column);
          valueArrays.back() = list->values();
        } break;
        case arrow::Type::BOOL: {
          // In case of arrays of booleans, we need to go back to their
          // char based representation for ROOT to save them.
          auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(column);

          int64_t length = boolArray->length();
          arrow::UInt8Builder builder;
          auto ok = builder.Reserve(length);

          for (int64_t i = 0; i < length; ++i) {
            if (boolArray->IsValid(i)) {
              // Expand each boolean value (true/false) to uint8 (1/0)
              uint8_t value = boolArray->Value(i) ? 1 : 0;
              auto ok = builder.Append(value);
            } else {
              // Append null for invalid entries
              auto ok = builder.AppendNull();
            }
          }
          valueArrays.back() = *builder.Finish();
        } break;
        default:
          valueArrays.back() = column;
      }
    }

    int64_t pos = 0;
    while (pos < batch->num_rows()) {
      for (size_t bi = 0; bi < branches.size(); ++bi) {
        auto* branch = branches[bi];
        auto* sizeBranch = sizesBranches[bi];
        auto array = batch->column(bi);
        auto& field = batch->schema()->field(bi);
        auto& listSize = listSizes[bi];
        auto valueType = valueTypes[bi];
        auto valueArray = valueArrays[bi];

        switch (field->type()->id()) {
          case arrow::Type::LIST: {
            auto list = std::static_pointer_cast<arrow::ListArray>(array);
            listSize = list->value_length(pos);
            uint8_t const* buffer = std::static_pointer_cast<arrow::PrimitiveArray>(valueArray)->values()->data() + array->offset() + list->value_offset(pos) * valueType->byte_width();
            branch->SetAddress((void*)buffer);
            sizeBranch->SetAddress(&listSize);
          } break;
          case arrow::Type::FIXED_SIZE_LIST:
          default: {
            // needed for the boolean case, I should probably cache this.
            auto byteWidth = valueType->byte_width() ? valueType->byte_width() : 1;
            uint8_t const* buffer = std::static_pointer_cast<arrow::PrimitiveArray>(valueArray)->values()->data() + array->offset() + pos * listSize * byteWidth;
            branch->SetAddress((void*)buffer);
          };
        }
      }
      tree->Fill();
      ++pos;
    }
    return arrow::Status::OK();
  }

  arrow::Future<> FinishInternal() override
  {
    auto treeStream = std::dynamic_pointer_cast<TTreeOutputStream>(destination_);
    auto* tree = treeStream->GetTree();
    tree->Write("", TObject::kOverwrite);
    tree->SetDirectory(nullptr);

    return {};
  };
};
arrow::Result<std::shared_ptr<arrow::dataset::FileWriter>> TTreeFileFormat::MakeWriter(std::shared_ptr<arrow::io::OutputStream> destination, std::shared_ptr<arrow::Schema> schema, std::shared_ptr<arrow::dataset::FileWriteOptions> options, arrow::fs::FileLocator destination_locator) const
{
  auto writer = std::make_shared<TTreeFileWriter>(schema, options, destination, destination_locator);
  return std::dynamic_pointer_cast<arrow::dataset::FileWriter>(writer);
}

std::shared_ptr<arrow::dataset::FileWriteOptions> TTreeFileFormat::DefaultWriteOptions()
{
  std::shared_ptr<TTreeFileWriteOptions> options(
    new TTreeFileWriteOptions(shared_from_this()));
  return options;
}

TTreeFileSystem::~TTreeFileSystem() = default;

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(TTreeObjectReadingImplementation, RootObjectReadingImplementation);
DEFINE_DPL_PLUGINS_END
} // namespace o2::framework

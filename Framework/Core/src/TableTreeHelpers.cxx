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
#include "Framework/TableTreeHelpers.h"
#include "Framework/Logger.h"
#include "Framework/Endian.h"

#include "arrow/type_traits.h"
#include <arrow/util/key_value_metadata.h>
#include <TBufferFile.h>

#include <utility>
namespace TableTreeHelpers
{
static constexpr char const* sizeBranchSuffix = "_size";
} // namespace TableTreeHelpers

namespace o2::framework
{
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
      throw runtime_error_f("Unsupported branch type: %d", static_cast<int>(type));
  }
}

auto basicROOTTypeFromArrow(arrow::Type::type id)
{
  switch (id) {
    case arrow::Type::BOOL:
      return ROOTTypeInfo{EDataType::kBool_t, "/O", TDataType::GetDataType(EDataType::kBool_t)->Size()};
    case arrow::Type::UINT8:
      return ROOTTypeInfo{EDataType::kUChar_t, "/b", TDataType::GetDataType(EDataType::kUChar_t)->Size()};
    case arrow::Type::UINT16:
      return ROOTTypeInfo{EDataType::kUShort_t, "/s", TDataType::GetDataType(EDataType::kUShort_t)->Size()};
    case arrow::Type::UINT32:
      return ROOTTypeInfo{EDataType::kUInt_t, "/i", TDataType::GetDataType(EDataType::kUInt_t)->Size()};
    case arrow::Type::UINT64:
      return ROOTTypeInfo{EDataType::kULong64_t, "/l", TDataType::GetDataType(EDataType::kULong64_t)->Size()};
    case arrow::Type::INT8:
      return ROOTTypeInfo{EDataType::kChar_t, "/B", TDataType::GetDataType(EDataType::kChar_t)->Size()};
    case arrow::Type::INT16:
      return ROOTTypeInfo{EDataType::kShort_t, "/S", TDataType::GetDataType(EDataType::kShort_t)->Size()};
    case arrow::Type::INT32:
      return ROOTTypeInfo{EDataType::kInt_t, "/I", TDataType::GetDataType(EDataType::kInt_t)->Size()};
    case arrow::Type::INT64:
      return ROOTTypeInfo{EDataType::kLong64_t, "/L", TDataType::GetDataType(EDataType::kLong64_t)->Size()};
    case arrow::Type::FLOAT:
      return ROOTTypeInfo{EDataType::kFloat_t, "/F", TDataType::GetDataType(EDataType::kFloat_t)->Size()};
    case arrow::Type::DOUBLE:
      return ROOTTypeInfo{EDataType::kDouble_t, "/D", TDataType::GetDataType(EDataType::kDouble_t)->Size()};
    default:
      throw runtime_error("Unsupported arrow column type");
  }
}

TBranch* BranchToColumn::branch()
{
  return mBranch;
}

BranchToColumn::BranchToColumn(TBranch* branch, bool VLA, std::string name, EDataType type, int listSize, arrow::MemoryPool* pool)
  : mBranch{branch},
    mVLA{VLA},
    mColumnName{std::move(name)},
    mType{type},
    mArrowType{arrowTypeFromROOT(type, listSize)},
    mListSize{listSize},
    mPool{pool}

{
  if (mType == EDataType::kBool_t) {
    if (mListSize > 1) {
      auto status = arrow::MakeBuilder(mPool, mArrowType->field(0)->type(), &mBuilder);
      if (!status.ok()) {
        throw runtime_error("Cannot create value builder");
      }
      mListBuilder = std::make_unique<arrow::FixedSizeListBuilder>(mPool, std::move(mBuilder), mListSize);
      mValueBuilder = static_cast<arrow::FixedSizeListBuilder*>(mListBuilder.get())->value_builder();
    } else {
      auto status = arrow::MakeBuilder(mPool, mArrowType, &mBuilder);
      if (!status.ok()) {
        throw runtime_error("Cannot create builder");
      }
      mValueBuilder = mBuilder.get();
    }
  }
}

template <typename T>
inline T doSwap(T)
{
  static_assert(always_static_assert_v<T>, "Unsupported type");
}

template <>
inline uint16_t doSwap(uint16_t x)
{
  return swap16_(x);
}

template <>
inline uint32_t doSwap(uint32_t x)
{
  return swap32_(x);
}

template <>
inline uint64_t doSwap(uint64_t x)
{
  return swap64_(x);
}

template <typename T>
void doSwapCopy_(void* dest, void* source, int size) noexcept
{
  auto tdest = static_cast<T*>(dest);
  auto tsrc = static_cast<T*>(source);
  for (auto i = 0; i < size; ++i) {
    tdest[i] = doSwap<T>(tsrc[i]);
  }
}

void swapCopy(unsigned char* dest, char* source, int size, int typeSize) noexcept
{
  switch (typeSize) {
    case 1:
      return (void)std::memcpy(dest, source, size);
    case 2:
      return doSwapCopy_<uint16_t>(dest, source, size);
    case 4:
      return doSwapCopy_<uint32_t>(dest, source, size);
    case 8:
      return doSwapCopy_<uint64_t>(dest, source, size);
  }
}

std::pair<std::shared_ptr<arrow::ChunkedArray>, std::shared_ptr<arrow::Field>> BranchToColumn::read(TBuffer* buffer)
{
  auto totalEntries = mBranch->GetEntries();
  arrow::Status status;
  int readEntries = 0;
  buffer->Reset();
  std::shared_ptr<arrow::Array> array;

  if (mType == EDataType::kBool_t) {
    // boolean array special case: we need to use builder to create the bitmap
    status = mValueBuilder->Reserve(totalEntries * mListSize);
    if (mListSize > 1) {
      status &= mListBuilder->Reserve(totalEntries);
    }
    if (!status.ok()) {
      throw runtime_error("Failed to reserve memory for array builder");
    }
    while (readEntries < totalEntries) {
      auto readLast = mBranch->GetBulkRead().GetBulkEntries(readEntries, *buffer);
      readEntries += readLast;
      status &= static_cast<arrow::BooleanBuilder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint8_t const*>(buffer->GetCurrent()), readLast * mListSize);
    }
    if (mListSize > 1) {
      status &= static_cast<arrow::FixedSizeListBuilder*>(mListBuilder.get())->AppendValues(readEntries);
    }
    if (!status.ok()) {
      throw runtime_error("Failed to append values to array");
    }
    if (mListSize > 1) {
      status &= mListBuilder->Finish(&array);
    } else {
      status &= mValueBuilder->Finish(&array);
    }
    if (!status.ok()) {
      throw runtime_error("Failed to create array");
    }
  } else {
    // other types: use serialized read to build arrays directly
    auto&& result = arrow::AllocateResizableBuffer(mBranch->GetTotBytes(), mPool);
    if (!result.ok()) {
      throw runtime_error("Cannot allocate values buffer");
    }
    std::shared_ptr<arrow::Buffer> arrowValuesBuffer = std::move(result).ValueUnsafe();
    auto ptr = arrowValuesBuffer->mutable_data();
    if (ptr == nullptr) {
      throw runtime_error("Invalid buffer");
    }

    auto typeSize = TDataType::GetDataType(mType)->Size();
    std::unique_ptr<TBufferFile> offsetBuffer = nullptr;

    uint32_t offset = 0;
    int count = 0;
    std::shared_ptr<arrow::Buffer> arrowOffsetBuffer;
    gsl::span<int> offsets;
    int size = 0;
    uint32_t totalSize = 0;
    TBranch* mSizeBranch = nullptr;
    if (mVLA) {
      mSizeBranch = mBranch->GetTree()->GetBranch((std::string{mBranch->GetName()} + TableTreeHelpers::sizeBranchSuffix).c_str());
      offsetBuffer = std::make_unique<TBufferFile>(TBuffer::EMode::kWrite, 4 * 1024 * 1024);
      result = arrow::AllocateResizableBuffer((totalEntries + 1) * (int64_t)sizeof(int), mPool);
      if (!result.ok()) {
        throw runtime_error("Cannot allocate offset buffer");
      }
      arrowOffsetBuffer = std::move(result).ValueUnsafe();
      unsigned char* ptrOffset = arrowOffsetBuffer->mutable_data();
      auto* tPtrOffset = reinterpret_cast<int*>(ptrOffset);
      offsets = gsl::span<int>{tPtrOffset, tPtrOffset + totalEntries + 1};

      // read sizes first
      while (readEntries < totalEntries) {
        auto readLast = mSizeBranch->GetBulkRead().GetEntriesSerialized(readEntries, *offsetBuffer);
        readEntries += readLast;
        for (auto i = 0; i < readLast; ++i) {
          offsets[count++] = (int)offset;
          offset += swap32_(reinterpret_cast<uint32_t*>(offsetBuffer->GetCurrent())[i]);
        }
      }
      offsets[count] = (int)offset;
      totalSize = offset;
      readEntries = 0;
    }

    while (readEntries < totalEntries) {
      auto readLast = mBranch->GetBulkRead().GetEntriesSerialized(readEntries, *buffer);
      if (mVLA) {
        size = offsets[readEntries + readLast] - offsets[readEntries];
      } else {
        size = readLast * mListSize;
      }
      readEntries += readLast;
      swapCopy(ptr, buffer->GetCurrent(), size, typeSize);
      ptr += (ptrdiff_t)(size * typeSize);
    }
    if (!mVLA) {
      totalSize = readEntries * mListSize;
    }
    std::shared_ptr<arrow::PrimitiveArray> varray;
    switch (mListSize) {
      case -1:
        varray = std::make_shared<arrow::PrimitiveArray>(mArrowType->field(0)->type(), totalSize, arrowValuesBuffer);
        array = std::make_shared<arrow::ListArray>(mArrowType, readEntries, arrowOffsetBuffer, varray);
        break;
      case 1:
        array = std::make_shared<arrow::PrimitiveArray>(mArrowType, readEntries, arrowValuesBuffer);
        break;
      default:
        varray = std::make_shared<arrow::PrimitiveArray>(mArrowType->field(0)->type(), totalSize, arrowValuesBuffer);
        array = std::make_shared<arrow::FixedSizeListArray>(mArrowType, readEntries, varray);
    }
  }

  auto fullArray = std::make_shared<arrow::ChunkedArray>(array);
  auto field = std::make_shared<arrow::Field>(mBranch->GetName(), mArrowType);

  mBranch->SetStatus(false);
  mBranch->DropBaskets("all");
  mBranch->Reset();
  mBranch->GetTransientBuffer(0)->Expand(0);

  return std::make_pair(fullArray, field);
}

ColumnToBranch::ColumnToBranch(TTree* tree, std::shared_ptr<arrow::ChunkedArray> const& column, std::shared_ptr<arrow::Field> const& field)
  : mBranchName{field->name()},
    mColumn{column.get()},
    mFieldSize{field->type()->byte_width()}
{
  std::string leafList;
  std::string sizeLeafList;
  auto arrowType = field->type();
  mFieldType = arrowType->id();
  switch (mFieldType) {
    case arrow::Type::FIXED_SIZE_LIST:
      mListSize = std::static_pointer_cast<arrow::FixedSizeListType>(arrowType)->list_size();
      arrowType = arrowType->field(0)->type();
      mElementType = basicROOTTypeFromArrow(arrowType->id());
      leafList = mBranchName + "[" + std::to_string(mListSize) + "]" + mElementType.suffix;
      mFieldSize = arrowType->byte_width() * mListSize;
      break;
    case arrow::Type::LIST:
      arrowType = arrowType->field(0)->type();
      mElementType = basicROOTTypeFromArrow(arrowType->id());
      leafList = mBranchName + "[" + mBranchName + TableTreeHelpers::sizeBranchSuffix + "]" + mElementType.suffix;
      sizeLeafList = mBranchName + TableTreeHelpers::sizeBranchSuffix + "/I";
      // Notice that this could be replaced by a better guess of the
      // average size of the list elements, but this is not trivial.
      mFieldSize = arrowType->byte_width();
      break;
    default:
      mElementType = basicROOTTypeFromArrow(arrowType->id());
      leafList = mBranchName + mElementType.suffix;
      break;
  }
  if (!sizeLeafList.empty()) {
    mSizeBranch = tree->GetBranch((mBranchName + TableTreeHelpers::sizeBranchSuffix).c_str());
    if (mSizeBranch == nullptr) {
      mSizeBranch = tree->Branch((mBranchName + TableTreeHelpers::sizeBranchSuffix).c_str(), (char*)nullptr, sizeLeafList.c_str());
    }
  }
  mBranch = tree->GetBranch(mBranchName.c_str());
  if (mBranch == nullptr) {
    mBranch = tree->Branch(mBranchName.c_str(), (char*)nullptr, leafList.c_str());
  }
  if (mElementType.type == EDataType::kBool_t) {
    cache.resize(mListSize);
  }
  accessChunk();
}

void ColumnToBranch::at(const int64_t* pos)
{
  if (O2_BUILTIN_UNLIKELY(*pos - mFirstIndex >= mChunkLength)) {
    nextChunk();
  }
  if (mElementType.type == EDataType::kBool_t) {
    auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(mCurrentArray);
    for (auto i = 0; i < mListSize; ++i) {
      cache[i] = boolArray->Value((*pos - mFirstIndex) * mListSize + i);
    }
    mBranch->SetAddress((void*)(cache.data()));
    return;
  }
  uint8_t const* buffer;
  switch (mFieldType) {
    case arrow::Type::LIST: {
      auto list = std::static_pointer_cast<arrow::ListArray>(mCurrentArray);
      mListSize = list->value_length((*pos - mFirstIndex));
      buffer = std::static_pointer_cast<arrow::PrimitiveArray>(list->values())->values()->data() + mCurrentArray->offset() + list->value_offset((*pos - mFirstIndex)) * mElementType.size;
      mBranch->SetAddress((void*)buffer);
      mSizeBranch->SetAddress(&mListSize);
    };
      break;
    case arrow::Type::FIXED_SIZE_LIST:
    default: {
      buffer = std::static_pointer_cast<arrow::PrimitiveArray>(mCurrentArray)->values()->data() + mCurrentArray->offset() + (*pos - mFirstIndex) * mListSize * mElementType.size;
      mBranch->SetAddress((void*)buffer);
    };
  }
}

void ColumnToBranch::accessChunk()
{
  auto array = mColumn->chunk(mCurrentChunk);
  switch (mFieldType) {
    case arrow::Type::FIXED_SIZE_LIST: {
      auto list = std::static_pointer_cast<arrow::FixedSizeListArray>(array);
      mChunkLength = list->length();
      mCurrentArray = list->values();
    };
      break;
    case arrow::Type::LIST: {
      auto list = std::static_pointer_cast<arrow::ListArray>(array);
      mChunkLength = list->length();
      mCurrentArray = list;
    };
      break;
    default:
      mCurrentArray = array;
      mChunkLength = mCurrentArray->length();
  }
}

void ColumnToBranch::nextChunk()
{
  mFirstIndex += mChunkLength;
  ++mCurrentChunk;
  accessChunk();
}

TableToTree::TableToTree(std::shared_ptr<arrow::Table> const& table, TFile* file, const char* treename)
{
  mTable = table.get();
  mTree.reset(static_cast<TTree*>(file->Get(treename)));
  if (mTree) {
    return;
  }
  std::string treeName(treename);
  auto pos = treeName.find_first_of('/');
  if (pos != std::string::npos) {
    file->cd(treeName.substr(0, pos).c_str());
    treeName = treeName.substr(pos + 1, std::string::npos);
  }
  mTree = std::make_shared<TTree>(treeName.c_str(), treeName.c_str());
}

void TableToTree::addAllBranches()
{
  mRows = mTable->num_rows();
  auto columns = mTable->columns();
  auto fields = mTable->schema()->fields();
  assert(columns.size() == fields.size());
  for (auto i = 0u; i < columns.size(); ++i) {
    addBranch(columns[i], fields[i]);
  }
}

void TableToTree::addBranch(std::shared_ptr<arrow::ChunkedArray> const& column, std::shared_ptr<arrow::Field> const& field)
{
  if (mRows == 0) {
    mRows = column->length();
  } else if (mRows != column->length()) {
    throw runtime_error_f("Adding incompatible column with size %d (num rows = %d)", column->length(), mRows);
  }
  mColumnReaders.emplace_back(new ColumnToBranch{mTree.get(), column, field});
}

std::shared_ptr<TTree> TableToTree::process()
{
  int64_t row = 0;
  if (mTree->GetNbranches() == 0 || mRows == 0) {
    mTree->Write("", TObject::kOverwrite);
    mTree->SetDirectory(nullptr);
    return mTree;
  }

  for (auto& reader : mColumnReaders) {
    int idealBasketSize = 1024 + reader->fieldSize() * mRows; // minimal additional size needed, otherwise we get 2 baskets
    int basketSize = std::max(32000, idealBasketSize);        // keep a minimum value
    mTree->SetBasketSize(reader->branchName(), basketSize);
  }

  while (row < mRows) {
    for (auto& reader : mColumnReaders) {
      reader->at(&row);
    }
    mTree->Fill();
    ++row;
  }
  mTree->Write("", TObject::kOverwrite);
  mTree->SetDirectory(nullptr);
  return mTree;
}

TreeToTable::TreeToTable(arrow::MemoryPool* pool)
  : mArrowMemoryPool{pool}
{
}

namespace
{
struct BranchInfo {
  std::string name;
  TBranch* ptr;
  bool mVLA;
};
} // namespace

void TreeToTable::addAllColumns(TTree* tree, std::vector<std::string>&& names)
{
  auto branches = tree->GetListOfBranches();
  auto n = branches->GetEntries();
  if (n == 0) {
    throw runtime_error("Tree has no branches");
  }

  std::vector<BranchInfo> branchInfos;
  for (auto i = 0; i < n; ++i) {
    auto branch = static_cast<TBranch*>(branches->At(i));
    auto name = std::string{branch->GetName()};
    auto pos = name.find(TableTreeHelpers::sizeBranchSuffix);
    if (pos != std::string::npos) {
      name.erase(pos);
      branchInfos.emplace_back(BranchInfo{name, (TBranch*)nullptr, true});
    } else {
      auto lookup = std::find_if(branchInfos.begin(), branchInfos.end(), [&](BranchInfo const& bi) {
        return bi.name == name;
      });
      if (lookup == branchInfos.end()) {
        branchInfos.emplace_back(BranchInfo{name, branch, false});
      } else {
        lookup->ptr = branch;
      }
    }
  }

  if (names.empty()) {
    for (auto& bi : branchInfos) {
      addReader(bi.ptr, bi.name, bi.mVLA);
    }
  } else {
    for (auto& name : names) {
      auto lookup = std::find_if(branchInfos.begin(), branchInfos.end(), [&](BranchInfo const& bi) {
        return name == bi.name;
      });
      if (lookup != branchInfos.end()) {
        addReader(lookup->ptr, lookup->name, lookup->mVLA);
      }
    }
    if (names.size() != mBranchReaders.size()) {
      LOGF(warn, "Not all requested columns were found in the tree");
    }
  }
  if (mBranchReaders.empty()) {
    throw runtime_error("No columns will be read");
  }
  //tree->SetCacheSize(50000000);
  // FIXME: see https://github.com/root-project/root/issues/8962 and enable
  // again once fixed.
  //tree->SetClusterPrefetch(true);
  //for (auto& reader : mBranchReaders) {
  //  tree->AddBranchToCache(reader->branch());
  //}
  //tree->StopCacheLearningPhase();
}

void TreeToTable::setLabel(const char* label)
{
  mTableLabel = label;
}

void TreeToTable::fill(TTree*)
{
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  static TBufferFile buffer{TBuffer::EMode::kWrite, 4 * 1024 * 1024};
  for (auto& reader : mBranchReaders) {
    buffer.Reset();
    auto arrayAndField = reader->read(&buffer);
    columns.push_back(arrayAndField.first);
    fields.push_back(arrayAndField.second);
  }

  auto schema = std::make_shared<arrow::Schema>(fields, std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{mTableLabel}));
  mTable = arrow::Table::Make(schema, columns);
}

void TreeToTable::addReader(TBranch* branch, std::string const& name, bool VLA)
{
  static TClass* cls;
  EDataType type;
  branch->GetExpectedType(cls, type);
  auto listSize = -1;
  if (!VLA) {
    listSize = static_cast<TLeaf*>(branch->GetListOfLeaves()->At(0))->GetLenStatic();
  }
  mBranchReaders.emplace_back(std::make_unique<BranchToColumn>(branch, VLA, name, type, listSize, mArrowMemoryPool));
}

std::shared_ptr<arrow::Table> TreeToTable::finalize()
{
  return mTable;
}

} // namespace o2::framework

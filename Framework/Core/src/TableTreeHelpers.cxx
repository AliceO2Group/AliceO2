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

#include "arrow/type_traits.h"
#include <arrow/util/key_value_metadata.h>
#include <TBufferFile.h>

#include <utility>

namespace o2::framework
{
auto arrowTypeFromROOT(EDataType type, int size)
{
  auto typeGenerator = [](std::shared_ptr<arrow::DataType>&& type, int size) -> std::shared_ptr<arrow::DataType> {
    if (size > 1) {
      return arrow::fixed_size_list(type, size);
    }
    if (size == 1) {
      return std::move(type);
    }
    if (size == -1) {
      return arrow::list(type);
    }
    O2_BUILTIN_UNREACHABLE();
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
      throw runtime_error("Unsupported branch type");
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

BranchToColumn::BranchToColumn(TBranch* branch, std::string name, EDataType type, int listSize, arrow::MemoryPool* pool)
  : mBranch{branch},
    mColumnName{std::move(name)},
    mType{type},
    mArrowType{arrowTypeFromROOT(type, listSize)},
    mListSize{listSize}

{
  if (mListSize > 1) {
    auto status = arrow::MakeBuilder(pool, mArrowType->field(0)->type(), &mBuilder);
    if (!status.ok()) {
      throw runtime_error("Cannot create value builder");
    }
    mListBuilder = std::make_unique<arrow::FixedSizeListBuilder>(pool, std::move(mBuilder), mListSize);
    mValueBuilder = static_cast<arrow::FixedSizeListBuilder*>(mListBuilder.get())->value_builder();
  } else {
    auto status = arrow::MakeBuilder(pool, mArrowType, &mBuilder);
    if (!status.ok()) {
      throw runtime_error("Cannot create builder");
    }
    mValueBuilder = mBuilder.get();
  }
}

BranchToColumn::BranchToColumn(TBranch* branch, TBranch* sizeBranch, std::string name, EDataType type, arrow::MemoryPool* pool)
  : mBranch{branch},
    mSizeBranch{sizeBranch},
    mColumnName{std::move(name)},
    mType{type},
    mArrowType{arrowTypeFromROOT(type, -1)}
{
  auto status = arrow::MakeBuilder(pool, mArrowType->field(0)->type(), &mBuilder);
  if (!status.ok()) {
    throw runtime_error("Cannot create value builder");
  }
  mListBuilder = std::make_unique<arrow::ListBuilder>(pool, std::move(mBuilder));
  mValueBuilder = static_cast<arrow::ListBuilder*>(mListBuilder.get())->value_builder();
}

std::pair<std::shared_ptr<arrow::ChunkedArray>, std::shared_ptr<arrow::Field>> BranchToColumn::read(TBuffer* buffer)
{
  auto totalEntries = static_cast<int>(mBranch->GetEntries());
  auto status = reserve(totalEntries);
  if (!status.ok()) {
    throw runtime_error("Failed to reserve memory for array builder");
  }
  if (mSizeBranch != nullptr) {
    status &= static_cast<arrow::ListBuilder*>(mListBuilder.get())->Append();
  }
  int readEntries = 0;
  buffer->Reset();
  while (readEntries < totalEntries) {
    auto readLast = mBranch->GetBulkRead().GetBulkEntries(readEntries, *buffer);
    readEntries += readLast;
    status &= appendValues(reinterpret_cast<unsigned char const*>(buffer->GetCurrent()), readLast);
  }
  if (mListSize > 1) {
    status &= static_cast<arrow::FixedSizeListBuilder*>(mListBuilder.get())->AppendValues(readEntries);
  } else if (mSizeBranch != nullptr) {
    std::vector<int> offsets(mSizeBranch->GetEntries() + 1);
    offsets[0] = 0;
    int readOffsets = 0;
    buffer->Reset();
    while (readOffsets < mSizeBranch->GetEntries()) {
      auto readLast = mSizeBranch->GetBulkRead().GetBulkEntries(readOffsets, *buffer);
      auto data = reinterpret_cast<int*>(buffer->GetCurrent());
      for (auto i = 0; i < readLast; ++i) {
        offsets[readOffsets + i + 1] = offsets[readOffsets + i] + data[i];
      }
      readOffsets += readLast;
    }
    offsets.push_back(readOffsets);
    status &= static_cast<arrow::ListBuilder*>(mListBuilder.get())->AppendValues(offsets.data(), totalEntries);
  }
  if (!status.ok()) {
    throw runtime_error("Failed to append values to array");
  }
  std::shared_ptr<arrow::Array> array;
  status &= finish(&array);
  if (!status.ok()) {
    throw runtime_error("Failed to create array");
  }
  auto fullArray = std::make_shared<arrow::ChunkedArray>(array);
  auto field = std::make_shared<arrow::Field>(mBranch->GetName(), mArrowType);

  mBranch->SetStatus(false);
  mBranch->DropBaskets("all");
  mBranch->Reset();
  mBranch->GetTransientBuffer(0)->Expand(0);

  return std::make_pair(fullArray, field);
}

arrow::Status BranchToColumn::appendValues(unsigned char const* buffer, int numEntries)
{
  switch (mType) {
    case EDataType::kBool_t:
      return static_cast<arrow::BooleanBuilder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint8_t const*>(buffer), numEntries * mListSize);
    case EDataType::kUChar_t:
      return static_cast<arrow::UInt8Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint8_t const*>(buffer), numEntries * mListSize);
    case EDataType::kUShort_t:
      return static_cast<arrow::UInt16Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint16_t const*>(buffer), numEntries * mListSize);
    case EDataType::kUInt_t:
      return static_cast<arrow::UInt32Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint32_t const*>(buffer), numEntries * mListSize);
    case EDataType::kULong64_t:
      return static_cast<arrow::UInt64Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint64_t const*>(buffer), numEntries * mListSize);
    case EDataType::kChar_t:
      return static_cast<arrow::Int8Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<int8_t const*>(buffer), numEntries * mListSize);
    case EDataType::kShort_t:
      return static_cast<arrow::Int16Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<int16_t const*>(buffer), numEntries * mListSize);
    case EDataType::kInt_t:
      return static_cast<arrow::Int32Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<int32_t const*>(buffer), numEntries * mListSize);
    case EDataType::kLong64_t:
      return static_cast<arrow::Int64Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<int64_t const*>(buffer), numEntries * mListSize);
    case EDataType::kFloat_t:
      return static_cast<arrow::FloatBuilder*>(mValueBuilder)->AppendValues(reinterpret_cast<float const*>(buffer), numEntries * mListSize);
    case EDataType::kDouble_t:
      return static_cast<arrow::DoubleBuilder*>(mValueBuilder)->AppendValues(reinterpret_cast<double const*>(buffer), numEntries * mListSize);
    default:
      throw runtime_error("Unsupported branch type");
  }
}

arrow::Status BranchToColumn::finish(std::shared_ptr<arrow::Array>* array)
{
  if (mListSize > 1 || mSizeBranch != nullptr) {
    return mListBuilder->Finish(array);
  }
  return mValueBuilder->Finish(array);
}

arrow::Status BranchToColumn::reserve(int numEntries)
{
  auto status = mValueBuilder->Reserve(numEntries * mListSize);
  if (mListSize > 1 || mSizeBranch != nullptr) {
    status &= mListBuilder->Reserve(numEntries);
  }
  return status;
}

ColumnToBranch::ColumnToBranch(TTree* tree, std::shared_ptr<arrow::ChunkedArray> const& column, std::shared_ptr<arrow::Field> const& field)
  : mBranchName{field->name()},
    mColumn{column.get()}
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
      break;
    case arrow::Type::LIST:
      arrowType = arrowType->field(0)->type();
      mElementType = basicROOTTypeFromArrow(arrowType->id());
      leafList = mBranchName + "[" + mBranchName + TableTreeHelpers::sizeBranchsuffix + "]" + mElementType.suffix;
      sizeLeafList = mBranchName + TableTreeHelpers::sizeBranchsuffix + "/I";
      break;
    default:
      mElementType = basicROOTTypeFromArrow(arrowType->id());
      leafList = mBranchName + mElementType.suffix;
      break;
  }
  if (!sizeLeafList.empty()) {
    mSizeBranch = tree->GetBranch((mBranchName + TableTreeHelpers::sizeBranchsuffix).c_str());
    if (mSizeBranch == nullptr) {
      mSizeBranch = tree->Branch((mBranchName + TableTreeHelpers::sizeBranchsuffix).c_str(), (char*)nullptr, sizeLeafList.c_str());
    }
  }
  mBranch = tree->GetBranch(mBranchName.c_str());
  if (mBranch == nullptr) {
    mBranch = tree->Branch(mBranchName.c_str(), (char*)nullptr, leafList.c_str());
  }
  if (mElementType.type == EDataType::kBool_t) {
    cache.reserve(mListSize);
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
    return mTree;
  }

  while (row < mRows) {
    for (auto& reader : mColumnReaders) {
      reader->at(&row);
    }
    mTree->Fill();
    ++row;
  }
  mTree->Write("", TObject::kOverwrite);
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
  TBranch* sizePtr;
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
    auto pos = name.find(TableTreeHelpers::sizeBranchsuffix);
    if (pos != std::string::npos) {
      name.erase(pos);
      branchInfos.emplace_back(BranchInfo{name, (TBranch*)nullptr, branch});
    } else {
      auto lookup = std::find_if(branchInfos.begin(), branchInfos.end(), [&](BranchInfo const& bi) {
        return bi.name == name;
      });
      if (lookup == branchInfos.end()) {
        branchInfos.emplace_back(BranchInfo{name, branch, (TBranch*)nullptr});
      } else {
        lookup->ptr = branch;
      }
    }
  }

  if (names.empty()) {
    for (auto& bi : branchInfos) {
      addReader(bi.ptr, bi.name, bi.sizePtr);
    }
  } else {
    for (auto& name : names) {
      auto lookup = std::find_if(branchInfos.begin(), branchInfos.end(), [&](BranchInfo const& bi) {
        return name == bi.name;
      });
      if (lookup != branchInfos.end()) {
        addReader(lookup->ptr, lookup->name, lookup->sizePtr);
      }
    }
    if (names.size() != mBranchReaders.size()) {
      LOGF(WARN, "Not all requested columns were found in the tree");
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

void TreeToTable::addReader(TBranch* branch, std::string const& name, TBranch* sizeBranch)
{
  static TClass* cls;
  EDataType type;
  branch->GetExpectedType(cls, type);
  if (sizeBranch == nullptr) {
    auto listSize = static_cast<TLeaf*>(branch->GetListOfLeaves()->At(0))->GetLenStatic();
    mBranchReaders.emplace_back(std::make_unique<BranchToColumn>(branch, name, type, listSize, mArrowMemoryPool));
    return;
  }
  mBranchReaders.emplace_back(std::make_unique<BranchToColumn>(branch, sizeBranch, name, type, mArrowMemoryPool));
}

std::shared_ptr<arrow::Table> TreeToTable::finalize()
{
  return mTable;
}

} // namespace o2::framework

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

namespace o2::framework
{
auto arrowTypeFromROOT(EDataType type, int size)
{
  auto typeGenerator = [](std::shared_ptr<arrow::DataType>&& type, int size) -> std::shared_ptr<arrow::DataType> {
    if (size == 1) {
      return std::move(type);
    }
    return arrow::fixed_size_list(type, size);
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

BranchToColumn::BranchToColumn(TBranch* branch, const char* name, EDataType type, int listSize, arrow::MemoryPool* pool)
  : mBranch{branch},
    mColumnName{name},
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
    mValueBuilder = mListBuilder->value_builder();
  } else {
    auto status = arrow::MakeBuilder(pool, mArrowType, &mBuilder);
    if (!status.ok()) {
      throw runtime_error("Cannot create builder");
    }
    mValueBuilder = mBuilder.get();
  }
}

std::pair<std::shared_ptr<arrow::ChunkedArray>, std::shared_ptr<arrow::Field>> BranchToColumn::read(TBuffer* buffer)
{
  auto totalEntries = static_cast<int>(mBranch->GetEntries());
  auto status = reserve(totalEntries);
  if (!status.ok()) {
    throw runtime_error("Failed to reserve memory for array builder");
  }
  int readEntries = 0;
  buffer->Reset();
  while (readEntries < totalEntries) {
    auto readLast = mBranch->GetBulkRead().GetBulkEntries(readEntries, *buffer);
    readEntries += readLast;
    status &= appendValues(reinterpret_cast<unsigned char const*>(buffer->GetCurrent()), readLast);
  }
  if (!status.ok()) {
    throw runtime_error("Failed to append values to array");
  }
  std::shared_ptr<arrow::Array> array;
  status &= finish(&array);
  if (!status.ok()) {
    throw runtime_error("Failed to create boolean array");
  }
  auto fullArray = std::make_shared<arrow::ChunkedArray>(array);
  auto field = std::make_shared<arrow::Field>(mBranch->GetName(), mArrowType);

  mBranch->SetStatus(0);
  mBranch->DropBaskets("all");
  mBranch->Reset();
  mBranch->GetTransientBuffer(0)->Expand(0);

  return std::make_pair(fullArray, field);
}

arrow::Status BranchToColumn::appendValues(unsigned char const* buffer, int numEntries)
{
  arrow::Status status;
  switch (mType) {
    case EDataType::kBool_t:
      status = static_cast<arrow::BooleanBuilder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint8_t const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kUChar_t:
      status = static_cast<arrow::UInt8Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint8_t const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kUShort_t:
      status = static_cast<arrow::UInt16Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint16_t const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kUInt_t:
      status = static_cast<arrow::UInt32Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint32_t const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kULong64_t:
      status = static_cast<arrow::UInt64Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<uint64_t const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kChar_t:
      status = static_cast<arrow::Int8Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<int8_t const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kShort_t:
      status = static_cast<arrow::Int16Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<int16_t const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kInt_t:
      status = static_cast<arrow::Int32Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<int32_t const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kLong64_t:
      status = static_cast<arrow::Int64Builder*>(mValueBuilder)->AppendValues(reinterpret_cast<int64_t const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kFloat_t:
      status = static_cast<arrow::FloatBuilder*>(mValueBuilder)->AppendValues(reinterpret_cast<float const*>(buffer), numEntries * mListSize);
      break;
    case EDataType::kDouble_t:
      status = static_cast<arrow::DoubleBuilder*>(mValueBuilder)->AppendValues(reinterpret_cast<double const*>(buffer), numEntries * mListSize);
      break;
    default:
      throw runtime_error("Unsupported branch type");
  }
  if (mListSize > 1) {
    status &= mListBuilder->AppendValues(numEntries);
  }

  return status;
}

arrow::Status BranchToColumn::finish(std::shared_ptr<arrow::Array>* array)
{
  if (mListSize > 1) {
    return mListBuilder->Finish(array);
  }
  return mValueBuilder->Finish(array);
}

arrow::Status BranchToColumn::reserve(int numEntries)
{
  auto status = mValueBuilder->Reserve(numEntries * mListSize);
  if (mListSize > 1) {
    status &= mListBuilder->Reserve(numEntries);
  }
  return status;
}

ColumnToBranch::ColumnToBranch(TTree* tree, std::shared_ptr<arrow::ChunkedArray> const& column, std::shared_ptr<arrow::Field> const& field)
  : mBranchName{field->name()},
    mColumn{column.get()}
{
  auto arrowType = field->type();
  switch (arrowType->id()) {
    case arrow::Type::FIXED_SIZE_LIST:
      mListSize = std::static_pointer_cast<arrow::FixedSizeListType>(arrowType)->list_size();
      arrowType = arrowType->field(0)->type();
      break;
    default:
      break;
  }
  mType = basicROOTTypeFromArrow(arrowType->id());
  if (mListSize > 1) {
    mLeafList = mBranchName + "[" + std::to_string(mListSize) + "]" + mType.suffix;
  } else {
    mLeafList = mBranchName + mType.suffix;
  }
  mBranch = tree->GetBranch(mBranchName.c_str());
  if (mBranch == nullptr) {
    mBranch = tree->Branch(mBranchName.c_str(), (char*)nullptr, mLeafList.c_str());
  }
  if (mType.type == EDataType::kBool_t) {
    cache.reserve(mListSize);
    mCurrent = reinterpret_cast<uint8_t*>(cache.data());
    mLast = mCurrent + mListSize * mType.size;
    allocated = true;
  }
  accessChunk(0);
}

void ColumnToBranch::at(const int64_t* pos)
{
  mCurrentPos = pos;
  resetBuffer();
}

auto ColumnToBranch::getCurrentBuffer()
{
  std::shared_ptr<arrow::PrimitiveArray> array;
  if (mListSize > 1) {
    array = std::static_pointer_cast<arrow::PrimitiveArray>(std::static_pointer_cast<arrow::FixedSizeListArray>(mColumn->chunk(mCurrentChunk))->values());
  } else {
    array = std::static_pointer_cast<arrow::PrimitiveArray>(mColumn->chunk(mCurrentChunk));
  }
  return array;
}

void ColumnToBranch::resetBuffer()
{
  if (mType.type == EDataType::kBool_t) {
    if (O2_BUILTIN_UNLIKELY((*mCurrentPos - mFirstIndex) * mListSize >= getCurrentBuffer()->length())) {
      nextChunk();
    }
  } else {
    if (O2_BUILTIN_UNLIKELY(mCurrent >= mLast)) {
      nextChunk();
    }
  }
  accessChunk(*mCurrentPos);
  mBranch->SetAddress((void*)(mCurrent));
}

void ColumnToBranch::accessChunk(int64_t at)
{
  auto array = getCurrentBuffer();

  if (mType.type == EDataType::kBool_t) {
    auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(array);
    for (auto i = 0; i < mListSize; ++i) {
      cache[i] = (bool)boolArray->Value((at - mFirstIndex) * mListSize + i);
    }
  } else {
    mCurrent = array->values()->data() + (at - mFirstIndex) * mListSize * mType.size;
    mLast = mCurrent + array->length() * mListSize * mType.size;
  }
}

void ColumnToBranch::nextChunk()
{
  ++mCurrentChunk;
  mFirstIndex += getCurrentBuffer()->length();
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

void TreeToTable::addAllColumns(TTree* tree, std::vector<std::string>&& names)
{
  auto branches = tree->GetListOfBranches();
  auto n = branches->GetEntries();
  if (n == 0) {
    throw runtime_error("Tree has no branches");
  }

  if (names.empty()) {
    for (auto i = 0; i < n; ++i) {
      auto branch = static_cast<TBranch*>(branches->At(i));
      addReader(branch, branch->GetName());
    }
  } else {
    for (auto i = 0; i < n; ++i) {
      auto branch = static_cast<TBranch*>(branches->At(i));
      auto lookup = std::find_if(names.begin(), names.end(), [&](auto name) { return name == branch->GetName(); });
      if (lookup != names.end()) {
        addReader(branch, branch->GetName());
      }
      if (mBranchReaders.size() != names.size()) {
        LOGF(warn, "Not all requested columns were found in the tree");
      }
    }
  }
  if (mBranchReaders.empty()) {
    throw runtime_error("No columns will be read");
  }
  //tree->SetCacheSize(50000000);
  //// FIXME: see https://github.com/root-project/root/issues/8962 and enable
  //// again once fixed.
  ////tree->SetClusterPrefetch(true);
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

void TreeToTable::addReader(TBranch* branch, const char* name)
{
  static TClass* cls;
  EDataType type;
  branch->GetExpectedType(cls, type);
  auto listSize = static_cast<TLeaf*>(branch->GetListOfLeaves()->At(0))->GetLenStatic();
  mBranchReaders.emplace_back(std::make_unique<BranchToColumn>(branch, name, type, listSize, mArrowMemoryPool));
}

std::shared_ptr<arrow::Table> TreeToTable::finalize()
{
  return mTable;
}

} // namespace o2::framework

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
#include "Framework/Signpost.h"

#include <arrow/dataset/file_base.h>
#include <arrow/record_batch.h>
#include <arrow/type.h>
#include <arrow/util/key_value_metadata.h>
#include <TBufferFile.h>

#include <memory>
#include <utility>

O2_DECLARE_DYNAMIC_LOG(tabletree_helpers);

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
    int idealBasketSize = 1024 + reader->fieldSize() * reader->columnEntries(); // minimal additional size needed, otherwise we get 2 baskets
    int basketSize = std::max(32000, idealBasketSize);                          // keep a minimum value
    // std::cout << "Setting baskets size for " << reader->branchName() << " to " << basketSize << " =  1024 + "
    //           << reader->fieldSize() << " * " << reader->columnEntries() << ". mRows was " << mRows << std::endl;
    mTree->SetBasketSize(reader->branchName(), basketSize);
    // If it starts with fIndexArray, also set the size branch basket size
    if (strncmp(reader->branchName(), "fIndexArray", strlen("fIndexArray")) == 0) {
      std::string sizeBranch = reader->branchName();
      sizeBranch += "_size";
      //  std::cout << "Setting baskets size for " << sizeBranch << " to " << basketSize << " =  1024 + "
      //            << reader->fieldSize() << " * " << reader->columnEntries() << ". mRows was " << mRows << std::endl;
      // One int per array to keep track of the size
      int idealBasketSize = 4 * mRows + 1024 + reader->fieldSize() * reader->columnEntries(); // minimal additional size needed, otherwise we get 2 baskets
      int basketSize = std::max(32000, idealBasketSize);                                      // keep a minimum value
      mTree->SetBasketSize(sizeBranch.c_str(), basketSize);
      mTree->SetBasketSize(reader->branchName(), basketSize);
    }
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

namespace
{
struct BranchInfo {
  std::string name;
  TBranch* ptr;
  bool mVLA;
};
} // namespace

} // namespace o2::framework

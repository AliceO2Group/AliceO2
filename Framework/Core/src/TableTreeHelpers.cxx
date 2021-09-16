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
#include "Framework/RuntimeError.h"

#include <arrow/type_traits.h>
#include <arrow/util/key_value_metadata.h>
#include <TBufferFile.h>

namespace o2::framework
{

// -----------------------------------------------------------------------------
// TreeToTable allows to fill the contents of a given TTree to an arrow::Table
//  ColumnIterator is used by TreeToTable
//
// To copy the contents of a tree tr to a table ta do:
//  . TreeToTable t2t();
//  . t2t.addColumns(tr, vector{name1, name2, ..}); // optional 2nd arg
//  . auto ta = t2t.read();
//
// .............................................................................
ColumnToBranchBase::ColumnToBranchBase(arrow::ChunkedArray* column, arrow::Field* field, int size)
  : mBranchName{field->name()},
    mColumn{column},
    listSize{size}
{
}

BranchToColumnBase::BranchToColumnBase(TBranch* branch, const char* name, EDataType type, int listSize)
  : mBranch{branch},
    mColumnName{name},
    mType{type},
    mListSize{listSize}
{
}

TableToTree::TableToTree(TFile* file, const char* treename)
{
  mTree = static_cast<TTree*>(file->Get(treename));
  if (mTree == nullptr) {
    std::string treeName(treename);
    auto pos = treeName.find_first_of('/');
    if (pos != std::string::npos) {
      file->cd(treeName.substr(0, pos).c_str());
      treeName = treeName.substr(pos + 1, std::string::npos);
    }
    mTree = new TTree(treeName.c_str(), treeName.c_str());
  }
}

void TableToTree::addBranches(arrow::Table* table)
{
  mRows = table->num_rows();
  auto columns = table->columns();
  auto fields = table->schema()->fields();
  assert(columns.size() == fields.size());
  for (auto i = 0u; i < columns.size(); ++i) {
    addBranch(columns[i].get(), fields[i].get());
  }
}

void TableToTree::addBranch(arrow::ChunkedArray* column, arrow::Field* field)
{
  if (mRows == 0) {
    mRows = column->length();
  } else {
    if (mRows != column->length()) {
      throw runtime_error_f("Adding incompatible column with size %d (num rows = %d)", column->length(), mRows);
    }
  }
  switch (field->type()->id()) {
    case arrow::Type::type::BOOL:
      mColumnReaders.emplace_back(new ColumnToBranch<bool>(mTree, column, field));
      break;
    case arrow::Type::type::UINT8:
      mColumnReaders.emplace_back(new ColumnToBranch<uint8_t>(mTree, column, field));
      break;
    case arrow::Type::type::UINT16:
      mColumnReaders.emplace_back(new ColumnToBranch<uint16_t>(mTree, column, field));
      break;
    case arrow::Type::type::UINT32:
      mColumnReaders.emplace_back(new ColumnToBranch<uint32_t>(mTree, column, field));
      break;
    case arrow::Type::type::UINT64:
      mColumnReaders.emplace_back(new ColumnToBranch<uint64_t>(mTree, column, field));
      break;
    case arrow::Type::type::INT8:
      mColumnReaders.emplace_back(new ColumnToBranch<int8_t>(mTree, column, field));
      break;
    case arrow::Type::type::INT16:
      mColumnReaders.emplace_back(new ColumnToBranch<int16_t>(mTree, column, field));
      break;
    case arrow::Type::type::INT32:
      mColumnReaders.emplace_back(new ColumnToBranch<int>(mTree, column, field));
      break;
    case arrow::Type::type::INT64:
      mColumnReaders.emplace_back(new ColumnToBranch<int64_t>(mTree, column, field));
      break;
    case arrow::Type::type::FLOAT:
      mColumnReaders.emplace_back(new ColumnToBranch<float>(mTree, column, field));
      break;
    case arrow::Type::type::DOUBLE:
      mColumnReaders.emplace_back(new ColumnToBranch<double>(mTree, column, field));
      break;
    case arrow::Type::type::FIXED_SIZE_LIST: {
      auto size = static_cast<const arrow::FixedSizeListType*>(field->type().get())->list_size();
      switch (field->type()->field(0)->type()->id()) {
        case arrow::Type::type::BOOL:
          mColumnReaders.emplace_back(new ColumnToBranch<bool*>(mTree, column, field, size));
          break;
        case arrow::Type::type::INT32:
          mColumnReaders.emplace_back(new ColumnToBranch<int*>(mTree, column, field, size));
          break;
        case arrow::Type::type::FLOAT:
          mColumnReaders.emplace_back(new ColumnToBranch<float*>(mTree, column, field, size));
          break;
        case arrow::Type::type::DOUBLE:
          mColumnReaders.emplace_back(new ColumnToBranch<double*>(mTree, column, field, size));
          break;
        default:
          throw runtime_error_f("Unsupported array column type for %s", field->name().c_str());
      }
    };
      break;
    case arrow::Type::type::LIST:
      throw runtime_error("Not implemented");
    default:
      throw runtime_error_f("Unsupported column type for %s", field->name().c_str());
  }
}

TTree* TableToTree::write()
{
  int64_t mRow = 0;
  bool writable = (mTree->GetNbranches() > 0) && (mRows > 0);
  if (writable) {
    while (mRow < mRows) {
      for (auto& reader : mColumnReaders) {
        reader->at(&mRow);
      }
      mTree->Fill();
      ++mRow;
    }
  }
  mTree->Write("", TObject::kOverwrite);
  return mTree;
}

void TreeToTable::addColumns(TTree* tree, std::vector<const char*>&& names)
{
  auto branches = tree->GetListOfBranches();
  auto n = branches->GetEntries();
  if (n == 0) {
    throw runtime_error("Tree has no branches");
  }
  if (names.empty()) {

    for (auto i = 0; i < n; ++i) {
      auto branch = static_cast<TBranch*>(branches->At(i));
      AddReader(branch, branch->GetName());
    }
  } else {
    for (auto i = 0; i < n; ++i) {
      auto branch = static_cast<TBranch*>(branches->At(i));
      auto lookup = std::find_if(names.begin(), names.end(), [&](auto name) { return name == branch->GetName(); });
      if (lookup != names.end()) {
        AddReader(branch, branch->GetName());
      }
      if (mBranchReaders.size() != names.size()) {
        LOGF(warn, "Not all requested columns were found in the tree");
      }
    }
  }
  if (mBranchReaders.empty()) {
    throw runtime_error("No columns will be read");
  }
  tree->SetCacheSize(50000000);
  // FIXME: see https://github.com/root-project/root/issues/8962 and enable
  // again once fixed.
  //tree->SetClusterPrefetch(true);
  for (auto& reader : mBranchReaders) {
    tree->AddBranchToCache(reader->branch());
  }
  tree->StopCacheLearningPhase();
}

void TreeToTable::setLabel(const char* label)
{
  mTableLabel = label;
}

void TreeToTable::read()
{
  std::vector<std::shared_ptr<arrow::ChunkedArray>> columns;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (auto& reader : mBranchReaders) {
    static TBufferFile buffer{TBuffer::EMode::kWrite, 4 * 1024 * 1024};
    buffer.Reset();
    auto chunkAndField = reader->read(&buffer);
    columns.push_back(chunkAndField.first);
    fields.push_back(chunkAndField.second);
  }
  auto schema = std::make_shared<arrow::Schema>(fields, std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{mTableLabel}));
  mTable = arrow::Table::Make(schema, columns);
}

void TreeToTable::AddReader(TBranch* branch, const char* name)
{
  static TClass* cls;
  EDataType type;
  branch->GetExpectedType(cls, type);
  auto listSize = static_cast<TLeaf*>(branch->GetListOfLeaves()->At(0))->GetLenStatic();
  switch (type) {
    case EDataType::kBool_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<bool>>(branch, name, type, listSize));
      break;
    case EDataType::kUChar_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<uint8_t>>(branch, name, type, listSize));
      break;
    case EDataType::kUShort_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<uint16_t>>(branch, name, type, listSize));
      break;
    case EDataType::kUInt_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<uint32_t>>(branch, name, type, listSize));
      break;
    case EDataType::kULong64_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<uint64_t>>(branch, name, type, listSize));
      break;
    case EDataType::kChar_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<int8_t>>(branch, name, type, listSize));
      break;
    case EDataType::kShort_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<int16_t>>(branch, name, type, listSize));
      break;
    case EDataType::kInt_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<int32_t>>(branch, name, type, listSize));
      break;
    case EDataType::kLong64_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<int64_t>>(branch, name, type, listSize));
      break;
    case EDataType::kFloat_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<float>>(branch, name, type, listSize));
      break;
    case EDataType::kDouble_t:
      mBranchReaders.emplace_back(std::make_unique<BranchToColumn<double>>(branch, name, type, listSize));
      break;
    default:
      throw runtime_error("Unsupported branch type");
  }
}

} // namespace o2::framework

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
#include <stdexcept>
#include "Framework/Logger.h"

#include "arrow/type_traits.h"
#include <arrow/util/key_value_metadata.h>
#include <TBufferFile.h>

namespace o2::framework
{

namespace
{
// -----------------------------------------------------------------------------
// TreeToTable allows to fill the contents of a given TTree to an arrow::Table
//  ColumnIterator is used by TreeToTable
//
// To copy the contents of a tree tr to a table ta do:
//  . TreeToTable t2t(tr);
//  . t2t.addColumn(columnname1); t2t.addColumn(columnname2); ...
//    OR
//    t2t.addAllColumns();
//  . auto ta = t2t.process();
//
// .............................................................................
class ColumnIterator
{

 private:
  // all the possible arrow::TBuilder types
  arrow::FixedSizeListBuilder* mTableBuilder_list = nullptr;

  arrow::BooleanBuilder* mTableBuilder_o = nullptr;
  arrow::UInt8Builder* mTableBuilder_ub = nullptr;
  arrow::UInt16Builder* mTableBuilder_us = nullptr;
  arrow::UInt32Builder* mTableBuilder_ui = nullptr;
  arrow::UInt64Builder* mTableBuilder_ul = nullptr;
  arrow::Int8Builder* mTableBuilder_b = nullptr;
  arrow::Int16Builder* mTableBuilder_s = nullptr;
  arrow::Int32Builder* mTableBuilder_i = nullptr;
  arrow::Int64Builder* mTableBuilder_l = nullptr;
  arrow::FloatBuilder* mTableBuilder_f = nullptr;
  arrow::DoubleBuilder* mTableBuilder_d = nullptr;

  bool mStatus = false;
  EDataType mElementType;
  int64_t mNumberElements;
  const char* mColumnName;
  int mPos = 0;
  int mNumEntries = 0;
  TBranch* mBranch = nullptr;

  std::shared_ptr<arrow::Field> mField;
  std::shared_ptr<arrow::Array> mArray;

 public:
  ColumnIterator(TTree* reader, const char* colname);
  ~ColumnIterator();

  // has the iterator been properly initialized
  bool getStatus();

  // copy the contents of the associated branch to the arrow::TBuilder
  size_t push();

  // reserve enough space to push s elements without reallocating
  void reserve(size_t s);

  std::shared_ptr<arrow::Array> getArray() { return mArray; }
  std::shared_ptr<arrow::Field> getSchema() { return mField; }

  // finish the arrow::TBuilder
  // with this mArray is prepared to be used in arrow::Table::Make
  void finish();
};
} // namespace

ColumnToBranchBase::ColumnToBranchBase(arrow::ChunkedArray* column, arrow::Field* field, int size)
  : mBranchName{field->name()},
    mColumn{column},
    listSize{size}
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

// -----------------------------------------------------------------------------
#define MAKE_LIST_BUILDER(ElementType, NumElements)                \
  std::unique_ptr<arrow::ArrayBuilder> ValueBuilder;               \
  arrow::MemoryPool* MemoryPool = arrow::default_memory_pool();    \
  auto stat = MakeBuilder(MemoryPool, ElementType, &ValueBuilder); \
  mTableBuilder_list = new arrow::FixedSizeListBuilder(            \
    MemoryPool,                                                    \
    std::move(ValueBuilder),                                       \
    NumElements);

#define MAKE_FIELD(ElementType, NumElements)                                                         \
  if (NumElements == 1) {                                                                            \
    mField =                                                                                         \
      std::make_shared<arrow::Field>(mColumnName, ElementType);                                      \
  } else {                                                                                           \
    mField =                                                                                         \
      std::make_shared<arrow::Field>(mColumnName, arrow::fixed_size_list(ElementType, NumElements)); \
  }

#define MAKE_FIELD_AND_BUILDER(ElementCType, NumElements, Builder)                                                                            \
  MAKE_FIELD(arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::type_singleton(), NumElements);                                  \
  if (NumElements == 1) {                                                                                                                     \
    arrow::MemoryPool* MemoryPool = arrow::default_memory_pool();                                                                             \
    Builder = new arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::BuilderType(MemoryPool);                                    \
  } else {                                                                                                                                    \
    MAKE_LIST_BUILDER(arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::type_singleton(), NumElements);                         \
    Builder = static_cast<arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::BuilderType*>(mTableBuilder_list->value_builder()); \
  }

// is used in TreeToTable
ColumnIterator::ColumnIterator(TTree* tree, const char* colname)
{
  mBranch = tree->GetBranch(colname);
  mNumEntries = mBranch->GetEntries();

  if (!mBranch) {
    LOGP(WARNING, "Can not locate branch {}", colname);
    return;
  }
  mColumnName = colname;

  // type of the branch elements
  TClass* cl;
  mBranch->GetExpectedType(cl, mElementType);

  // currently only single-value or single-array branches are accepted
  // thus of the form e.g. alpha/D or alpha[5]/D
  // check if this is a single-value or single-array branch
  mNumberElements = 1;
  std::string branchTitle = mBranch->GetTitle();
  Int_t pos0 = branchTitle.find("[");
  Int_t pos1 = branchTitle.find("]");
  if (pos0 > 0 && pos1 > 0) {
    mNumberElements = atoi(branchTitle.substr(pos0 + 1, pos1 - pos0 - 1).c_str());
  }

  // initialize the TTreeReaderValue<T> / TTreeReaderArray<T>
  //            the corresponding arrow::TBuilder
  //            the column field
  // the TTreeReaderValue is incremented by reader->Next()
  // switch according to mElementType
  mStatus = true;

  if (mNumberElements == 1) {
    switch (mElementType) {
      case EDataType::kBool_t:
        MAKE_FIELD_AND_BUILDER(bool, 1, mTableBuilder_o);
        break;
      case EDataType::kUChar_t:
        MAKE_FIELD_AND_BUILDER(uint8_t, 1, mTableBuilder_ub);
        break;
      case EDataType::kUShort_t:
        MAKE_FIELD_AND_BUILDER(uint16_t, 1, mTableBuilder_us);
        break;
      case EDataType::kUInt_t:
        MAKE_FIELD_AND_BUILDER(uint32_t, 1, mTableBuilder_ui);
        break;
      case EDataType::kULong64_t:
        MAKE_FIELD_AND_BUILDER(uint64_t, 1, mTableBuilder_ul);
        break;
      case EDataType::kChar_t:
        MAKE_FIELD_AND_BUILDER(int8_t, 1, mTableBuilder_b);
        break;
      case EDataType::kShort_t:
        MAKE_FIELD_AND_BUILDER(int16_t, 1, mTableBuilder_s);
        break;
      case EDataType::kInt_t:
        MAKE_FIELD_AND_BUILDER(int32_t, 1, mTableBuilder_i);
        break;
      case EDataType::kLong64_t:
        MAKE_FIELD_AND_BUILDER(int64_t, 1, mTableBuilder_l);
        break;
      case EDataType::kFloat_t:
        MAKE_FIELD_AND_BUILDER(float, 1, mTableBuilder_f);
        break;
      case EDataType::kDouble_t:
        MAKE_FIELD_AND_BUILDER(double, 1, mTableBuilder_d);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  } else {
    switch (mElementType) {
      case EDataType::kBool_t:
        MAKE_FIELD_AND_BUILDER(bool, mNumberElements, mTableBuilder_o);
        break;
      case EDataType::kUChar_t:
        MAKE_FIELD_AND_BUILDER(uint8_t, mNumberElements, mTableBuilder_ub);
        break;
      case EDataType::kUShort_t:
        MAKE_FIELD_AND_BUILDER(uint16_t, mNumberElements, mTableBuilder_us);
        break;
      case EDataType::kUInt_t:
        MAKE_FIELD_AND_BUILDER(uint32_t, mNumberElements, mTableBuilder_ui);
        break;
      case EDataType::kULong64_t:
        MAKE_FIELD_AND_BUILDER(uint64_t, mNumberElements, mTableBuilder_ul);
        break;
      case EDataType::kChar_t:
        MAKE_FIELD_AND_BUILDER(int8_t, mNumberElements, mTableBuilder_b);
        break;
      case EDataType::kShort_t:
        MAKE_FIELD_AND_BUILDER(int16_t, mNumberElements, mTableBuilder_s);
        break;
      case EDataType::kInt_t:
        MAKE_FIELD_AND_BUILDER(int32_t, mNumberElements, mTableBuilder_i);
        break;
      case EDataType::kLong64_t:
        MAKE_FIELD_AND_BUILDER(int64_t, mNumberElements, mTableBuilder_l);
        break;
      case EDataType::kFloat_t:
        MAKE_FIELD_AND_BUILDER(float, mNumberElements, mTableBuilder_f);
        break;
      case EDataType::kDouble_t:
        MAKE_FIELD_AND_BUILDER(double, mNumberElements, mTableBuilder_d);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  }
}

ColumnIterator::~ColumnIterator()
{
  if (mTableBuilder_list) {
    delete mTableBuilder_list;
  } else {
    delete mTableBuilder_o;
    delete mTableBuilder_ub;
    delete mTableBuilder_us;
    delete mTableBuilder_ui;
    delete mTableBuilder_ul;
    delete mTableBuilder_b;
    delete mTableBuilder_s;
    delete mTableBuilder_i;
    delete mTableBuilder_l;
    delete mTableBuilder_f;
    delete mTableBuilder_d;
  }
};

bool ColumnIterator::getStatus()
{
  return mStatus;
}

void ColumnIterator::reserve(size_t s)
{
  arrow::Status stat;
  if (mNumberElements != 1) {
    stat = mTableBuilder_list->Reserve(s);
  }

  switch (mElementType) {
    case EDataType::kBool_t:
      stat = mTableBuilder_o->Reserve(s * mNumberElements);
      break;
    case EDataType::kUChar_t:
      stat = mTableBuilder_ub->Reserve(s * mNumberElements);
      break;
    case EDataType::kUShort_t:
      stat = mTableBuilder_us->Reserve(s * mNumberElements);
      break;
    case EDataType::kUInt_t:
      stat = mTableBuilder_ui->Reserve(s * mNumberElements);
      break;
    case EDataType::kULong64_t:
      stat = mTableBuilder_ul->Reserve(s * mNumberElements);
      break;
    case EDataType::kChar_t:
      stat = mTableBuilder_b->Reserve(s * mNumberElements);
      break;
    case EDataType::kShort_t:
      stat = mTableBuilder_s->Reserve(s * mNumberElements);
      break;
    case EDataType::kInt_t:
      stat = mTableBuilder_i->Reserve(s * mNumberElements);
      break;
    case EDataType::kLong64_t:
      stat = mTableBuilder_l->Reserve(s * mNumberElements);
      break;
    case EDataType::kFloat_t:
      stat = mTableBuilder_f->Reserve(s * mNumberElements);
      break;
    case EDataType::kDouble_t:
      stat = mTableBuilder_d->Reserve(s * mNumberElements);
      break;
    default:
      LOGP(FATAL, "Type {} not handled!", mElementType);
      break;
  }
}

template <typename T, typename Builder>
arrow::Status appendValues(Builder builder, TBuffer& buffer, int64_t size)
{
  return builder->AppendValues(reinterpret_cast<T const*>(buffer.GetCurrent()), size, nullptr);
}

size_t ColumnIterator::push()
{
  arrow::Status stat;

  static TBufferFile buffer{TBuffer::EMode::kWrite, 4 * 1024 * 1024};
  buffer.Reset();
  auto size = mBranch->GetBulkRead().GetBulkEntries(mPos, buffer);
  if (size < 0) {
    return 0;
  }
  if ((mPos + size) > mNumEntries) {
    size = mNumEntries - mPos;
  }
  mPos += size;

  // switch according to mElementType
  switch (mElementType) {
    case EDataType::kBool_t:
      stat = appendValues<unsigned char>(mTableBuilder_o, buffer, size * mNumberElements);
      break;
    case EDataType::kUChar_t:
      stat = appendValues<unsigned char>(mTableBuilder_ub, buffer, size * mNumberElements);
      break;
    case EDataType::kUShort_t:
      stat = appendValues<unsigned short>(mTableBuilder_us, buffer, size * mNumberElements);
      break;
    case EDataType::kUInt_t:
      stat = appendValues<unsigned int>(mTableBuilder_ui, buffer, size * mNumberElements);
      break;
    case EDataType::kULong64_t:
      stat = appendValues<uint64_t>(mTableBuilder_ul, buffer, size * mNumberElements);
      break;
    case EDataType::kChar_t:
      stat = appendValues<signed char>(mTableBuilder_b, buffer, size * mNumberElements);
      break;
    case EDataType::kShort_t:
      stat = appendValues<short>(mTableBuilder_s, buffer, size * mNumberElements);
      break;
    case EDataType::kInt_t:
      stat = appendValues<int>(mTableBuilder_i, buffer, size * mNumberElements);
      break;
    case EDataType::kLong64_t:
      stat = appendValues<int64_t>(mTableBuilder_l, buffer, size * mNumberElements);
      break;
    case EDataType::kFloat_t:
      stat = appendValues<float>(mTableBuilder_f, buffer, size * mNumberElements);
      break;
    case EDataType::kDouble_t:
      stat = appendValues<double>(mTableBuilder_d, buffer, size * mNumberElements);
      break;
    default:
      LOGP(FATAL, "Type {} not handled!", mElementType);
      break;
  }
  if (mNumberElements != 1) {
    stat = mTableBuilder_list->AppendValues(size);
  }
  return size;
}

void ColumnIterator::finish()
{
  arrow::Status stat;

  if (mNumberElements != 1) {
    stat = mTableBuilder_list->Finish(&mArray);
    return;
  }

  // switch according to mElementType
    switch (mElementType) {
      case EDataType::kBool_t:
        stat = mTableBuilder_o->Finish(&mArray);
        break;
      case EDataType::kUChar_t:
        stat = mTableBuilder_ub->Finish(&mArray);
        break;
      case EDataType::kUShort_t:
        stat = mTableBuilder_us->Finish(&mArray);
        break;
      case EDataType::kUInt_t:
        stat = mTableBuilder_ui->Finish(&mArray);
        break;
      case EDataType::kULong64_t:
        stat = mTableBuilder_ul->Finish(&mArray);
        break;
      case EDataType::kChar_t:
        stat = mTableBuilder_b->Finish(&mArray);
        break;
      case EDataType::kShort_t:
        stat = mTableBuilder_s->Finish(&mArray);
        break;
      case EDataType::kInt_t:
        stat = mTableBuilder_i->Finish(&mArray);
        break;
      case EDataType::kLong64_t:
        stat = mTableBuilder_l->Finish(&mArray);
        break;
      case EDataType::kFloat_t:
        stat = mTableBuilder_f->Finish(&mArray);
        break;
      case EDataType::kDouble_t:
        stat = mTableBuilder_d->Finish(&mArray);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
}

void TreeToTable::setLabel(const char* label)
{
  mTableLabel = label;
}

void TreeToTable::addColumn(const char* colname)
{
  mColumnNames.push_back(colname);
}

bool TreeToTable::addAllColumns(TTree* tree)
{
  auto branchList = tree->GetListOfBranches();

  // loop over branches
  if (branchList->IsEmpty()) {
    return false;
  }
  for (Int_t ii = 0; ii < branchList->GetEntries(); ii++) {
    auto br = (TBranch*)branchList->At(ii);

    // IMPROVE: make sure that a column is not added more than one time
    mColumnNames.push_back(br->GetName());
  }
  return true;
}

void TreeToTable::fill(TTree* tree)
{
  std::vector<std::unique_ptr<ColumnIterator>> columnIterators;

  tree->SetCacheSize(50000000);
  // FIXME: see https://github.com/root-project/root/issues/8962 and enable
  // again once fixed.
  //tree->SetClusterPrefetch(true);
  for (auto&& columnName : mColumnNames) {
    tree->AddBranchToCache(columnName.c_str(), true);
    auto colit = std::make_unique<ColumnIterator>(tree, columnName.c_str());
    auto stat = colit->getStatus();
    if (!stat) {
      throw std::runtime_error("Unable to convert column " + columnName);
    }
    columnIterators.push_back(std::move(colit));
  }
  tree->StopCacheLearningPhase();
  auto numEntries = tree->GetEntries();
  if (numEntries > 0) {

    for (size_t ci = 0; ci < columnIterators.size(); ++ci) {
      auto& column = columnIterators[ci];
      auto& columnName = mColumnNames[ci];
      column->reserve(numEntries);
      while (column->push() != 0) {
      }
    }
  }

  // prepare the elements needed to create the final table
  std::vector<std::shared_ptr<arrow::Array>> array_vector;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;
  for (auto&& colit : columnIterators) {
    colit->finish();
    array_vector.push_back(colit->getArray());
    schema_vector.push_back(colit->getSchema());
  }
  auto fields = std::make_shared<arrow::Schema>(schema_vector, std::make_shared<arrow::KeyValueMetadata>(std::vector{std::string{"label"}}, std::vector{mTableLabel}));

  // create the final table
  // ta is of type std::shared_ptr<arrow::Table>
  mTable = (arrow::Table::Make(fields, array_vector));
}

std::shared_ptr<arrow::Table> TreeToTable::finalize()
{
  return mTable;
}

} // namespace o2::framework

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

// is used in TableToTree
BranchIterator::BranchIterator(TTree* tree, std::shared_ptr<arrow::ChunkedArray> col, std::shared_ptr<arrow::Field> field)
{
  mField = field.get();
  mBranchName = mField->name();

  mFieldType = mField->type()->id();
  mChunks = col->chunks();
  mNumberChuncs = mChunks.size();

  mLeaflistString = mBranchName;
  mElementType = mFieldType;
  mNumberElements = 1;
  if (mFieldType == arrow::Type::type::FIXED_SIZE_LIST) {

    // element type
    if (mField->type()->num_fields() <= 0) {
      LOGP(FATAL, "Field {} of type {} has no children!", mField->name(), mField->type()->ToString().c_str());
    }
    mElementType = mField->type()->field(0)->type()->id();
    // number of elements
    mNumberElements = static_cast<const arrow::FixedSizeListType*>(mField->type().get())->list_size();
    mLeaflistString += "[" + std::to_string(mNumberElements) + "]";
  }

  // initialize the branch
  mStatus = initBranch(tree);

  mCounterChunk = 0;
  mStatus &= initDataBuffer(mCounterChunk);
}

BranchIterator::~BranchIterator()
{
  delete mBranchBuffer;

  delete[] mVariable_o;
}

bool BranchIterator::getStatus()
{
  return mStatus;
}

bool BranchIterator::initBranch(TTree* tree)
{
  // try to find branch in tree
  mBranchPtr = tree->GetBranch(mBranchName.c_str());
  if (mBranchPtr) {
    return true;
  }

  // create new branch of given data type
  switch (mElementType) {
    case arrow::Type::type::BOOL:
      mLeaflistString += "/O";
      break;
    case arrow::Type::type::UINT8:
      mLeaflistString += "/b";
      break;
    case arrow::Type::type::UINT16:
      mLeaflistString += "/s";
      break;
    case arrow::Type::type::UINT32:
      mLeaflistString += "/i";
      break;
    case arrow::Type::type::UINT64:
      mLeaflistString += "/l";
      break;
    case arrow::Type::type::INT8:
      mLeaflistString += "/B";
      break;
    case arrow::Type::type::INT16:
      mLeaflistString += "/S";
      break;
    case arrow::Type::type::INT32:
      mLeaflistString += "/I";
      break;
    case arrow::Type::type::INT64:
      mLeaflistString += "/L";
      break;
    case arrow::Type::type::FLOAT:
      mLeaflistString += "/F";
      break;
    case arrow::Type::type::DOUBLE:
      mLeaflistString += "/D";
      break;
    default:
      LOGP(FATAL, "Type {} not handled!", mElementType);
      break;
  }

  mBranchPtr = tree->Branch(mBranchName.c_str(), mBranchBuffer, mLeaflistString.c_str());
  return mBranchPtr != nullptr;
}

bool BranchIterator::initDataBuffer(Int_t ib)
{

  auto chunkToUse = mChunks.at(ib);
  if (mFieldType == arrow::Type::type::FIXED_SIZE_LIST) {
    chunkToUse = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(chunkToUse)->values();
  }

  // reset actual row number
  mCounterRow = 0;

  // get next chunk of given data type mElementType
  switch (mElementType) {
    case arrow::Type::type::BOOL:
      if (!mVariable_o) {
        mVariable_o = new bool[mNumberElements];
      }
      mArray_o = std::dynamic_pointer_cast<arrow::BooleanArray>(chunkToUse);
      for (int ii = 0; ii < mNumberElements; ii++) {
        mVariable_o[ii] = (bool)mArray_o->Value(ii);
      }
      mValueBuffer = (void*)mVariable_o;
      break;
    case arrow::Type::type::UINT8:
      mVariable_ub = (uint8_t*)std::dynamic_pointer_cast<arrow::UInt8Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_ub;
      break;
    case arrow::Type::type::UINT16:
      mVariable_us = (uint16_t*)std::dynamic_pointer_cast<arrow::UInt16Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_us;
      break;
    case arrow::Type::type::UINT32:
      mVariable_ui = (uint32_t*)std::dynamic_pointer_cast<arrow::UInt32Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_ui;
      break;
    case arrow::Type::type::UINT64:
      mVariable_ul = (uint64_t*)std::dynamic_pointer_cast<arrow::UInt64Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_ul;
      break;
    case arrow::Type::type::INT8:
      mVariable_b = (int8_t*)std::dynamic_pointer_cast<arrow::Int8Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_b;
      break;
    case arrow::Type::type::INT16:
      mVariable_s = (int16_t*)std::dynamic_pointer_cast<arrow::Int16Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_s;
      break;
    case arrow::Type::type::INT32:
      mVariable_i = (int32_t*)std::dynamic_pointer_cast<arrow::Int32Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_i;
      break;
    case arrow::Type::type::INT64:
      mVariable_l = (int64_t*)std::dynamic_pointer_cast<arrow::Int64Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_l;
      break;
    case arrow::Type::type::FLOAT:
      mVariable_f = (Float_t*)std::dynamic_pointer_cast<arrow::FloatArray>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_f;
      break;
    case arrow::Type::type::DOUBLE:
      mVariable_d = (double*)std::dynamic_pointer_cast<arrow::DoubleArray>(chunkToUse)->raw_values();
      mValueBuffer = (void*)mVariable_d;
    default:
      break;
      LOGP(FATAL, "Type {} not handled!", mElementType);
      break;
  }
  mBranchPtr->SetAddress(mValueBuffer);

  // reset number of rows mNumberRows and row counter mCounterRow
  mNumberRows = mChunks.at(ib)->length();

  return true;
}

bool BranchIterator::push()
{
  // increment row counter
  mCounterRow++;

  // mCounterChunk and mCounterRow contain the current chunk and row
  // return the next element if available
  if (mCounterRow >= mNumberRows) {
    mCounterChunk++;
    if (mCounterChunk < mNumberChuncs) {
      initDataBuffer(mCounterChunk);
    } else {
      // end of data buffer reached
      return false;
    }
  } else {
    switch (mElementType) {
      case arrow::Type::type::BOOL:
        for (int ii = 0; ii < mNumberElements; ii++) {
          mVariable_o[ii] = (bool)mArray_o->Value(mCounterRow * mNumberElements + ii);
        }
        mValueBuffer = (void*)mVariable_o;
        break;
      case arrow::Type::type::UINT8:
        mVariable_ub += mNumberElements;
        mValueBuffer = (void*)mVariable_ub;
        break;
      case arrow::Type::type::UINT16:
        mVariable_us += mNumberElements;
        mValueBuffer = (void*)mVariable_us;
        break;
      case arrow::Type::type::UINT32:
        mVariable_ui += mNumberElements;
        mValueBuffer = (void*)mVariable_ui;
        break;
      case arrow::Type::type::UINT64:
        mVariable_ul += mNumberElements;
        mValueBuffer = (void*)mVariable_ul;
        break;
      case arrow::Type::type::INT8:
        mVariable_b += mNumberElements;
        mValueBuffer = (void*)mVariable_b;
        break;
      case arrow::Type::type::INT16:
        mVariable_s += mNumberElements;
        mValueBuffer = (void*)mVariable_s;
        break;
      case arrow::Type::type::INT32:
        mVariable_i += mNumberElements;
        mValueBuffer = (void*)mVariable_i;
        break;
      case arrow::Type::type::INT64:
        mVariable_l += mNumberElements;
        mValueBuffer = (void*)mVariable_l;
        break;
      case arrow::Type::type::FLOAT:
        mVariable_f += mNumberElements;
        mValueBuffer = (void*)mVariable_f;
        break;
      case arrow::Type::type::DOUBLE:
        mVariable_d += mNumberElements;
        mValueBuffer = (void*)mVariable_d;
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  }
  mBranchPtr->SetAddress(mValueBuffer);

  return true;
}

TableToTree::TableToTree(std::shared_ptr<arrow::Table> table,
                         TFile* file,
                         const char* treename)
{
  mTable = table;

  // try to get the tree
  mTreePtr = (TTree*)file->Get(treename);

  // create the tree if it does not exist already
  if (!mTreePtr) {
    // does treename containe folder?
    std::string treeName(treename);
    auto pos = treeName.find_first_of("/");
    if (pos != std::string::npos) {
      file->cd(treeName.substr(0, pos).c_str());
    }
    treeName = treeName.substr(pos + 1, std::string::npos);
    mTreePtr = new TTree(treeName.c_str(), treeName.c_str());
  }
}

bool TableToTree::addBranch(std::shared_ptr<arrow::ChunkedArray> col, std::shared_ptr<arrow::Field> field)
{
  auto brit = std::make_unique<BranchIterator>(mTreePtr, col, field);
  if (!brit->getStatus()) {
    return false;
  }
  mBranchIterators.emplace_back(std::move(brit));
  return true;
}

bool TableToTree::addAllBranches()
{

  bool status = mTable->num_columns() > 0;

  for (auto ii = 0; ii < mTable->num_columns(); ii++) {
    status &= addBranch(mTable->column(ii), mTable->schema()->field(ii));
  }

  return status;
}

TTree* TableToTree::process()
{

  bool togo = mTreePtr->GetNbranches() > 0;
  while (togo) {
    // fill the tree
    mTreePtr->Fill();

    // update the branches
    for (auto& brit : mBranchIterators) {
      togo &= brit->push();
    }
  }
  mTreePtr->Write("", TObject::kOverwrite);

  return mTreePtr;
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
  tree->SetClusterPrefetch(true);
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

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/TableTreeHelpers.h"
#include "Framework/Logger.h"

#include "arrow/type_traits.h"

namespace o2
{
namespace framework
{

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
    if (mField->type()->num_children() <= 0) {
      LOGP(FATAL, "Field {} of type {} has no children!", mField->name(), mField->type()->ToString().c_str());
    }
    mElementType = mField->type()->child(0)->type()->id();

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

  delete mVariable_o;
  delete mVariable_ub;
  delete mVariable_us;
  delete mVariable_ui;
  delete mVariable_ul;
  delete mVariable_s;
  delete mVariable_i;
  delete mVariable_l;
  delete mVariable_f;
  delete mVariable_d;
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
  if (mBranchPtr) {
    return true;
  } else {
    return false;
  }
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
        mVariable_o = new bool(mNumberElements);
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
    mTreePtr = new TTree(treename, treename);
  }
}

TableToTree::~TableToTree()
{
  // clean up branch iterators
  mBranchIterators.clear();
}

bool TableToTree::addBranch(std::shared_ptr<arrow::ChunkedArray> col, std::shared_ptr<arrow::Field> field)
{
  BranchIterator* brit = new BranchIterator(mTreePtr, col, field);
  if (brit->getStatus()) {
    mBranchIterators.push_back(brit);
  }

  return brit->getStatus();
}

bool TableToTree::addAllBranches()
{

  bool status = mTable->num_columns() > 0;
  for (auto ii = 0; ii < mTable->num_columns(); ii++) {
    BranchIterator* brit =
      new BranchIterator(mTreePtr, mTable->column(ii), mTable->schema()->field(ii));
    if (brit->getStatus()) {
      mBranchIterators.push_back(brit);
    } else {
      status = false;
    }
  }

  return status;
}

TTree* TableToTree::process()
{

  bool togo = true;
  while (togo) {
    // fill the tree
    mTreePtr->Fill();

    // update the branches
    for (auto brit : mBranchIterators) {
      togo &= brit->push();
    }
  }
  mTreePtr->Write();

  return mTreePtr;
}

// -----------------------------------------------------------------------------
#define MAKE_LIST_BUILDER(ElementType, NumElements)                   \
  std::unique_ptr<arrow::ArrayBuilder> ValueBuilder;                  \
  arrow::MemoryPool* MemoryPool = arrow::default_memory_pool();       \
  auto stat = MakeBuilder(MemoryPool, ElementType, &ValueBuilder);    \
  mTableBuilder_list = std::make_shared<arrow::FixedSizeListBuilder>( \
    MemoryPool,                                                       \
    std::move(ValueBuilder),                                          \
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
ColumnIterator::ColumnIterator(TTreeReader* reader, const char* colname)
{

  // find branch
  auto tree = reader->GetTree();
  if (!tree) {
    LOGP(FATAL, "Can not locate tree!");
    return;
  }
  //tree->Print();
  auto br = tree->GetBranch(colname);
  if (!br) {
    LOGP(FATAL, "Can not locate branch {}", colname);
    return;
  }
  mColumnName = colname;

  // type of the branch elements
  TClass* cl;
  br->GetExpectedType(cl, mElementType);

  // currently only single-value or single-array branches are accepted
  // thus of the form e.g. alpha/D or alpha[5]/D
  // check if this is a single-value or single-array branch
  mNumberElements = 1;
  std::string branchTitle = br->GetTitle();
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
        mReaderValue_o = new TTreeReaderValue<bool>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(bool, 1, mTableBuilder_o);
        break;
      case EDataType::kUChar_t:
        mReaderValue_ub = new TTreeReaderValue<uint8_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint8_t, 1, mTableBuilder_ub);
        break;
      case EDataType::kUShort_t:
        mReaderValue_us = new TTreeReaderValue<uint16_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint16_t, 1, mTableBuilder_us);
        break;
      case EDataType::kUInt_t:
        mReaderValue_ui = new TTreeReaderValue<uint32_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint32_t, 1, mTableBuilder_ui);
        break;
      case EDataType::kULong64_t:
        mReaderValue_ul = new TTreeReaderValue<uint64_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint64_t, 1, mTableBuilder_ul);
        break;
      case EDataType::kChar_t:
        mReaderValue_b = new TTreeReaderValue<int8_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int8_t, 1, mTableBuilder_b);
        break;
      case EDataType::kShort_t:
        mReaderValue_s = new TTreeReaderValue<int16_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int16_t, 1, mTableBuilder_s);
        break;
      case EDataType::kInt_t:
        mReaderValue_i = new TTreeReaderValue<int32_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int32_t, 1, mTableBuilder_i);
        break;
      case EDataType::kLong64_t:
        mReaderValue_l = new TTreeReaderValue<int64_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int64_t, 1, mTableBuilder_l);
        break;
      case EDataType::kFloat_t:
        mReaderValue_f = new TTreeReaderValue<float>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(float, 1, mTableBuilder_f);
        break;
      case EDataType::kDouble_t:
        mReaderValue_d = new TTreeReaderValue<double>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(double, 1, mTableBuilder_d);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  } else {
    switch (mElementType) {
      case EDataType::kBool_t:
        mReaderArray_o = new TTreeReaderArray<bool>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(bool, mNumberElements, mTableBuilder_o);
        break;
      case EDataType::kUChar_t:
        mReaderArray_ub = new TTreeReaderArray<uint8_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint8_t, mNumberElements, mTableBuilder_ub);
        break;
      case EDataType::kUShort_t:
        mReaderArray_us = new TTreeReaderArray<uint16_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint16_t, mNumberElements, mTableBuilder_us);
        break;
      case EDataType::kUInt_t:
        mReaderArray_ui = new TTreeReaderArray<uint32_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint32_t, mNumberElements, mTableBuilder_ui);
        break;
      case EDataType::kULong64_t:
        mReaderArray_ul = new TTreeReaderArray<uint64_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint64_t, mNumberElements, mTableBuilder_ul);
        break;
      case EDataType::kChar_t:
        mReaderArray_b = new TTreeReaderArray<int8_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int8_t, mNumberElements, mTableBuilder_b);
        break;
      case EDataType::kShort_t:
        mReaderArray_s = new TTreeReaderArray<int16_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int16_t, mNumberElements, mTableBuilder_s);
        break;
      case EDataType::kInt_t:
        mReaderArray_i = new TTreeReaderArray<int32_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int32_t, mNumberElements, mTableBuilder_i);
        break;
      case EDataType::kLong64_t:
        mReaderArray_l = new TTreeReaderArray<int64_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int64_t, mNumberElements, mTableBuilder_l);
        break;
      case EDataType::kFloat_t:
        mReaderArray_f = new TTreeReaderArray<float>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(float, mNumberElements, mTableBuilder_f);
        break;
      case EDataType::kDouble_t:
        mReaderArray_d = new TTreeReaderArray<double>(*reader, mColumnName);
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
  // delete all pointers
  delete mReaderValue_o;
  delete mReaderValue_ub;
  delete mReaderValue_us;
  delete mReaderValue_ui;
  delete mReaderValue_ul;
  delete mReaderValue_b;
  delete mReaderValue_s;
  delete mReaderValue_i;
  delete mReaderValue_l;
  delete mReaderValue_f;
  delete mReaderValue_d;

  delete mReaderArray_o;
  delete mReaderArray_ub;
  delete mReaderArray_us;
  delete mReaderArray_ui;
  delete mReaderArray_ul;
  delete mReaderArray_b;
  delete mReaderArray_s;
  delete mReaderArray_i;
  delete mReaderArray_l;
  delete mReaderArray_f;
  delete mReaderArray_d;
};

bool ColumnIterator::getStatus()
{
  return mStatus;
}

void ColumnIterator::push()
{
  arrow::Status stat;

  // switch according to mElementType
  if (mNumberElements == 1) {
    switch (mElementType) {
      case EDataType::kBool_t:
        stat = mTableBuilder_o->Append((bool)**mReaderValue_o);
        break;
      case EDataType::kUChar_t:
        stat = mTableBuilder_ub->Append(**mReaderValue_ub);
        break;
      case EDataType::kUShort_t:
        stat = mTableBuilder_us->Append(**mReaderValue_us);
        break;
      case EDataType::kUInt_t:
        stat = mTableBuilder_ui->Append(**mReaderValue_ui);
        break;
      case EDataType::kULong64_t:
        stat = mTableBuilder_ul->Append(**mReaderValue_ul);
        break;
      case EDataType::kChar_t:
        stat = mTableBuilder_b->Append(**mReaderValue_b);
        break;
      case EDataType::kShort_t:
        stat = mTableBuilder_s->Append(**mReaderValue_s);
        break;
      case EDataType::kInt_t:
        stat = mTableBuilder_i->Append(**mReaderValue_i);
        break;
      case EDataType::kLong64_t:
        stat = mTableBuilder_l->Append(**mReaderValue_l);
        break;
      case EDataType::kFloat_t:
        stat = mTableBuilder_f->Append(**mReaderValue_f);
        break;
      case EDataType::kDouble_t:
        stat = mTableBuilder_d->Append(**mReaderValue_d);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  } else {
    stat = mTableBuilder_list->AppendValues(1);
    switch (mElementType) {
      case EDataType::kBool_t:
        stat &= mTableBuilder_o->AppendValues((uint8_t*)&((*mReaderArray_o)[0]), mNumberElements);
        break;
      case EDataType::kUChar_t:
        stat &= mTableBuilder_ub->AppendValues(&((*mReaderArray_ub)[0]), mNumberElements);
        break;
      case EDataType::kUShort_t:
        stat &= mTableBuilder_us->AppendValues(&((*mReaderArray_us)[0]), mNumberElements);
        break;
      case EDataType::kUInt_t:
        stat &= mTableBuilder_ui->AppendValues(&((*mReaderArray_ui)[0]), mNumberElements);
        break;
      case EDataType::kULong64_t:
        stat &= mTableBuilder_ul->AppendValues(&((*mReaderArray_ul)[0]), mNumberElements);
        break;
      case EDataType::kChar_t:
        stat &= mTableBuilder_b->AppendValues(&((*mReaderArray_b)[0]), mNumberElements);
        break;
      case EDataType::kShort_t:
        stat &= mTableBuilder_s->AppendValues(&((*mReaderArray_s)[0]), mNumberElements);
        break;
      case EDataType::kInt_t:
        stat &= mTableBuilder_i->AppendValues(&((*mReaderArray_i)[0]), mNumberElements);
        break;
      case EDataType::kLong64_t:
        stat &= mTableBuilder_l->AppendValues(&((*mReaderArray_l)[0]), mNumberElements);
        break;
      case EDataType::kFloat_t:
        stat &= mTableBuilder_f->AppendValues(&((*mReaderArray_f)[0]), mNumberElements);
        break;
      case EDataType::kDouble_t:
        stat &= mTableBuilder_d->AppendValues(&((*mReaderArray_d)[0]), mNumberElements);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  }
}

void ColumnIterator::finish()
{
  arrow::Status stat;

  // switch according to mElementType
  if (mNumberElements == 1) {
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
  } else {
    stat = mTableBuilder_list->Finish(&mArray);
  }
}

TreeToTable::TreeToTable(TTree* tree)
{
  // initialize the TTreeReader
  mTreeReader = new TTreeReader(tree);
}

TreeToTable::~TreeToTable()
{
  delete mTreeReader;
};

bool TreeToTable::addColumn(const char* colname)
{
  auto colit = std::make_shared<ColumnIterator>(mTreeReader, colname);
  auto stat = colit->getStatus();
  if (stat) {
    mColumnIterators.push_back(std::move(colit));
  }

  return stat;
}

bool TreeToTable::addAllColumns()
{
  // get a list of column names
  auto tree = mTreeReader->GetTree();
  if (!tree) {
    LOGP(FATAL, "Tree not found!");
    return false;
  }
  auto branchList = tree->GetListOfBranches();

  // loop over branches
  bool status = !branchList->IsEmpty();
  for (Int_t ii = 0; ii < branchList->GetEntries(); ii++) {
    auto br = (TBranch*)branchList->At(ii);

    // IMPROVE: make sure that a column is not added more than one time
    auto colit = std::make_shared<ColumnIterator>(mTreeReader, br->GetName());
    if (colit->getStatus()) {
      mColumnIterators.push_back(std::move(colit));
    } else {
      status = false;
    }
  }

  return status;
}

void TreeToTable::push()
{
  for (auto colit : mColumnIterators) {
    colit->push();
  }
}

void TreeToTable::fill()
{
  // copy all values from the tree to the table builders
  mTreeReader->Restart();
  while (mTreeReader->Next()) {
    push();
  }
}

std::shared_ptr<arrow::Table> TreeToTable::finalize()
{
  // prepare the elements needed to create the final table
  std::vector<std::shared_ptr<arrow::Array>> array_vector;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;
  for (auto colit : mColumnIterators) {
    colit->finish();
    array_vector.push_back(colit->getArray());
    schema_vector.push_back(colit->getSchema());
  }
  auto fields = std::make_shared<arrow::Schema>(schema_vector);

  // create the final table
  // ta is of type std::shared_ptr<arrow::Table>
  auto table = (arrow::Table::Make(fields, array_vector));

  return table;
}

std::shared_ptr<arrow::Table> TreeToTable::process()
{
  // do the looping with the TTreeReader
  fill();

  // create the table
  return finalize();
}

} // namespace framework
} // namespace o2

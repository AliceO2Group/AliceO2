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
#include <stdexcept>
#include "Framework/Logger.h"

#include "arrow/type_traits.h"

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
  // all the possible TTreeReaderValue<T> types
  TTreeReaderValue<bool>* mReaderValue_o = nullptr;
  TTreeReaderValue<uint8_t>* mReaderValue_ub = nullptr;
  TTreeReaderValue<uint16_t>* mReaderValue_us = nullptr;
  TTreeReaderValue<uint32_t>* mReaderValue_ui = nullptr;
  TTreeReaderValue<ULong64_t>* mReaderValue_ul = nullptr;
  TTreeReaderValue<int8_t>* mReaderValue_b = nullptr;
  TTreeReaderValue<int16_t>* mReaderValue_s = nullptr;
  TTreeReaderValue<int32_t>* mReaderValue_i = nullptr;
  TTreeReaderValue<int64_t>* mReaderValue_l = nullptr;
  TTreeReaderValue<float>* mReaderValue_f = nullptr;
  TTreeReaderValue<double>* mReaderValue_d = nullptr;

  // all the possible TTreeReaderArray<T> types
  TTreeReaderArray<bool>* mReaderArray_o = nullptr;
  TTreeReaderArray<uint8_t>* mReaderArray_ub = nullptr;
  TTreeReaderArray<uint16_t>* mReaderArray_us = nullptr;
  TTreeReaderArray<uint32_t>* mReaderArray_ui = nullptr;
  TTreeReaderArray<uint64_t>* mReaderArray_ul = nullptr;
  TTreeReaderArray<int8_t>* mReaderArray_b = nullptr;
  TTreeReaderArray<int16_t>* mReaderArray_s = nullptr;
  TTreeReaderArray<int32_t>* mReaderArray_i = nullptr;
  TTreeReaderArray<int64_t>* mReaderArray_l = nullptr;
  TTreeReaderArray<float>* mReaderArray_f = nullptr;
  TTreeReaderArray<double>* mReaderArray_d = nullptr;

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

  std::shared_ptr<arrow::Field> mField;
  std::shared_ptr<arrow::Array> mArray;

 public:
  ColumnIterator(TTreeReader& reader, const char* colname);
  ~ColumnIterator();

  // has the iterator been properly initialized
  bool getStatus();

  // copy the TTreeReaderValue to the arrow::TBuilder
  void push();

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

  bool togo = mTreePtr->GetNbranches() > 0;
  while (togo) {
    // fill the tree
    mTreePtr->Fill();

    // update the branches
    for (auto brit : mBranchIterators) {
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
ColumnIterator::ColumnIterator(TTreeReader& reader, const char* colname)
{

  // find branch
  auto tree = reader.GetTree();
  if (!tree) {
    LOGP(FATAL, "Can not locate tree!");
    return;
  }

  auto br = tree->GetBranch(colname);
  if (!br) {
    LOGP(WARNING, "Can not locate branch {}", colname);
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
        mReaderValue_o = new TTreeReaderValue<bool>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(bool, 1, mTableBuilder_o);
        break;
      case EDataType::kUChar_t:
        mReaderValue_ub = new TTreeReaderValue<uint8_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint8_t, 1, mTableBuilder_ub);
        break;
      case EDataType::kUShort_t:
        mReaderValue_us = new TTreeReaderValue<uint16_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint16_t, 1, mTableBuilder_us);
        break;
      case EDataType::kUInt_t:
        mReaderValue_ui = new TTreeReaderValue<uint32_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint32_t, 1, mTableBuilder_ui);
        break;
      case EDataType::kULong64_t:
        mReaderValue_ul = new TTreeReaderValue<ULong64_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint64_t, 1, mTableBuilder_ul);
        break;
      case EDataType::kChar_t:
        mReaderValue_b = new TTreeReaderValue<int8_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int8_t, 1, mTableBuilder_b);
        break;
      case EDataType::kShort_t:
        mReaderValue_s = new TTreeReaderValue<int16_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int16_t, 1, mTableBuilder_s);
        break;
      case EDataType::kInt_t:
        mReaderValue_i = new TTreeReaderValue<int32_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int32_t, 1, mTableBuilder_i);
        break;
      case EDataType::kLong64_t:
        mReaderValue_l = new TTreeReaderValue<int64_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int64_t, 1, mTableBuilder_l);
        break;
      case EDataType::kFloat_t:
        mReaderValue_f = new TTreeReaderValue<float>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(float, 1, mTableBuilder_f);
        break;
      case EDataType::kDouble_t:
        mReaderValue_d = new TTreeReaderValue<double>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(double, 1, mTableBuilder_d);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  } else {
    switch (mElementType) {
      case EDataType::kBool_t:
        mReaderArray_o = new TTreeReaderArray<bool>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(bool, mNumberElements, mTableBuilder_o);
        break;
      case EDataType::kUChar_t:
        mReaderArray_ub = new TTreeReaderArray<uint8_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint8_t, mNumberElements, mTableBuilder_ub);
        break;
      case EDataType::kUShort_t:
        mReaderArray_us = new TTreeReaderArray<uint16_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint16_t, mNumberElements, mTableBuilder_us);
        break;
      case EDataType::kUInt_t:
        mReaderArray_ui = new TTreeReaderArray<uint32_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint32_t, mNumberElements, mTableBuilder_ui);
        break;
      case EDataType::kULong64_t:
        mReaderArray_ul = new TTreeReaderArray<uint64_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint64_t, mNumberElements, mTableBuilder_ul);
        break;
      case EDataType::kChar_t:
        mReaderArray_b = new TTreeReaderArray<int8_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int8_t, mNumberElements, mTableBuilder_b);
        break;
      case EDataType::kShort_t:
        mReaderArray_s = new TTreeReaderArray<int16_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int16_t, mNumberElements, mTableBuilder_s);
        break;
      case EDataType::kInt_t:
        mReaderArray_i = new TTreeReaderArray<int32_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int32_t, mNumberElements, mTableBuilder_i);
        break;
      case EDataType::kLong64_t:
        mReaderArray_l = new TTreeReaderArray<int64_t>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int64_t, mNumberElements, mTableBuilder_l);
        break;
      case EDataType::kFloat_t:
        mReaderArray_f = new TTreeReaderArray<float>(reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(float, mNumberElements, mTableBuilder_f);
        break;
      case EDataType::kDouble_t:
        mReaderArray_d = new TTreeReaderArray<double>(reader, mColumnName);
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

void ColumnIterator::push()
{
  arrow::Status stat;

  // switch according to mElementType
  if (mNumberElements == 1) {
    switch (mElementType) {
      case EDataType::kBool_t:
        mTableBuilder_o->UnsafeAppend((bool)**mReaderValue_o);
        break;
      case EDataType::kUChar_t:
        mTableBuilder_ub->UnsafeAppend(**mReaderValue_ub);
        break;
      case EDataType::kUShort_t:
        mTableBuilder_us->UnsafeAppend(**mReaderValue_us);
        break;
      case EDataType::kUInt_t:
        mTableBuilder_ui->UnsafeAppend(**mReaderValue_ui);
        break;
      case EDataType::kULong64_t:
        mTableBuilder_ul->UnsafeAppend(**mReaderValue_ul);
        break;
      case EDataType::kChar_t:
        mTableBuilder_b->UnsafeAppend(**mReaderValue_b);
        break;
      case EDataType::kShort_t:
        mTableBuilder_s->UnsafeAppend(**mReaderValue_s);
        break;
      case EDataType::kInt_t:
        mTableBuilder_i->UnsafeAppend(**mReaderValue_i);
        break;
      case EDataType::kLong64_t:
        mTableBuilder_l->UnsafeAppend(**mReaderValue_l);
        break;
      case EDataType::kFloat_t:
        mTableBuilder_f->UnsafeAppend(**mReaderValue_f);
        break;
      case EDataType::kDouble_t:
        mTableBuilder_d->UnsafeAppend(**mReaderValue_d);
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
  TTreeReader treeReader{tree};

  tree->SetCacheSize(50000000);
  tree->SetClusterPrefetch(true);
  for (auto&& columnName : mColumnNames) {
    tree->AddBranchToCache(columnName.c_str(), true);
    auto colit = std::make_unique<ColumnIterator>(treeReader, columnName.c_str());
    auto stat = colit->getStatus();
    if (!stat) {
      throw std::runtime_error("Unable to convert column " + columnName);
    }
    columnIterators.push_back(std::move(colit));
  }
  tree->StopCacheLearningPhase();
  
  auto numEntries = treeReader.GetEntries(true);
  if (numEntries > 0) {
    for (auto&& column : columnIterators) {
      column->reserve(numEntries);
    }
    // copy all values from the tree to the table builders
    treeReader.Restart();
    while (treeReader.Next()) {
      for (auto&& column : columnIterators) {
        column->push();
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
  auto fields = std::make_shared<arrow::Schema>(schema_vector);

  // create the final table
  // ta is of type std::shared_ptr<arrow::Table>
  mTable = (arrow::Table::Make(fields, array_vector));
}

std::shared_ptr<arrow::Table> TreeToTable::finalize()
{
  return mTable;
}

} // namespace o2::framework

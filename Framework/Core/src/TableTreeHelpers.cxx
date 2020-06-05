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

  delete var_ub;
  delete var_f;
  delete var_d;
  delete var_us;
  delete var_ui;
  delete var_ul;
  delete var_s;
  delete var_i;
  delete var_l;
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
    case arrow::Type::type::FLOAT:
      mLeaflistString += "/F";
      break;
    case arrow::Type::type::DOUBLE:
      mLeaflistString += "/D";
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
    case arrow::Type::type::INT16:
      mLeaflistString += "/S";
      break;
    case arrow::Type::type::INT32:
      mLeaflistString += "/I";
      break;
    case arrow::Type::type::INT64:
      mLeaflistString += "/L";
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
      var_o = std::dynamic_pointer_cast<arrow::BooleanArray>(chunkToUse);
      boolValueHolder = (Bool_t)var_o->Value(mCounterRow);
      mValueBuffer = (void*)&boolValueHolder;
      break;
    case arrow::Type::type::UINT8:
      var_ub = (UChar_t*)std::dynamic_pointer_cast<arrow::UInt8Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)var_ub;
      break;
    case arrow::Type::type::FLOAT:
      var_f = (Float_t*)std::dynamic_pointer_cast<arrow::FloatArray>(chunkToUse)->raw_values();
      mValueBuffer = (void*)var_f;
      break;
    case arrow::Type::type::DOUBLE:
      var_d = (Double_t*)std::dynamic_pointer_cast<arrow::DoubleArray>(chunkToUse)->raw_values();
      mValueBuffer = (void*)var_d;
      break;
    case arrow::Type::type::UINT16:
      var_us = (UShort_t*)std::dynamic_pointer_cast<arrow::UInt16Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)var_us;
      break;
    case arrow::Type::type::UINT32:
      var_ui = (UInt_t*)std::dynamic_pointer_cast<arrow::UInt32Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)var_ui;
      break;
    case arrow::Type::type::UINT64:
      var_ul = (long unsigned int*)std::dynamic_pointer_cast<arrow::UInt64Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)var_ul;
      break;
    case arrow::Type::type::INT16:
      var_s = (Short_t*)std::dynamic_pointer_cast<arrow::Int16Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)var_s;
      break;
    case arrow::Type::type::INT32:
      var_i = (Int_t*)std::dynamic_pointer_cast<arrow::Int32Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)var_i;
      break;
    case arrow::Type::type::INT64:
      var_l = (long int*)std::dynamic_pointer_cast<arrow::Int64Array>(chunkToUse)->raw_values();
      mValueBuffer = (void*)var_l;
      break;
    default:
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
        boolValueHolder = (Bool_t)var_o->Value(mCounterRow);
        mValueBuffer = (void*)&boolValueHolder;
        break;
      case arrow::Type::type::UINT8:
        var_ub += mNumberElements;
        mValueBuffer = (void*)var_ub;
        break;
      case arrow::Type::type::FLOAT:
        var_f += mNumberElements;
        mValueBuffer = (void*)var_f;
        break;
      case arrow::Type::type::DOUBLE:
        var_d += mNumberElements;
        mValueBuffer = (void*)var_d;
        break;
      case arrow::Type::type::UINT16:
        var_us += mNumberElements;
        mValueBuffer = (void*)var_us;
        break;
      case arrow::Type::type::UINT32:
        var_ui += mNumberElements;
        mValueBuffer = (void*)var_ui;
        break;
      case arrow::Type::type::UINT64:
        var_ul += mNumberElements;
        mValueBuffer = (void*)var_ul;
        break;
      case arrow::Type::type::INT16:
        var_s += mNumberElements;
        mValueBuffer = (void*)var_s;
        break;
      case arrow::Type::type::INT32:
        var_i += mNumberElements;
        mValueBuffer = (void*)var_i;
        break;
      case arrow::Type::type::INT64:
        var_l += mNumberElements;
        mValueBuffer = (void*)var_l;
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
#define MAKE_LIST_BUILDER(ElementType, NumElements)         \
  std::unique_ptr<arrow::ArrayBuilder> ValueBuilder;        \
  MakeBuilder(mMemoryPool, ElementType, &ValueBuilder);     \
  bui_list = std::make_shared<arrow::FixedSizeListBuilder>( \
    mMemoryPool,                                            \
    std::move(ValueBuilder),                                \
    NumElements);

#define MAKE_FIELD(ElementType, NumElements)                                                         \
  if (NumElements == 1) {                                                                            \
    mField =                                                                                         \
      std::make_shared<arrow::Field>(mColumnName, ElementType);                                      \
  } else {                                                                                           \
    mField =                                                                                         \
      std::make_shared<arrow::Field>(mColumnName, arrow::fixed_size_list(ElementType, NumElements)); \
  }

#define MAKE_FIELD_AND_BUILDER(ElementCType, NumElements, Builder)                                                                  \
  MAKE_FIELD(arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::type_singleton(), NumElements);                        \
  if (NumElements == 1) {                                                                                                           \
    Builder = new arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::BuilderType(mMemoryPool);                         \
  } else {                                                                                                                          \
    MAKE_LIST_BUILDER(arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::type_singleton(), NumElements);               \
    Builder = static_cast<arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::BuilderType*>(bui_list->value_builder()); \
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
        var_o = new TTreeReaderValue<Bool_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(bool, 1, bui_o);
        break;
      case EDataType::kUChar_t:
        var_ub = new TTreeReaderValue<UChar_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint8_t, 1, bui_ub);
        break;
      case EDataType::kFloat_t:
        var_f = new TTreeReaderValue<Float_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(float, 1, bui_f);
        break;
      case EDataType::kDouble_t:
        var_d = new TTreeReaderValue<Double_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(double, 1, bui_d);
        break;
      case EDataType::kUShort_t:
        var_us = new TTreeReaderValue<UShort_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint16_t, 1, bui_us);
        break;
      case EDataType::kUInt_t:
        var_ui = new TTreeReaderValue<UInt_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint32_t, 1, bui_ui);
        break;
      case EDataType::kULong64_t:
        var_ul = new TTreeReaderValue<long unsigned int>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint64_t, 1, bui_ul);
        break;
      case EDataType::kShort_t:
        var_s = new TTreeReaderValue<Short_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int16_t, 1, bui_s);
        break;
      case EDataType::kInt_t:
        var_i = new TTreeReaderValue<Int_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int32_t, 1, bui_i);
        break;
      case EDataType::kLong64_t:
        var_l = new TTreeReaderValue<long int>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int64_t, 1, bui_l);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  } else {
    switch (mElementType) {
      case EDataType::kBool_t:
        arr_o = new TTreeReaderArray<Bool_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(bool, mNumberElements, bui_o);
        break;
      case EDataType::kUChar_t:
        arr_ub = new TTreeReaderArray<UChar_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint8_t, mNumberElements, bui_ub);
        break;
      case EDataType::kFloat_t:
        arr_f = new TTreeReaderArray<Float_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(float, mNumberElements, bui_f);
        break;
      case EDataType::kDouble_t:
        arr_d = new TTreeReaderArray<Double_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(double, mNumberElements, bui_d);
        break;
      case EDataType::kUShort_t:
        arr_us = new TTreeReaderArray<UShort_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint16_t, mNumberElements, bui_us);
        break;
      case EDataType::kUInt_t:
        arr_ui = new TTreeReaderArray<UInt_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint32_t, mNumberElements, bui_ui);
        break;
      case EDataType::kULong64_t:
        arr_ul = new TTreeReaderArray<long unsigned int>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(uint64_t, mNumberElements, bui_ul);
        break;
      case EDataType::kShort_t:
        arr_s = new TTreeReaderArray<Short_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int16_t, mNumberElements, bui_s);
        break;
      case EDataType::kInt_t:
        arr_i = new TTreeReaderArray<Int_t>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int32_t, mNumberElements, bui_i);
        break;
      case EDataType::kLong64_t:
        arr_l = new TTreeReaderArray<long int>(*reader, mColumnName);
        MAKE_FIELD_AND_BUILDER(int64_t, mNumberElements, bui_l);
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
  delete var_o;
  delete var_ub;
  delete var_f;
  delete var_d;
  delete var_us;
  delete var_ui;
  delete var_ul;
  delete var_s;
  delete var_i;
  delete var_l;

  delete arr_o;
  delete arr_ub;
  delete arr_f;
  delete arr_d;
  delete arr_us;
  delete arr_ui;
  delete arr_ul;
  delete arr_s;
  delete arr_i;
  delete arr_l;
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
        stat = bui_o->Append((bool)**var_o);
        break;
      case EDataType::kUChar_t:
        stat = bui_ub->Append(**var_ub);
        break;
      case EDataType::kFloat_t:
        stat = bui_f->Append(**var_f);
        break;
      case EDataType::kDouble_t:
        stat = bui_d->Append(**var_d);
        break;
      case EDataType::kUShort_t:
        stat = bui_us->Append(**var_us);
        break;
      case EDataType::kUInt_t:
        stat = bui_ui->Append(**var_ui);
        break;
      case EDataType::kULong64_t:
        stat = bui_ul->Append(**var_ul);
        break;
      case EDataType::kShort_t:
        stat = bui_s->Append(**var_s);
        break;
      case EDataType::kInt_t:
        stat = bui_i->Append(**var_i);
        break;
      case EDataType::kLong64_t:
        stat = bui_l->Append(**var_l);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  } else {
    bui_list->AppendValues(1);
    switch (mElementType) {
      case EDataType::kBool_t:
        stat = bui_o->AppendValues((uint8_t*)&((*arr_o)[0]), mNumberElements);
        break;
      case EDataType::kUChar_t:
        stat = bui_ub->AppendValues(&((*arr_ub)[0]), mNumberElements);
        break;
      case EDataType::kFloat_t:
        stat = bui_f->AppendValues(&((*arr_f)[0]), mNumberElements);
        break;
      case EDataType::kDouble_t:
        stat = bui_d->AppendValues(&((*arr_d)[0]), mNumberElements);
        break;
      case EDataType::kUShort_t:
        stat = bui_us->AppendValues(&((*arr_us)[0]), mNumberElements);
        break;
      case EDataType::kUInt_t:
        stat = bui_ui->AppendValues(&((*arr_ui)[0]), mNumberElements);
        break;
      case EDataType::kULong64_t:
        stat = bui_ul->AppendValues(&((*arr_ul)[0]), mNumberElements);
        break;
      case EDataType::kShort_t:
        stat = bui_s->AppendValues(&((*arr_s)[0]), mNumberElements);
        break;
      case EDataType::kInt_t: {
        stat = bui_i->AppendValues(&((*arr_i)[0]), mNumberElements);
        break;
      }
      case EDataType::kLong64_t:
        stat = bui_l->AppendValues(&((*arr_l)[0]), mNumberElements);
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
        stat = bui_o->Finish(&mArray);
        break;
      case EDataType::kUChar_t:
        stat = bui_ub->Finish(&mArray);
        break;
      case EDataType::kFloat_t:
        stat = bui_f->Finish(&mArray);
        break;
      case EDataType::kDouble_t:
        stat = bui_d->Finish(&mArray);
        break;
      case EDataType::kUShort_t:
        stat = bui_us->Finish(&mArray);
        break;
      case EDataType::kUInt_t:
        stat = bui_ui->Finish(&mArray);
        break;
      case EDataType::kULong64_t:
        stat = bui_ul->Finish(&mArray);
        break;
      case EDataType::kShort_t:
        stat = bui_s->Finish(&mArray);
        break;
      case EDataType::kInt_t:
        stat = bui_i->Finish(&mArray);
        break;
      case EDataType::kLong64_t:
        stat = bui_l->Finish(&mArray);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", mElementType);
        break;
    }
  } else {
    bui_list->Finish(&mArray);
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

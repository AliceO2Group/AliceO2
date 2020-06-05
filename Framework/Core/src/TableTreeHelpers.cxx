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
branchIterator::branchIterator(TTree* tree, std::shared_ptr<arrow::ChunkedArray> col, std::shared_ptr<arrow::Field> field)
{
  mfield = field.get();
  mbranchName = mfield->name();
  
  mfieldType = mfield->type()->id();
  mchunks = col->chunks();
  mnumberChuncs = mchunks.size();
  
  mleaflistString = mbranchName;
  melementType = mfieldType;
  mnumColumnElements = 1;
  if (mfieldType == arrow::Type::type::FIXED_SIZE_LIST) {
  
    // element type
    if (mfield->type()->num_children() <= 0) {
      LOGP(FATAL, "Field {} of type {} has no children!", mfield->name(),mfield->type()->ToString().c_str());
    }
    melementType = mfield->type()->child(0)->type()->id();
    
    // number of elements
    mnumColumnElements = static_cast<const arrow::FixedSizeListType*>(mfield->type().get())->list_size();
    mleaflistString += "[" + std::to_string(mnumColumnElements) + "]";
  }
  LOGP(INFO, "Column {} [{}] with {} element(s) of type {}",mbranchName,mfield->type()->ToString(),mnumColumnElements,melementType);

  // initialize the branch
  mstatus = initBranch(tree);

  mcounterChunk = 0;
  mstatus &= initDataBuffer(mcounterChunk);
  LOGP(INFO, "branchIterator {} with {} chunks.\n", col->type()->ToString(), mnumberChuncs);
}

branchIterator::~branchIterator()
{
  delete mbranchPtr;
  delete dbuf;

  delete var_b;
  delete var_f;
  delete var_d;
  delete var_us;
  delete var_ui;
  delete var_ul;
  delete var_s;
  delete var_i;
  delete var_l;
}

bool branchIterator::getStatus()
{
  return mstatus;
}

bool branchIterator::initBranch(TTree* tree)
{
  // try to find branch in tree
  mbranchPtr = tree->GetBranch(mbranchName.c_str());
  if (mbranchPtr) {
    return true;
  }

  // create new branch of given data type
  switch (melementType) {
    case arrow::Type::type::BOOL:
      mleaflistString += "/O";
      break;
    case arrow::Type::type::UINT8:
      mleaflistString += "/b";
      break;
    case arrow::Type::type::FLOAT:
      mleaflistString += "/F";
      break;
    case arrow::Type::type::DOUBLE:
      mleaflistString += "/D";
      break;
    case arrow::Type::type::UINT16:
      mleaflistString += "/s";
      break;
    case arrow::Type::type::UINT32:
      mleaflistString += "/i";
      break;
    case arrow::Type::type::UINT64:
      mleaflistString += "/l";
      break;
    case arrow::Type::type::INT16:
      mleaflistString += "/S";
      break;
    case arrow::Type::type::INT32:
      mleaflistString += "/I";
      break;
    case arrow::Type::type::INT64:
      mleaflistString += "/L";
      break;
    default:
      LOGP(FATAL, "Type {} not handled!", melementType);
      break;
  }

  mbranchPtr = tree->Branch(mbranchName.c_str(), dbuf, mleaflistString.c_str());
  if (mbranchPtr) {
    LOGP(INFO, "Branch {} created with leaflist {}.",mbranchName.c_str(),mleaflistString.c_str());
    return true;
  } else {
    return false;
  }
}

bool branchIterator::initDataBuffer(Int_t ib)
{
  
  auto chunkToUse = mchunks.at(ib);
  if (mfieldType == arrow::Type::type::FIXED_SIZE_LIST) {
    chunkToUse = std::dynamic_pointer_cast<arrow::FixedSizeListArray>(chunkToUse)->values();
  }
  LOGP(INFO, "chunk of type {}",chunkToUse->type()->ToString());
  
  // reset actual row number
  mcounterRow = 0;

  // get next chunk of given data type melementType
  LOGP(INFO, "Get chunk of type {}.",melementType);
  switch (melementType) {
    case arrow::Type::type::BOOL:
      var_o = std::dynamic_pointer_cast<arrow::BooleanArray>(chunkToUse);
      boolValueHolder = (Bool_t)var_o->Value(mcounterRow);
      v = (void*)&boolValueHolder;
      break;
    case arrow::Type::type::UINT8:
      var_b = (UChar_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt8Type>>(chunkToUse)->raw_values();
      v = (void*)var_b;
      break;
    case arrow::Type::type::FLOAT:
      var_f = (Float_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::FloatType>>(chunkToUse)->raw_values();
      v = (void*)var_f;
      break;
    case arrow::Type::type::DOUBLE:
      var_d = (Double_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(chunkToUse)->raw_values();
      v = (void*)var_d;
      break;
    case arrow::Type::type::UINT16:
      var_us = (UShort_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt16Type>>(chunkToUse)->raw_values();
      v = (void*)var_us;
      break;
    case arrow::Type::type::UINT32:
      var_ui = (UInt_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt32Type>>(chunkToUse)->raw_values();
      v = (void*)var_ui;
      break;
    case arrow::Type::type::UINT64:
      var_ul = (long unsigned int*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt64Type>>(chunkToUse)->raw_values();
      v = (void*)var_ul;
      break;
    case arrow::Type::type::INT16:
      var_s = (Short_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int16Type>>(chunkToUse)->raw_values();
      v = (void*)var_s;
      break;
    case arrow::Type::type::INT32:
      var_i = (Int_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(chunkToUse)->raw_values();
      v = (void*)var_i;
      break;
    case arrow::Type::type::INT64:
      var_l = (long int*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(chunkToUse)->raw_values();
      v = (void*)var_l;
      break;
    default:
      LOGP(FATAL, "Type {} not handled!", melementType);
      break;
  }
  mbranchPtr->SetAddress(v);

  // reset number of rows mnumberRows and row counter mcounterRow
  mnumberRows = mchunks.at(ib)->length();

  return true;
}

bool branchIterator::push()
{
  // increment row counter
  mcounterRow++;

  // mcounterChunk and mcounterRow contain the current chunk and row
  // return the next element if available
  if (mcounterRow >= mnumberRows) {
    mcounterChunk++;
    if (mcounterChunk < mnumberChuncs) {
      initDataBuffer(mcounterChunk);
    } else {
      // end of data buffer reached
      LOGP(INFO, "End of {} data buffer reached!",mbranchName);
      return false;
    }
  } else {
    switch (melementType) {
      case arrow::Type::type::BOOL:
        boolValueHolder = (Bool_t)var_o->Value(mcounterRow);
        v = (void*)&boolValueHolder;
        break;
      case arrow::Type::type::UINT8:
        var_b += mnumColumnElements;
        v = (void*)var_b;
        break;
      case arrow::Type::type::FLOAT:
        var_f += mnumColumnElements;
        v = (void*)var_f;
        break;
      case arrow::Type::type::DOUBLE:
        var_d += mnumColumnElements;
        v = (void*)var_d;
        break;
      case arrow::Type::type::UINT16:
        var_us += mnumColumnElements;
        v = (void*)var_us;
        break;
      case arrow::Type::type::UINT32:
        var_ui += mnumColumnElements;
        v = (void*)var_ui;
        break;
      case arrow::Type::type::UINT64:
        var_ul += mnumColumnElements;
        v = (void*)var_ul;
        break;
      case arrow::Type::type::INT16:
        var_s += mnumColumnElements;
        v = (void*)var_s;
        break;
      case arrow::Type::type::INT32:
        var_i += mnumColumnElements;
        v = (void*)var_i;
        break;
      case arrow::Type::type::INT64:
        var_l += mnumColumnElements;
        v = (void*)var_l;
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", melementType);
        break;
    }
  }
  mbranchPtr->SetAddress(v);

  return true;
}

TableToTree::TableToTree(std::shared_ptr<arrow::Table> table,
                         TFile* file,
                         const char* treename)
{
  mtable = table;

  // try to get the tree
  mtreePtr = (TTree*)file->Get(treename);

  // create the tree if it does not exist already
  if (!mtreePtr) {
    mtreePtr = new TTree(treename, treename);
  }
}

TableToTree::~TableToTree()
{
  // clean up branch iterators
  mbranchIterators.clear();
}

bool TableToTree::addBranch(std::shared_ptr<arrow::ChunkedArray> col, std::shared_ptr<arrow::Field> field)
{
  branchIterator* brit = new branchIterator(mtreePtr, col, field);
  if (brit->getStatus()) {
    mbranchIterators.push_back(brit);
  }

  return brit->getStatus();
}

bool TableToTree::addAllBranches()
{

  bool status = mtable->num_columns() > 0;
  for (auto ii = 0; ii < mtable->num_columns(); ii++) {
    branchIterator* brit =
      new branchIterator(mtreePtr, mtable->column(ii), mtable->schema()->field(ii));
    if (brit->getStatus()) {
      mbranchIterators.push_back(brit);
    } else {
      status = false;
    }
  }

  return status;
}

TTree* TableToTree::process()
{

  bool togo = true;
  LOGP(INFO, "Number of colums: {}", mbranchIterators.size());
  while (togo) {
    // fill the tree
    mtreePtr->Fill();

    // update the branches
    for (auto brit : mbranchIterators) {
      togo &= brit->push();
    }
  }
  mtreePtr->Write();

  return mtreePtr;
}

// -----------------------------------------------------------------------------
#define MAKE_LIST_BUILDER(ElementType, NumElements)           \
  MakeBuilder(mpool,ElementType,&mvalueBuilder);              \
  bui_list = std::make_shared<arrow::FixedSizeListBuilder>(   \
    mpool,                                                    \
    std::move(mvalueBuilder),                                 \
    NumElements );

#define MAKE_FIELD(ElementType, NumElements)                                                        \
  if (NumElements == 1) {                                                                           \
    mfield =                                                                                        \
      std::make_shared<arrow::Field>(mcolumnName,ElementType);                                      \
  } else {                                                                                          \
    mfield =                                                                                        \
      std::make_shared<arrow::Field>(mcolumnName, arrow::fixed_size_list(ElementType,NumElements)); \
  }

#define MAKE_FIELD_AND_BUILDER(ElementCType, NumElements, Builder)                                                                  \
  MAKE_FIELD(arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::type_singleton(), NumElements);                        \
  if (NumElements == 1) {                                                                                                           \
    Builder = new arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::BuilderType(mpool);                               \
  } else {                                                                                                                          \
    MAKE_LIST_BUILDER(arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::type_singleton(),NumElements);                \
    Builder = static_cast<arrow::TypeTraits<arrow::CTypeTraits<ElementCType>::ArrowType>::BuilderType*>(bui_list->value_builder()); \
  }


// is used in TreeToTable
columnIterator::columnIterator(TTreeReader* reader, const char* colname)
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
  mcolumnName = colname;
  
  // type of the branch elements
  TClass* cl;
  br->GetExpectedType(cl, melementType);

  // currently only single-value or single-array branches are accepted
  // thus of the form e.g. alpha/D or alpha[5]/D
  // check if this is a single-value or single-array branch
  mnumColumnElements = 1;
  std::string branchTitle = br->GetTitle();
  Int_t pos0 = branchTitle.find("[");
  Int_t pos1 = branchTitle.find("]");
  if (pos0>0 && pos1>0) {
    mnumColumnElements = atoi(branchTitle.substr(pos0+1,pos1-pos0-1).c_str());
  }
  printf("melementType %i mnumColumnElements %i\n",melementType,mnumColumnElements);

  // initialize the TTreeReaderValue<T>
  //            the corresponding arrow::TBuilder
  //            the column schema
  // the TTreeReaderValue is incremented by reader->Next()
  // switch according to melementType
  mstatus = true;

  if (mnumColumnElements == 1) {
    switch (melementType) {
      case EDataType::kBool_t:
        var_o = new TTreeReaderValue<Bool_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(bool,1,bui_o);
        break;
      case EDataType::kUChar_t:
        var_b = new TTreeReaderValue<UChar_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(uint8_t,1,bui_ub);
        break;
      case EDataType::kFloat_t:
        var_f = new TTreeReaderValue<Float_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(float,1,bui_f);
        break;
      case EDataType::kDouble_t:
        var_d = new TTreeReaderValue<Double_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(double,1,bui_d);
        break;
      case EDataType::kUShort_t:
        var_us = new TTreeReaderValue<UShort_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(uint16_t,1,bui_us);
        break;
      case EDataType::kUInt_t:
        var_ui = new TTreeReaderValue<UInt_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(uint32_t,1,bui_ui);
        break;
      case EDataType::kULong64_t:
        var_ul = new TTreeReaderValue<long unsigned int>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(uint64_t,1,bui_ul);
        break;
      case EDataType::kShort_t:
        var_s = new TTreeReaderValue<Short_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(int16_t,1,bui_s);
        break;
      case EDataType::kInt_t:
        var_i = new TTreeReaderValue<Int_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(int32_t,1,bui_i);
        break;
      case EDataType::kLong64_t:
        var_l = new TTreeReaderValue<long int>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(int64_t,1,bui_l);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", melementType);
        break;
    }
  } else {
    switch (melementType) {
      case EDataType::kBool_t:
        arr_o = new TTreeReaderArray<Bool_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(bool,mnumColumnElements,bui_o);
        break;
      case EDataType::kUChar_t:
        arr_b = new TTreeReaderArray<UChar_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(uint8_t,mnumColumnElements,bui_ub);
        break;
      case EDataType::kFloat_t:
        arr_f = new TTreeReaderArray<Float_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(float,mnumColumnElements,bui_f);
        break;
      case EDataType::kDouble_t:
        arr_d = new TTreeReaderArray<Double_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(double,mnumColumnElements,bui_d);
        break;
      case EDataType::kUShort_t:
        arr_us = new TTreeReaderArray<UShort_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(uint16_t,mnumColumnElements,bui_us);
        break;
      case EDataType::kUInt_t:
        arr_ui = new TTreeReaderArray<UInt_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(uint32_t,mnumColumnElements,bui_ui);
        break;
      case EDataType::kULong64_t:
        arr_ul = new TTreeReaderArray<long unsigned int>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(uint64_t,mnumColumnElements,bui_ul);
        break;
      case EDataType::kShort_t:
        arr_s = new TTreeReaderArray<Short_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(int16_t,mnumColumnElements,bui_s);
        break;
      case EDataType::kInt_t:
        arr_i = new TTreeReaderArray<Int_t>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(int32_t,mnumColumnElements,bui_i);
        break;
      case EDataType::kLong64_t:
        arr_l = new TTreeReaderArray<long int>(*reader, mcolumnName);
        MAKE_FIELD_AND_BUILDER(int64_t,mnumColumnElements,bui_l);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", melementType);
        break;
    }
  }
}

columnIterator::~columnIterator()
{
  // delete all pointers
  delete var_o;
  delete var_b;
  delete var_f;
  delete var_d;
  delete var_us;
  delete var_ui;
  delete var_ul;
  delete var_s;
  delete var_i;
  delete var_l;

  delete arr_o;
  delete arr_b;
  delete arr_f;
  delete arr_d;
  delete arr_us;
  delete arr_ui;
  delete arr_ul;
  delete arr_s;
  delete arr_i;
  delete arr_l;

};

bool columnIterator::getStatus()
{
  return mstatus;
}

void columnIterator::push()
{
  arrow::Status stat;

  // switch according to melementType
  if (mnumColumnElements == 1) {
    switch (melementType) {
      case EDataType::kBool_t:
        stat = bui_o->Append((bool)**var_o);
        break;
      case EDataType::kUChar_t:
        stat = bui_ub->Append(**var_b);
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
        LOGP(FATAL, "Type {} not handled!", melementType);
        break;
    }
  } else {
    bui_list->AppendValues(1);
    switch (melementType) {
      case EDataType::kBool_t:
        stat = bui_o->AppendValues((uint8_t*)&((*arr_o)[0]),mnumColumnElements);
        break;
      case EDataType::kUChar_t:
        stat = bui_ub->AppendValues(&((*arr_b)[0]),mnumColumnElements);
        break;
      case EDataType::kFloat_t:
        stat = bui_f->AppendValues(&((*arr_f)[0]),mnumColumnElements);
        break;
      case EDataType::kDouble_t:
        stat = bui_d->AppendValues(&((*arr_d)[0]),mnumColumnElements);
        break;
      case EDataType::kUShort_t:
        stat = bui_us->AppendValues(&((*arr_us)[0]),mnumColumnElements);
        break;
      case EDataType::kUInt_t:
        stat = bui_ui->AppendValues(&((*arr_ui)[0]),mnumColumnElements);
        break;
      case EDataType::kULong64_t:
        stat = bui_ul->AppendValues(&((*arr_ul)[0]),mnumColumnElements);
        break;
      case EDataType::kShort_t:
        stat = bui_s->AppendValues(&((*arr_s)[0]),mnumColumnElements);
        break;
      case EDataType::kInt_t: {
        stat = bui_i->AppendValues(&((*arr_i)[0]),mnumColumnElements);
        break; }
      case EDataType::kLong64_t:
        stat = bui_l->AppendValues(&((*arr_l)[0]),mnumColumnElements);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", melementType);
        break;
    }
  }  
}

void columnIterator::finish()
{
  arrow::Status stat;

  // switch according to melementType
  if (mnumColumnElements == 1) {
    switch (melementType) {
      case EDataType::kBool_t:
        stat = bui_o->Finish(&marray);
        break;
      case EDataType::kUChar_t:
        stat = bui_ub->Finish(&marray);
        break;
      case EDataType::kFloat_t:
        stat = bui_f->Finish(&marray);
        break;
      case EDataType::kDouble_t:
        stat = bui_d->Finish(&marray);
        break;
      case EDataType::kUShort_t:
        stat = bui_us->Finish(&marray);
        break;
      case EDataType::kUInt_t:
        stat = bui_ui->Finish(&marray);
        break;
      case EDataType::kULong64_t:
        stat = bui_ul->Finish(&marray);
        break;
      case EDataType::kShort_t:
        stat = bui_s->Finish(&marray);
        break;
      case EDataType::kInt_t:
        stat = bui_i->Finish(&marray);
        break;
      case EDataType::kLong64_t:
        stat = bui_l->Finish(&marray);
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", melementType);
        break;
    }
  } else {
    bui_list->Finish(&marray);
  }
}

TreeToTable::TreeToTable(TTree* tree)
{
  // initialize the TTreeReader
  mreader = new TTreeReader(tree);
}

TreeToTable::~TreeToTable()
{
  delete mreader;
};

bool TreeToTable::addColumn(const char* colname)
{
  auto colit = std::make_shared<columnIterator>(mreader, colname);
  auto stat = colit->getStatus();
  if (stat) {
    mcolumnIterators.push_back(std::move(colit));
  }

  return stat;
}

bool TreeToTable::addAllColumns()
{
  // get a list of column names
  auto tree = mreader->GetTree();
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
    auto colit = std::make_shared<columnIterator>(mreader, br->GetName());
    if (colit->getStatus()) {
      mcolumnIterators.push_back(std::move(colit));
    } else {
      status = false;
    }
  }

  return status;
}

void TreeToTable::push()
{
  for (auto colit : mcolumnIterators) {
    colit->push();
  }
}

void TreeToTable::fill()
{
  // copy all values from the tree to the table builders
  mreader->Restart();
  while (mreader->Next()) {
    push();
  }
}

std::shared_ptr<arrow::Table> TreeToTable::finalize()
{
  // prepare the elements needed to create the final table
  std::vector<std::shared_ptr<arrow::Array>> array_vector;
  std::vector<std::shared_ptr<arrow::Field>> schema_vector;
  for (auto colit : mcolumnIterators) {
    colit->finish();
    array_vector.push_back(colit->getArray());
    schema_vector.push_back(colit->getSchema());
  }
  auto fields = std::make_shared<arrow::Schema>(schema_vector);
  LOGP(INFO, "Schema {}",fields->ToString());

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

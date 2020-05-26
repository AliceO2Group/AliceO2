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

namespace o2
{
namespace framework
{

branchIterator::branchIterator(TTree* tree, std::shared_ptr<arrow::ChunkedArray> col, std::shared_ptr<arrow::Field> field)
{
  mbranchName = field->name().c_str();
  marrowType = field->type()->id();
  mchunks = col->chunks();
  mnumberChuncs = mchunks.size();

  // initialize the branch
  mstatus = initBranch(tree);

  mcounterChunk = 0;
  mstatus &= initDataBuffer(mcounterChunk);
  LOGP(DEBUG, "branchIterator: {} with {} chunks", col->type()->ToString(), mnumberChuncs);
}

branchIterator::~branchIterator()
{
  delete mbranchPtr;
  delete dbuf;

  delete var_b;
  delete var_f;
  delete var_d;
  delete vs;
  delete vi;
  delete vl;
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
  switch (marrowType) {
    case arrow::Type::type::BOOL:
      mleaflistString = mbranchName + "/O";
      break;
    case arrow::Type::type::UINT8:
      mleaflistString = mbranchName + "/b";
      break;
    case arrow::Type::type::FLOAT:
      mleaflistString = mbranchName + "/F";
      break;
    case arrow::Type::type::DOUBLE:
      mleaflistString = mbranchName + "/D";
      break;
    case arrow::Type::type::UINT16:
      mleaflistString = mbranchName + "/s";
      break;
    case arrow::Type::type::UINT32:
      mleaflistString = mbranchName + "/i";
      break;
    case arrow::Type::type::UINT64:
      mleaflistString = mbranchName + "/l";
      break;
    case arrow::Type::type::INT16:
      mleaflistString = mbranchName + "/S";
      break;
    case arrow::Type::type::INT32:
      mleaflistString = mbranchName + "/I";
      break;
    case arrow::Type::type::INT64:
      mleaflistString = mbranchName + "/L";
      break;
    default:
      LOGP(FATAL, "Type {} not handled!", marrowType);
      break;
  }

  mbranchPtr = tree->Branch(mbranchName.c_str(), dbuf, mleaflistString.c_str());
  if (mbranchPtr) {
    return true;
  } else {
    return false;
  }
}

bool branchIterator::initDataBuffer(Int_t ib)
{
  // reset actual row number
  mcounterRow = 0;

  // get next chunk of given data type marrowType
  switch (marrowType) {
    case arrow::Type::type::BOOL:
      var_o = std::dynamic_pointer_cast<arrow::BooleanArray>(mchunks.at(ib));
      boolValueHolder = (Bool_t)var_o->Value(mcounterRow);
      v = (void*)&boolValueHolder;
      break;
    case arrow::Type::type::UINT8:
      var_b = (UChar_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt8Type>>(mchunks.at(ib))->raw_values();
      v = (void*)var_b;
      break;
    case arrow::Type::type::FLOAT:
      var_f = (Float_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::FloatType>>(mchunks.at(ib))->raw_values();
      v = (void*)var_f;
      break;
    case arrow::Type::type::DOUBLE:
      var_d = (Double_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(mchunks.at(ib))->raw_values();
      v = (void*)var_d;
      break;
    case arrow::Type::type::UINT16:
      vs = (UShort_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt16Type>>(mchunks.at(ib))->raw_values();
      v = (void*)vs;
      break;
    case arrow::Type::type::UINT32:
      vi = (UInt_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt32Type>>(mchunks.at(ib))->raw_values();
      v = (void*)vi;
      break;
    case arrow::Type::type::UINT64:
      vl = (ULong64_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt64Type>>(mchunks.at(ib))->raw_values();
      v = (void*)vl;
      break;
    case arrow::Type::type::INT16:
      var_s = (Short_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int16Type>>(mchunks.at(ib))->raw_values();
      v = (void*)var_s;
      break;
    case arrow::Type::type::INT32:
      var_i = (Int_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(mchunks.at(ib))->raw_values();
      v = (void*)var_i;
      break;
    case arrow::Type::type::INT64:
      var_l = (Long64_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(mchunks.at(ib))->raw_values();
      v = (void*)var_l;
      break;
    default:
      LOGP(FATAL, "Type {} not handled!", marrowType);
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
      return false;
    }
  } else {
    switch (marrowType) {
      case arrow::Type::type::BOOL:
        boolValueHolder = (Bool_t)var_o->Value(mcounterRow);
        v = (void*)&boolValueHolder;
        break;
      case arrow::Type::type::UINT8:
        v = (void*)++var_b;
        break;
      case arrow::Type::type::FLOAT:
        v = (void*)++var_f;
        break;
      case arrow::Type::type::DOUBLE:
        v = (void*)++var_d;
        break;
      case arrow::Type::type::UINT16:
        v = (void*)++vs;
        break;
      case arrow::Type::type::UINT32:
        v = (void*)++vi;
        break;
      case arrow::Type::type::UINT64:
        v = (void*)++vl;
        break;
      case arrow::Type::type::INT16:
        v = (void*)++var_s;
        break;
      case arrow::Type::type::INT32:
        v = (void*)++var_i;
        break;
      case arrow::Type::type::INT64:
        v = (void*)++var_l;
        break;
      default:
        LOGP(FATAL, "Type {} not handled!", marrowType);
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
  LOGP(DEBUG, "Number of colums: {}", mbranchIterators.size());
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

  TClass* cl;
  br->GetExpectedType(cl, marrowType);
  // initialize the TTreeReaderValue<T>
  //            the corresponding arrow::TBuilder
  //            the column schema
  // the TTreeReaderValue is incremented by reader->Next()
  // switch according to marrowType
  mstatus = true;

  switch (marrowType) {
    case EDataType::kBool_t:
      var_o = new TTreeReaderValue<Bool_t>(*reader, mcolumnName);
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::boolean());
      bui_o = new arrow::BooleanBuilder(mpool);
      break;
    case EDataType::kUChar_t:
      var_b = new TTreeReaderValue<UChar_t>(*reader, mcolumnName);
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::uint8());
      bb = new arrow::UInt8Builder(mpool);
      break;
    case EDataType::kFloat_t:
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::float32());
      var_f = new TTreeReaderValue<Float_t>(*reader, mcolumnName);
      bui_f = new arrow::FloatBuilder(mpool);
      break;
    case EDataType::kDouble_t:
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::float64());
      var_d = new TTreeReaderValue<Double_t>(*reader, mcolumnName);
      bui_d = new arrow::DoubleBuilder(mpool);
      break;
    case EDataType::kUShort_t:
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::uint16());
      vs = new TTreeReaderValue<UShort_t>(*reader, mcolumnName);
      bs = new arrow::UInt16Builder(mpool);
      break;
    case EDataType::kUInt_t:
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::uint32());
      vi = new TTreeReaderValue<UInt_t>(*reader, mcolumnName);
      bi = new arrow::UInt32Builder(mpool);
      break;
    case EDataType::kULong64_t:
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::uint64());
      vl = new TTreeReaderValue<ULong64_t>(*reader, mcolumnName);
      bl = new arrow::UInt64Builder(mpool);
      break;
    case EDataType::kShort_t:
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::int16());
      var_s = new TTreeReaderValue<Short_t>(*reader, mcolumnName);
      bui_s = new arrow::Int16Builder(mpool);
      break;
    case EDataType::kInt_t:
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::int32());
      var_i = new TTreeReaderValue<Int_t>(*reader, mcolumnName);
      bui_i = new arrow::Int32Builder(mpool);
      break;
    case EDataType::kLong64_t:
      mfield = std::make_shared<arrow::Field>(mcolumnName, arrow::int64());
      var_l = new TTreeReaderValue<Long64_t>(*reader, mcolumnName);
      bui_l = new arrow::Int64Builder(mpool);
      break;
    default:
      LOGP(FATAL, "Type {} not handled!", marrowType);
      break;
  }
}

columnIterator::~columnIterator()
{
  // delete all pointers
  delete var_o;
  delete var_b;
  delete var_f;
  delete var_d;
  delete vs;
  delete vi;
  delete vl;
  delete var_s;
  delete var_i;
  delete var_l;

  delete bui_o;
  delete bb;
  delete bui_f;
  delete bui_d;
  delete bs;
  delete bi;
  delete bl;
  delete bui_s;
  delete bui_i;
  delete bui_l;
};

bool columnIterator::getStatus()
{
  return mstatus;
}

void columnIterator::push()
{
  arrow::Status stat;

  // switch according to marrowType
  switch (marrowType) {
    case EDataType::kBool_t:
      stat = bui_o->Append((bool)**var_o);
      break;
    case EDataType::kUChar_t:
      stat = bb->Append(**var_b);
      break;
    case EDataType::kFloat_t:
      stat = bui_f->Append(**var_f);
      break;
    case EDataType::kDouble_t:
      stat = bui_d->Append(**var_d);
      break;
    case EDataType::kUShort_t:
      stat = bs->Append(**vs);
      break;
    case EDataType::kUInt_t:
      stat = bi->Append(**vi);
      break;
    case EDataType::kULong64_t:
      stat = bl->Append(**vl);
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
      LOGP(FATAL, "Type {} not handled!", marrowType);
      break;
  }
}

void columnIterator::finish()
{
  arrow::Status stat;

  // switch according to marrowType
  switch (marrowType) {
    case EDataType::kBool_t:
      stat = bui_o->Finish(&marray);
      break;
    case EDataType::kUChar_t:
      stat = bb->Finish(&marray);
      break;
    case EDataType::kFloat_t:
      stat = bui_f->Finish(&marray);
      break;
    case EDataType::kDouble_t:
      stat = bui_d->Finish(&marray);
      break;
    case EDataType::kUShort_t:
      stat = bs->Finish(&marray);
      break;
    case EDataType::kUInt_t:
      stat = bi->Finish(&marray);
      break;
    case EDataType::kULong64_t:
      stat = bl->Finish(&marray);
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
      LOGP(FATAL, "Type {} not handled!", marrowType);
      break;
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

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_TABLETREE_H
#define FRAMEWORK_TABLETREE_H

#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TableBuilder.h"

// =============================================================================
namespace o2
{
namespace framework
{

// -----------------------------------------------------------------------------
// TableToTree allows to save the contents of a given arrow::Table to a TTree
//  branchIterator is used by TableToTree
//
// To write the contents of a table ta to a tree tr on file f do:
//  . TableToTree t2t(ta,f,treename);
//  . t2t.AddBranch(branchname1); t2t.AddBranch(branchname2); ...
//    OR
//    t2t.AddAllBranches();
//  . t2t.Process();
//
// .............................................................................
class branchIterator
{

 private:
  std::string brn;        // branch name
  arrow::ArrayVector chs; // chunks
  Int_t nchs;             // number of chunks
  Int_t ich;              // chunk counter
  Int_t nr;               // number of rows
  Int_t ir;               // row counter

  // data buffers for each data type
  bool status = false;
  arrow::Type::type dt;
  std::string leaflist;

  TBranch* br = nullptr;

  char* dbuf = nullptr;
  void* v = nullptr;

  Float_t* var_f = nullptr;
  Double_t* var_d = nullptr;
  UShort_t* vs = nullptr;
  UInt_t* vi = nullptr;
  ULong64_t* vl = nullptr;
  Short_t* var_s = nullptr;
  Int_t* var_i = nullptr;
  Long64_t* var_l = nullptr;

  // initialize a branch
  bool branchini(TTree* tree)
  {

    // try to find branch in tree
    br = tree->GetBranch(brn.c_str());

    if (!br) {

      // create new branch of given data type
      switch (dt) {
        case arrow::Type::type::FLOAT:
          leaflist = brn + "/F";
          break;
        case arrow::Type::type::DOUBLE:
          leaflist = brn + "/D";
          break;
        case arrow::Type::type::UINT16:
          leaflist = brn + "/s";
          break;
        case arrow::Type::type::UINT32:
          leaflist = brn + "/i";
          break;
        case arrow::Type::type::UINT64:
          leaflist = brn + "/l";
          break;
        case arrow::Type::type::INT16:
          leaflist = brn + "/S";
          break;
        case arrow::Type::type::INT32:
          leaflist = brn + "/I";
          break;
        case arrow::Type::type::INT64:
          leaflist = brn + "/L";
          break;
        default:
          LOG(FATAL) << "Type not handled: " << dt << std::endl;
          break;
      }

      br = tree->Branch(brn.c_str(), dbuf, leaflist.c_str());
    }
    if (!br)
      return false;

    return true;
  };

  // initialize chunk ib
  bool dbufini(Int_t ib)
  {
    // get next chunk of given data type dt
    switch (dt) {
      case arrow::Type::type::FLOAT:
        var_f = (Float_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::FloatType>>(chs.at(ib))->raw_values();
        v = (void*)var_f;
        break;
      case arrow::Type::type::DOUBLE:
        var_d = (Double_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::DoubleType>>(chs.at(ib))->raw_values();
        v = (void*)var_d;
        break;
      case arrow::Type::type::UINT16:
        vs = (UShort_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt16Type>>(chs.at(ib))->raw_values();
        v = (void*)vs;
        break;
      case arrow::Type::type::UINT32:
        vi = (UInt_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt32Type>>(chs.at(ib))->raw_values();
        v = (void*)vi;
        break;
      case arrow::Type::type::UINT64:
        vl = (ULong64_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::UInt64Type>>(chs.at(ib))->raw_values();
        v = (void*)vl;
        break;
      case arrow::Type::type::INT16:
        var_s = (Short_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int16Type>>(chs.at(ib))->raw_values();
        v = (void*)var_s;
        break;
      case arrow::Type::type::INT32:
        var_i = (Int_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(chs.at(ib))->raw_values();
        v = (void*)var_i;
        break;
      case arrow::Type::type::INT64:
        var_l = (Long64_t*)std::dynamic_pointer_cast<arrow::NumericArray<arrow::Int64Type>>(chs.at(ib))->raw_values();
        v = (void*)var_l;
        break;
      default:
        LOG(FATAL) << "Type not handled: " << dt << std::endl;
        break;
    }
    br->SetAddress(v);

    // reset number of rows nr and row counter ir
    nr = chs.at(ib)->length();
    ir = 0;

    return true;
  };

 public:
  branchIterator(TTree* tree, std::shared_ptr<arrow::Column> col)
  {
    brn = col->name().c_str();
    dt = col->type()->id();
    chs = col->data()->chunks();
    nchs = chs.size();

    // initialize the branch
    status = branchini(tree);

    ich = 0;
    status &= dbufini(ich);
    LOG(DEBUG) << "branchIterator: " << col->type()->ToString() << " with " << nchs << " chunks" << std::endl;
  };

  ~branchIterator() = default;

  // has the iterator been properly initialized
  bool Status()
  {
    return status;
  };

  // fills buffer with next value
  // returns false if end of buffer reached
  bool push()
  {

    // ich and ir contain the current chunk and row
    // return the next element if available
    if (ir >= nr) {
      ich++;
      if (ich < nchs) {
        dbufini(ich);
      } else {
        // end of data buffer reached
        return false;
      }
    } else {
      switch (dt) {
        case arrow::Type::type::FLOAT:
          v = (void*)var_f++;
          break;
        case arrow::Type::type::DOUBLE:
          v = (void*)var_d++;
          break;
        case arrow::Type::type::UINT16:
          v = (void*)vs++;
          break;
        case arrow::Type::type::UINT32:
          v = (void*)vi++;
          break;
        case arrow::Type::type::UINT64:
          v = (void*)vl++;
          break;
        case arrow::Type::type::INT16:
          v = (void*)var_s++;
          break;
        case arrow::Type::type::INT32:
          v = (void*)var_i++;
          break;
        case arrow::Type::type::INT64:
          v = (void*)var_l++;
          break;
        default:
          LOG(FATAL) << "Type not handled: " << dt << std::endl;
          break;
      }
    }
    br->SetAddress(v);
    ir++;

    return true;
  };
};

// -----------------------------------------------------------------------------
class TableToTree
{

 private:
  TTree* tr;

  // a list of branchIterator
  std::vector<branchIterator*> brits;

  // table to convert
  std::shared_ptr<arrow::Table> ta;

 public:
  TableToTree(std::shared_ptr<arrow::Table> table,
              TFile* file,
              const char* treename)
  {

    ta = table;

    // try to get the tree
    tr = (TTree*)file->Get(treename);

    // create the tree if it does not exist already
    if (!tr)
      tr = new TTree(treename, treename);
  }

  ~TableToTree()
  {

    // clean up branch iterators
    brits.clear();
  }

  bool AddBranch(std::shared_ptr<arrow::Column> col)
  {
    branchIterator* brit = new branchIterator(tr, col);
    if (brit->Status())
      brits.push_back(brit);

    return brit->Status();
  }

  bool AddAllBranches()
  {

    bool status = ta->num_columns() > 0;
    for (auto ii = 0; ii < ta->num_columns(); ii++) {
      branchIterator* brit =
        new branchIterator(tr, ta->column(ii));
      if (brit->Status()) {
        brits.push_back(brit);
      } else {
        status = false;
      }
    }

    return status;
  }

  TTree* Process()
  {

    bool togo = true;
    while (togo) {
      // fill the tree
      tr->Fill();

      // update the branches
      for (auto ii = 0; ii < ta->num_columns(); ii++)
        togo &= brits.at(ii)->push();
    }
    tr->Write();

    return tr;
  }
};

// -----------------------------------------------------------------------------
// TreeToTable allows to fill the contents of a given TTree to an arrow::Table
//  columnIterator is used by TreeToTable
//
// To copy the contents of a tree tr to a table ta do:
//  . TreeToTable t2t(tr);
//  . t2t.AddColumn(columnname1); t2t.AddColumn(columnname2); ...
//    OR
//    t2t.AddAllColumns();
//  . auto ta = t2t.Process();
//
// .............................................................................
class columnIterator
{

 private:
  // all the possible TTreeReaderValue<T> types
  TTreeReaderValue<Float_t>* var_f = nullptr;
  TTreeReaderValue<Double_t>* var_d = nullptr;
  TTreeReaderValue<UShort_t>* vs = nullptr;
  TTreeReaderValue<UInt_t>* vi = nullptr;
  TTreeReaderValue<ULong64_t>* vl = nullptr;
  TTreeReaderValue<Short_t>* var_s = nullptr;
  TTreeReaderValue<Int_t>* var_i = nullptr;
  TTreeReaderValue<Long64_t>* var_l = nullptr;

  // all the possible arrow::TBuilder types
  arrow::FloatBuilder* bui_f = nullptr;
  arrow::DoubleBuilder* bui_d = nullptr;
  arrow::UInt16Builder* bs = nullptr;
  arrow::UInt32Builder* bi = nullptr;
  arrow::UInt64Builder* bl = nullptr;
  arrow::Int16Builder* bui_s = nullptr;
  arrow::Int32Builder* bui_i = nullptr;
  arrow::Int64Builder* bui_l = nullptr;

  bool status = false;
  EDataType dt;
  const char* cname;

  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Field> field;
  std::shared_ptr<arrow::Array> ar;

 public:
  columnIterator(TTreeReader* reader, const char* colname)
  {

    // find branch
    auto tree = reader->GetTree();
    if (!tree) {
      LOG(INFO) << "Can not locate tree!";
      return;
    }
    //tree->Print();
    auto br = tree->GetBranch(colname);
    if (!br) {
      LOG(INFO) << "Can not locate branch " << colname;
      return;
    }
    cname = colname;

    TClass* cl;
    br->GetExpectedType(cl, dt);
    //LOG(INFO) << "Initialisation of TTreeReaderValue";
    //LOG(INFO) << "The column " << cname << " is of type " << dt;

    // initialize the TTreeReaderValue<T>
    //            the corresponding arrow::TBuilder
    //            the column schema
    // the TTreeReaderValue is incremented by reader->Next()
    // switch according to dt
    status = true;
    switch (dt) {
      case EDataType::kFloat_t:
        field = std::make_shared<arrow::Field>(cname, arrow::float32());
        var_f = new TTreeReaderValue<Float_t>(*reader, cname);
        bui_f = new arrow::FloatBuilder(pool);
        break;
      case EDataType::kDouble_t:
        field = std::make_shared<arrow::Field>(cname, arrow::float64());
        var_d = new TTreeReaderValue<Double_t>(*reader, cname);
        bui_d = new arrow::DoubleBuilder(pool);
        break;
      case EDataType::kUShort_t:
        field = std::make_shared<arrow::Field>(cname, arrow::uint16());
        vs = new TTreeReaderValue<UShort_t>(*reader, cname);
        bs = new arrow::UInt16Builder(pool);
        break;
      case EDataType::kUInt_t:
        field = std::make_shared<arrow::Field>(cname, arrow::uint32());
        vi = new TTreeReaderValue<UInt_t>(*reader, cname);
        bi = new arrow::UInt32Builder(pool);
        break;
      case EDataType::kULong64_t:
        field = std::make_shared<arrow::Field>(cname, arrow::uint64());
        vl = new TTreeReaderValue<ULong64_t>(*reader, cname);
        bl = new arrow::UInt64Builder(pool);
        break;
      case EDataType::kShort_t:
        field = std::make_shared<arrow::Field>(cname, arrow::int16());
        var_s = new TTreeReaderValue<Short_t>(*reader, cname);
        bui_s = new arrow::Int16Builder(pool);
        break;
      case EDataType::kInt_t:
        field = std::make_shared<arrow::Field>(cname, arrow::int32());
        var_i = new TTreeReaderValue<Int_t>(*reader, cname);
        bui_i = new arrow::Int32Builder(pool);
        break;
      case EDataType::kLong64_t:
        field = std::make_shared<arrow::Field>(cname, arrow::int64());
        var_l = new TTreeReaderValue<Long64_t>(*reader, cname);
        bui_l = new arrow::Int64Builder(pool);
        break;
      default:
        LOG(FATAL) << "Type not handled: " << dt << std::endl;
        break;
    }
  }

  ~columnIterator()
  {

    // delete all pointers
    if (var_f)
      delete var_f;
    if (var_d)
      delete var_d;
    if (vs)
      delete vs;
    if (vi)
      delete vi;
    if (vl)
      delete vl;
    if (var_s)
      delete var_s;
    if (var_i)
      delete var_i;
    if (var_l)
      delete var_l;

    if (bui_f)
      delete bui_f;
    if (bui_d)
      delete bui_d;
    if (bs)
      delete bs;
    if (bi)
      delete bi;
    if (bl)
      delete bl;
    if (bui_s)
      delete bui_s;
    if (bui_i)
      delete bui_i;
    if (bui_l)
      delete bui_l;
  };

  // has the iterator been properly initialized
  bool Status()
  {
    return status;
  }

  // copy the TTreeReaderValue to the arrow::TBuilder
  void push()
  {
    arrow::Status stat;

    // switch according to dt
    switch (dt) {
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
        LOG(FATAL) << "Type not handled: " << dt << std::endl;
        break;
    }
  }

  std::shared_ptr<arrow::Array> Array()
  {
    return ar;
  }

  std::shared_ptr<arrow::Field> Schema()
  {
    return field;
  }

  // finish the arrow::TBuilder
  // with this ar is prepared to be used in arrow::Table::Make
  void finish()
  {

    //LOG(INFO) << "columnIterator::finish " << dt;

    arrow::Status stat;

    // switch according to dt
    switch (dt) {
      case EDataType::kFloat_t:
        stat = bui_f->Finish(&ar);
        break;
      case EDataType::kDouble_t:
        stat = bui_d->Finish(&ar);
        break;
      case EDataType::kUShort_t:
        stat = bs->Finish(&ar);
        break;
      case EDataType::kUInt_t:
        stat = bi->Finish(&ar);
        break;
      case EDataType::kULong64_t:
        stat = bl->Finish(&ar);
        break;
      case EDataType::kShort_t:
        stat = bui_s->Finish(&ar);
        break;
      case EDataType::kInt_t:
        stat = bui_i->Finish(&ar);
        break;
      case EDataType::kLong64_t:
        stat = bui_l->Finish(&ar);
        break;
      default:
        LOG(FATAL) << "Type not handled: " << dt << std::endl;
        break;
    }
  }
};

// -----------------------------------------------------------------------------
class TreeToTable
{

 private:
  // the TTreeReader allows to efficiently loop over
  // the rows of a TTree
  TTreeReader* reader;

  // a list of columnIterator*
  std::vector<std::shared_ptr<columnIterator>> colits;

  // Append next set of branch values to the
  // corresponding table columns
  void push()
  {
    for (auto colit : colits)
      colit->push();
  }

 public:
  TreeToTable(TTree* tree)
  {
    // initialize the TTreeReader
    reader = new TTreeReader(tree);
  }

  ~TreeToTable()
  {
    if (reader)
      delete reader;
  };

  // add a column to be included in the arrow::table
  bool AddColumn(const char* colname)
  {
    auto colit = std::make_shared<columnIterator>(reader, colname);
    auto stat = colit->Status();
    if (stat)
      colits.push_back(std::move(colit));

    return stat;
  }

  // add all columns
  bool AddAllColumns()
  {
    // get a list of column names
    auto tree = reader->GetTree();
    if (!tree) {
      LOG(INFO) << "Tree not found!";
      return false;
    }
    auto branchList = tree->GetListOfBranches();

    // loop over branches
    bool status = !branchList->IsEmpty();
    for (Int_t ii = 0; ii < branchList->GetEntries(); ii++) {
      auto br = (TBranch*)branchList->At(ii);

      // IMPROVE: make sure that a column is not added more than one time
      auto colit = std::make_shared<columnIterator>(reader, br->GetName());
      if (colit->Status()) {
        colits.push_back(std::move(colit));
      } else {
        status = false;
      }
    }

    return status;
  }

  // do the looping with the TTreeReader and create the table
  std::shared_ptr<arrow::Table> Process()
  {
    // do the looping with the TTreeReader
    Fill();

    // create the table
    return Finalize();
  }

  // do the looping with the TTreeReader
  void Fill()
  {
    // copy all values from the tree to the table builders
    reader->Restart();
    while (reader->Next())
      push();
  }

  // create the table
  std::shared_ptr<arrow::Table> Finalize()
  {

    // prepare the elements needed to create the final table
    std::vector<std::shared_ptr<arrow::Array>> array_vector;
    std::vector<std::shared_ptr<arrow::Field>> schema_vector;
    for (auto colit : colits) {
      colit->finish();
      array_vector.push_back(colit->Array());
      schema_vector.push_back(colit->Schema());
    }
    auto fields = std::make_shared<arrow::Schema>(schema_vector);

    // create the final table
    // ta is of type std::shared_ptr<arrow::Table>
    auto ta = (arrow::Table::Make(fields, array_vector));

    return ta;
  }
};

// -----------------------------------------------------------------------------
} // namespace framework
} // namespace o2

// =============================================================================
#endif // FRAMEWORK_TABLETREE_H

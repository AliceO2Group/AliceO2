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
#include "TTreeReaderArray.h"
#include "TableBuilder.h"

// =============================================================================
namespace o2
{
namespace framework
{

// -----------------------------------------------------------------------------
// TableToTree allows to save the contents of a given arrow::Table to a TTree
//  BranchIterator is used by TableToTree
//
// To write the contents of a table ta to a tree tr on file f do:
//  . TableToTree t2t(ta,f,treename);
//  . t2t.addBranch(coumn1); t2t.addBranch(coumn1); ...
//    OR
//    t2t.addAllBranches();
//  . t2t.process();
//
// .............................................................................
class BranchIterator
{

 private:
  arrow::MemoryPool* mMemoryPool = arrow::default_memory_pool();
  std::string mBranchName;    // branch name
  arrow::ArrayVector mChunks; // chunks
  Int_t mNumberChuncs;        // number of chunks
  Int_t mCounterChunk;        // chunk counter
  Int_t mNumberRows;          // number of rows
  Int_t mCounterRow;          // row counter

  // data buffers for each data type
  bool mStatus = false;
  arrow::Field* mField;
  arrow::Type::type mFieldType;
  arrow::Type::type mElementType;
  int32_t mNumberElements;
  std::string mLeaflistString;

  TBranch* mBranchPtr = nullptr;

  char* mBranchBuffer = nullptr;
  void* mValueBuffer = nullptr;

  std::shared_ptr<arrow::BooleanArray> var_o = nullptr;
  Bool_t boolValueHolder;
  UChar_t* var_ub = nullptr;
  Float_t* var_f = nullptr;
  Double_t* var_d = nullptr;
  UShort_t* var_us = nullptr;
  UInt_t* var_ui = nullptr;
  long unsigned int* var_ul = nullptr;
  Short_t* var_s = nullptr;
  Int_t* var_i = nullptr;
  long int* var_l = nullptr;

  // initialize a branch
  bool initBranch(TTree* tree);

  // initialize chunk ib
  bool initDataBuffer(Int_t ib);

 public:
  BranchIterator(TTree* tree, std::shared_ptr<arrow::ChunkedArray> col, std::shared_ptr<arrow::Field> field);
  ~BranchIterator();

  // has the iterator been properly initialized
  bool getStatus();

  // fills buffer with next value
  // returns false if end of buffer reached
  bool push();
};

class TableToTree
{

 private:
  TTree* mTreePtr;

  // a list of BranchIterator
  std::vector<BranchIterator*> mBranchIterators;

  // table to convert
  std::shared_ptr<arrow::Table> mTable;

 public:
  TableToTree(std::shared_ptr<arrow::Table> table,
              TFile* file,
              const char* treename);
  ~TableToTree();

  // add branches
  bool addBranch(std::shared_ptr<arrow::ChunkedArray> col, std::shared_ptr<arrow::Field> field);
  bool addAllBranches();

  // write table to tree
  TTree* process();
};

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
  TTreeReaderValue<Bool_t>* var_o = nullptr;
  TTreeReaderValue<UChar_t>* var_ub = nullptr;
  TTreeReaderValue<Float_t>* var_f = nullptr;
  TTreeReaderValue<Double_t>* var_d = nullptr;
  TTreeReaderValue<UShort_t>* var_us = nullptr;
  TTreeReaderValue<UInt_t>* var_ui = nullptr;
  TTreeReaderValue<long unsigned int>* var_ul = nullptr;
  TTreeReaderValue<Short_t>* var_s = nullptr;
  TTreeReaderValue<Int_t>* var_i = nullptr;
  TTreeReaderValue<long int>* var_l = nullptr;

  // all the possible TTreeReaderArray<T> types
  TTreeReaderArray<Bool_t>* arr_o = nullptr;
  TTreeReaderArray<UChar_t>* arr_ub = nullptr;
  TTreeReaderArray<Float_t>* arr_f = nullptr;
  TTreeReaderArray<Double_t>* arr_d = nullptr;
  TTreeReaderArray<UShort_t>* arr_us = nullptr;
  TTreeReaderArray<UInt_t>* arr_ui = nullptr;
  TTreeReaderArray<long unsigned int>* arr_ul = nullptr;
  TTreeReaderArray<Short_t>* arr_s = nullptr;
  TTreeReaderArray<Int_t>* arr_i = nullptr;
  TTreeReaderArray<long int>* arr_l = nullptr;

  // all the possible arrow::TBuilder types
  std::shared_ptr<arrow::FixedSizeListBuilder> bui_list;

  arrow::BooleanBuilder* bui_o = nullptr;
  arrow::UInt8Builder* bui_ub = nullptr;
  arrow::FloatBuilder* bui_f = nullptr;
  arrow::DoubleBuilder* bui_d = nullptr;
  arrow::UInt16Builder* bui_us = nullptr;
  arrow::UInt32Builder* bui_ui = nullptr;
  arrow::UInt64Builder* bui_ul = nullptr;
  arrow::Int16Builder* bui_s = nullptr;
  arrow::Int32Builder* bui_i = nullptr;
  arrow::Int64Builder* bui_l = nullptr;

  bool mStatus = false;
  EDataType mElementType;
  int64_t mNumberElements;
  const char* mColumnName;

  arrow::MemoryPool* mMemoryPool = arrow::default_memory_pool();
  std::shared_ptr<arrow::Field> mField;
  std::shared_ptr<arrow::Array> mArray;

 public:
  ColumnIterator(TTreeReader* reader, const char* colname);
  ~ColumnIterator();

  // has the iterator been properly initialized
  bool getStatus();

  // copy the TTreeReaderValue to the arrow::TBuilder
  void push();

  std::shared_ptr<arrow::Array> getArray() { return mArray; }
  std::shared_ptr<arrow::Field> getSchema() { return mField; }

  // finish the arrow::TBuilder
  // with this mArray is prepared to be used in arrow::Table::Make
  void finish();
};

class TreeToTable
{

 private:
  // the TTreeReader allows to efficiently loop over
  // the rows of a TTree
  TTreeReader* mTreeReader;

  // a list of ColumnIterator*
  std::vector<std::shared_ptr<ColumnIterator>> mColumnIterators;

  // Append next set of branch values to the
  // corresponding table columns
  void push();

 public:
  TreeToTable(TTree* tree);
  ~TreeToTable();

  // add a column to be included in the arrow::table
  bool addColumn(const char* colname);

  // add all columns
  bool addAllColumns();

  // do the looping with the TTreeReader
  void fill();

  // create the table
  std::shared_ptr<arrow::Table> finalize();

  // do the looping with the TTreeReader and create the table
  std::shared_ptr<arrow::Table> process();
};

// -----------------------------------------------------------------------------
} // namespace framework
} // namespace o2

// =============================================================================
#endif // FRAMEWORK_TABLETREE_H

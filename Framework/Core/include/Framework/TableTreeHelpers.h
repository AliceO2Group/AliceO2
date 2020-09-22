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

  std::shared_ptr<arrow::BooleanArray> mArray_o = nullptr;
  //bool mBoolValueHolder;
  bool* mVariable_o = nullptr;

  uint8_t* mVariable_ub = nullptr;
  uint16_t* mVariable_us = nullptr;
  uint32_t* mVariable_ui = nullptr;
  uint64_t* mVariable_ul = nullptr;
  int8_t* mVariable_b = nullptr;
  int16_t* mVariable_s = nullptr;
  int32_t* mVariable_i = nullptr;
  int64_t* mVariable_l = nullptr;
  float* mVariable_f = nullptr;
  double* mVariable_d = nullptr;

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
  std::shared_ptr<arrow::FixedSizeListBuilder> mTableBuilder_list;

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
  ColumnIterator(TTreeReader* reader, const char* colname);
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

  // reserve enough space to push s rows without reallocating
  void reserve(size_t s);

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

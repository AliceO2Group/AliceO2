// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_TABLETREEHELPERS_H_
#define O2_FRAMEWORK_TABLETREEHELPERS_H_

#include "TFile.h"
#include "TTreeReader.h"
#include "TTreeReaderValue.h"
#include "TTreeReaderArray.h"
#include "TableBuilder.h"

// =============================================================================
namespace o2::framework
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

class TreeToTable
{

 private:
  std::shared_ptr<arrow::Table> mTable;
  std::vector<std::string> mColumnNames;

 public:
  // add a column to be included in the arrow::table
  void addColumn(const char* colname);

  // add all branches in @a tree as columns
  bool addAllColumns(TTree* tree);

  // do the looping with the TTreeReader
  void fill(TTree* tree);

  // create the table
  std::shared_ptr<arrow::Table> finalize();
};

// -----------------------------------------------------------------------------
} // namespace o2::framework

// =============================================================================
#endif // O2_FRAMEWORK_TABLETREEHELPERS_H_

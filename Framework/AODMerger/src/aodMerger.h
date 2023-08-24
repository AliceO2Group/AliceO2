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

#include <TString.h>

const char* removeVersionSuffix(const char* treeName)
{
  // remove version suffix, e.g. O2v0_001 becomes O2v0
  static TString tmp;
  tmp = treeName;
  if (tmp.First("_") >= 0) {
    tmp.Remove(tmp.First("_"));
  }
  return tmp;
}

const char* getTableName(const char* branchName, const char* treeName)
{
  // Syntax for branchName:
  //   fIndex<Table>[_<Suffix>]
  //   fIndexArray<Table>[_<Suffix>]
  //   fIndexSlice<Table>[_<Suffix>]
  // if <Table> is empty it is a self index and treeName is used as table name
  static TString tableName;
  tableName = branchName;
  if (tableName.BeginsWith("fIndexArray") || tableName.BeginsWith("fIndexSlice")) {
    tableName.Remove(0, 11);
  } else {
    tableName.Remove(0, 6);
  }
  if (tableName.First("_") >= 0) {
    tableName.Remove(tableName.First("_"));
  }
  if (tableName.Length() == 0) {
    return removeVersionSuffix(treeName);
  }
  tableName.Remove(tableName.Length() - 1); // remove s
  tableName.ToLower();
  tableName = "O2" + tableName;
  // printf("%s --> %s\n", branchName, tableName.Data());
  return tableName;
}

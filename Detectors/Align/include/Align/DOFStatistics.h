// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   DOFStatistics.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Mergable Object for statistics of points used by each DOF

#ifndef DOFStatistics_H
#define DOFStatistics_H

#include <TNamed.h>
class TH1F;
class TCollection;

namespace o2
{
namespace align
{

class Controller;

class DOFStatistics : public TNamed
{
 public:
  DOFStatistics(int n = 0);
  virtual ~DOFStatistics();
  //
  int getNDOFs() const { return mNDOFs; }
  int getStat(int idf) const { return idf < mNDOFs ? mStat[idf] : 0; }
  int* getStat() const { return (int*)mStat; };
  void setStat(int idf, int v) { mStat[idf] = v; }
  void addStat(int idf, int v) { mStat[idf] += v; }
  int getNMerges() const { return mNMerges; }
  TH1F* createHisto(Controller* st) const;
  virtual void Print(Option_t* opt) const;
  virtual int64_t merge(TCollection* list);
  //
 protected:
  //
  DOFStatistics(const DOFStatistics&);
  DOFStatistics& operator=(const DOFStatistics&);
  //
 protected:
  int mNDOFs;   // number of dofs defined
  int mNMerges; // number of merges
  int* mStat;   //[mNDOFs] statistics per DOF
  //
  ClassDef(DOFStatistics, 1);
};
} // namespace align
} // namespace o2
#endif

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

#include <vector>
#include <TNamed.h>
#include <iostream>

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
  explicit DOFStatistics(int n = 0) : TNamed("DOFStatistics", "DOF statistics"), mStat{n, 0} {};

  inline int getNDOFs() const noexcept { return mStat.size(); }
  inline int getStat(int idf) const noexcept { return idf < getNDOFs() ? mStat[idf] : 0; }
  inline const int* getStat() const noexcept { return mStat.data(); };
  inline void setStat(int idf, int v) { mStat[idf] = v; }
  inline void addStat(int idf, int v) { mStat[idf] += v; }
  inline int getNMerges() const noexcept { return mNMerges; }
  std::unique_ptr<TH1F> buildHistogram(Controller* st) const;
  void Print(Option_t* opt) const final { std::cout << "NDOFs: " << mStat.size() << " NMerges: " << mNMerges << "\n"; };
  int64_t merge(TCollection* list);

 protected:
  DOFStatistics(const DOFStatistics&);
  DOFStatistics& operator=(const DOFStatistics&);

 protected:
  int mNMerges{1};        // number of merges
  std::vector<int> mStat; // statistics per DOF

  ClassDef(DOFStatistics, 1);
};
} // namespace align
} // namespace o2
#endif

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   AliAlgDOFStat.h
/// @author ruben.shahoyan@cern.ch, michael.lettrich@cern.ch
/// @since  2021-02-01
/// @brief  Mergable bbject for statistics of points used by each DOF

#ifndef ALIALGDOFSTAT_H
#define ALIALGDOFSTAT_H

#include <TNamed.h>
class TH1F;
class TCollection;

namespace o2
{
namespace align
{

class AliAlgSteer;

class AliAlgDOFStat : public TNamed
{
 public:
  AliAlgDOFStat(int n = 0);
  virtual ~AliAlgDOFStat();
  //
  int GetNDOFs() const { return fNDOFs; }
  int GetStat(int idf) const { return idf < fNDOFs ? fStat[idf] : 0; }
  int* GetStat() const { return (int*)fStat; };
  void SetStat(int idf, int v) { fStat[idf] = v; }
  void AddStat(int idf, int v) { fStat[idf] += v; }
  int GetNMerges() const { return fNMerges; }
  TH1F* CreateHisto(AliAlgSteer* st) const;
  virtual void Print(Option_t* opt) const;
  virtual int64_t Merge(TCollection* list);
  //
 protected:
  //
  AliAlgDOFStat(const AliAlgDOFStat&);
  AliAlgDOFStat& operator=(const AliAlgDOFStat&);
  //
 protected:
  int fNDOFs;   // number of dofs defined
  int fNMerges; // number of merges
  int* fStat;   //[fNDOFs] statistics per DOF
  //
  ClassDef(AliAlgDOFStat, 1);
};
} // namespace align
} // namespace o2
#endif

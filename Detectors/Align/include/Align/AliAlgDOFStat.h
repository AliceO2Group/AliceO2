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
//class AliAlgSteer;
class TH1F;
class TCollection;

namespace o2
{
namespace align
{

class AliAlgDOFStat : public TNamed
{
 public:
  AliAlgDOFStat(Int_t n = 0);
  virtual ~AliAlgDOFStat();
  //
  Int_t GetNDOFs() const { return fNDOFs; }
  Int_t GetStat(int idf) const { return idf < fNDOFs ? fStat[idf] : 0; }
  Int_t* GetStat() const { return (Int_t*)fStat; };
  void SetStat(int idf, int v) { fStat[idf] = v; }
  void AddStat(int idf, int v) { fStat[idf] += v; }
  Int_t GetNMerges() const { return fNMerges; }
  // FIXME(milettri): needs AliAlgSteer
  //  TH1F* CreateHisto(AliAlgSteer* st) const;
  virtual void Print(Option_t* opt) const;
  virtual Long64_t Merge(TCollection* list);
  //
 protected:
  //
  AliAlgDOFStat(const AliAlgDOFStat&);
  AliAlgDOFStat& operator=(const AliAlgDOFStat&);
  //
 protected:
  Int_t fNDOFs;   // number of dofs defined
  Int_t fNMerges; // number of merges
  Int_t* fStat;   //[fNDOFs] statistics per DOF
  //
  ClassDef(AliAlgDOFStat, 1);
};
} // namespace align
} // namespace o2
#endif

// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
// Cut class manipulating groups of cuts
//

#ifndef AnalysisCompositeCut_H
#define AnalysisCompositeCut_H

#include "PWGDQCore/AnalysisCut.h"
#include <vector>

//_________________________________________________________________________
class AnalysisCompositeCut : public AnalysisCut
{
 public:
  AnalysisCompositeCut(bool useAND = kTRUE);
  AnalysisCompositeCut(const char* name, const char* title, bool useAND = kTRUE);
  AnalysisCompositeCut(const AnalysisCompositeCut& c) = default;
  AnalysisCompositeCut& operator=(const AnalysisCompositeCut& c) = default;
  ~AnalysisCompositeCut() override;

  void AddCut(AnalysisCut* cut)
  {
    if (cut->IsA() == AnalysisCompositeCut::Class()) {
      fCompositeCutList.push_back(*(AnalysisCompositeCut*)cut);
    } else {
      fCutList.push_back(*cut);
    }
  };

  bool GetUseAND() const { return fOptionUseAND; }
  int GetNCuts() const { return fCutList.size() + fCompositeCutList.size(); }

  bool IsSelected(float* values) override;

 protected:
  bool fOptionUseAND;                                  // true (default): apply AND on all cuts; false: use OR
  std::vector<AnalysisCut> fCutList;                   // list of cuts
  std::vector<AnalysisCompositeCut> fCompositeCutList; // list of composite cuts

  ClassDef(AnalysisCompositeCut, 2);
};

#endif

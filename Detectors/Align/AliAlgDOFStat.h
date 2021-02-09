#ifndef ALIALGDOFSTAT_H
#define ALIALGDOFSTAT_H

/*--------------------------------------------------------
  Mergable bbject for statistics of points used by each DOF
  -------------------------------------------------------*/

// Author: ruben.shahoyan@cern.ch

#include <TNamed.h>
class AliAlgSteer;
class TH1F;
class TCollection;

class AliAlgDOFStat : public TNamed
{
 public:
  AliAlgDOFStat(Int_t n=0);
  virtual ~AliAlgDOFStat();
  //
  Int_t          GetNDOFs()                   const {return fNDOFs;}
  Int_t          GetStat(int idf)             const {return idf<fNDOFs ? fStat[idf] : 0;}
  Int_t*         GetStat()                    const {return (Int_t*)fStat;};
  void           SetStat(int idf,int v)             {fStat[idf] = v;}
  void           AddStat(int idf,int v)             {fStat[idf] += v;}
  Int_t          GetNMerges()                 const {return fNMerges;}
  //
  TH1F*          CreateHisto(AliAlgSteer* st) const;
  virtual void   Print(Option_t* opt)         const;
  virtual Long64_t Merge(TCollection* list);
  //
 protected:
  //
  AliAlgDOFStat(const AliAlgDOFStat&);
  AliAlgDOFStat& operator=(const AliAlgDOFStat&);
  //
 protected:

  Int_t fNDOFs;                // number of dofs defined
  Int_t fNMerges;              // number of merges
  Int_t *fStat;                //[fNDOFs] statistics per DOF
  //
  ClassDef(AliAlgDOFStat,1);
};


#endif

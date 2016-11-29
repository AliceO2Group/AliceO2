//
//  HitAnalysis.h
//  ALICEO2
//
//  Created by Markus Fasel on 28.07.15.
//
//

#ifndef __ALICEO2__HitAnalysis__
#define __ALICEO2__HitAnalysis__

#include <map>
#include "FairTask.h"  // for FairTask, InitStatus
#include "Rtypes.h"    // for Bool_t, HitAnalysis::Class, ClassDef, etc

class TClonesArray;  // lines 17-17
class TH1;  // lines 16-16
namespace AliceO2 { namespace ITS { class GeometryTGeo; }}  // lines 23-23


class TH1;

class TClonesArray;

namespace AliceO2 {
namespace ITS {

class Chip;

class HitAnalysis : public FairTask
{
  public:
    HitAnalysis();

    virtual ~HitAnalysis();

    virtual InitStatus Init();

    virtual void Exec(Option_t *option);

    virtual void FinishTask();

    void SetProcessHits()
    { fProcessChips = kFALSE; }

    void SetProcessChips()
    { fProcessChips = kTRUE; }

  protected:
    void ProcessChips();

    void ProcessHits();

  private:
    Bool_t fIsInitialized;       ///< Check whether task is initialized
    Bool_t fProcessChips;        ///< Process chips or hits
    std::map<int, Chip *> fChips;               ///< lookup map for ITS chips
    TClonesArray *fPointsArray;        ///< Array with ITS space points, filled by the FairRootManager
    GeometryTGeo *fGeometry;           ///<  geometry
    TH1 *fLineSegment;        ///< Histogram for line segment
    TH1 *fLocalX0;            ///< Histogram for Starting X position in local coordinates
    TH1 *fLocalX1;            ///< Histogram for Hit X position in local coordinates
    TH1 *fLocalY0;            ///< Histogram for Starting Y position in local coordinates
    TH1 *fLocalY1;            ///< Histogram for Hit Y position in local coordinates
    TH1 *fLocalZ0;            ///< Histogram for Starting Z position in local coordinates
    TH1 *fLocalZ1;            ///< Histogram for Hit Z position in local coordinates
    TH1 *fHitCounter;         ///< simple hit counter histogram

  ClassDef(HitAnalysis, 1);
};
}
}

#endif /* defined(__ALICEO2__HitAnalysis__) */

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
namespace o2 { namespace ITS { class GeometryTGeo; }}  // lines 23-23


class TH1;

class TClonesArray;

namespace o2 {
namespace ITSMFT {
class Chip;
}
}

namespace o2 {
namespace ITS {

class HitAnalysis : public FairTask
{
  public:
    HitAnalysis();

    ~HitAnalysis() override;

    InitStatus Init() override;

    void Exec(Option_t *option) override;

    void FinishTask() override;

    void SetProcessHits()
    { mProcessChips = kFALSE; }

    void SetProcessChips()
    { mProcessChips = kTRUE; }

  protected:
    void ProcessChips();

    void ProcessHits();

  private:
    Bool_t mIsInitialized;       ///< Check whether task is initialized
    Bool_t mProcessChips;        ///< Process chips or hits
    std::map<int, o2::ITSMFT::Chip *> mChips; ///< lookup map for ITS chips
    TClonesArray *mPointsArray;        ///< Array with ITS space points, filled by the FairRootManager
    GeometryTGeo *mGeometry;           ///<  geometry
    TH1 *mLineSegment;        ///< Histogram for line segment
    TH1 *mLocalX0;            ///< Histogram for Starting X position in local coordinates
    TH1 *mLocalX1;            ///< Histogram for Hit X position in local coordinates
    TH1 *mLocalY0;            ///< Histogram for Starting Y position in local coordinates
    TH1 *mLocalY1;            ///< Histogram for Hit Y position in local coordinates
    TH1 *mLocalZ0;            ///< Histogram for Starting Z position in local coordinates
    TH1 *mLocalZ1;            ///< Histogram for Hit Z position in local coordinates
    TH1 *mHitCounter;         ///< simple hit counter histogram

  ClassDefOverride(HitAnalysis, 1);
};
}
}

#endif /* defined(__ALICEO2__HitAnalysis__) */

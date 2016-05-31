//
//  DigitContainer.h
//  ALICEO2
//
//  Created by Markus Fasel on 25.03.15.
//
//

#ifndef _ALICEO2_ITS_DigitContainer_
#define _ALICEO2_ITS_DigitContainer_

class TClonesArray;  // lines 14-14
namespace AliceO2 { namespace ITS { class Digit; }}  // lines 19-19
namespace AliceO2 { namespace ITS { class DigitLayer; }}  // lines 20-20
namespace AliceO2 { namespace ITS { class UpgradeGeometryTGeo; }}


namespace AliceO2 {
namespace ITS {

class Digit;

class DigitLayer;

class DigitContainer
{
  public:
    DigitContainer(const UpgradeGeometryTGeo *geo);

    ~DigitContainer();

    void Reset();

    void AddDigit(Digit *digi);

    Digit *FindDigit(int layer, int stave, int index);

    void FillOutputContainer(TClonesArray *output);

  private:
    DigitLayer *fDigitLayer[7];
    const UpgradeGeometryTGeo *fGeometry;
};
}
}

#endif /* defined(__ALICEO2__DigitContainer__) */

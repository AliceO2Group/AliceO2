#ifndef ALICEO2_GENERATORGEANTINOS_H
#define ALICEO2_GENERATORGEANTINOS_H

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//                                                                           //
//    Utility class to compute and draw Radiation Length Map                 //
//                                                                           //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "FairGenerator.h"
#include <TClonesArray.h>
namespace o2
{
namespace eventgen
{
class GeneratorGeantinos:
public FairGenerator
{

public: 
    GeneratorGeantinos();
    GeneratorGeantinos(Int_t mode, Int_t nc1, Float_t c1min, Float_t c1max,
		     Int_t nc2, Float_t c2min, Float_t c2max,
		     Float_t rmin, Float_t rmax, Float_t zmax);
    virtual ~GeneratorGeantinos() {}
    Bool_t ReadEvent(FairPrimaryGenerator* primGen) override;
    // Getters	    
    virtual Float_t ZMax()   const {return mZMax;}
    virtual Float_t RadMax() const {return mRadMax;}
    virtual Int_t   NCoor1() const {return mNCoor1;}
    virtual Int_t   NCoor2() const {return mNCoor2;}
    // Helpers    
    Float_t         PropagateCylinder(Float_t *x, Float_t *v, Float_t r, Float_t z);
 protected:
    Int_t      mMode;               // generation mode
    Float_t    mRadMin;             // Generation radius
    Float_t    mRadMax;             // Maximum tracking radius
    Float_t    mZMax;               // Maximum tracking Z
    Int_t      mNCoor1;             // Number of bins in Coor1
    Int_t      mNCoor2;             // Number of bins in Coor2

    Float_t    mCoor1Min;           // Minimum Coor1
    Float_t    mCoor1Max;           // Maximum Coor1
    Float_t    mCoor2Min;           // Minimum Coor2
    Float_t    mCoor2Max;           // Maximum Coor2 
};
}
}
#endif


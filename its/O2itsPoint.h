#ifndef O2ITSPOINT_H
#define O2ITSPOINT_H


#include "FairMCPoint.h"

#include "TObject.h"
#include "TVector3.h"

class O2itsPoint : public FairMCPoint
{

  public:

    /** Default constructor **/
    O2itsPoint();

    /** Constructor with arguments
     *@param trackID    Index of MCTrack
     *@param detID      Detector ID
     *@param startPos   Coordinates at entrance to active volume [cm]
     *@param pos        Coordinates to active volume [cm]
     *@param mom        Momentum of track at entrance [GeV]
     *@param startTime  Time at entrance [ns]
     *@param time       Time since event start [ns]
     *@param length     Track length since creation [cm]
     *@param eLoss      Energy deposit [GeV]
     *@param shunt      Shunt value
     **/
    O2itsPoint(Int_t trackID, Int_t detID, TVector3 startPos, TVector3 pos, TVector3 mom,
                     Double_t startTime, Double_t time, Double_t length, Double_t eLoss,
                     Int_t shunt);

    /** Destructor **/
    virtual ~O2itsPoint();

    /** Output to screen **/
    virtual void Print(const Option_t* opt) const;
    
    /** Accessors **/
    Double_t GetXIn()   const { return fX; }
    Double_t GetYIn()   const { return fY; }
    Double_t GetZIn()   const { return fZ; }
    Double_t GetXOut()  const { return fX_out; }
    Double_t GetYOut()  const { return fY_out; }
    Double_t GetZOut()  const { return fZ_out; }
    Double_t GetPxOut() const { return fPx_out; }
    Double_t GetPyOut() const { return fPy_out; }
    Double_t GetPzOut() const { return fPz_out; }
    Double_t GetPxIn()  const { return fPx; }
    Double_t GetPyIn()  const { return fPy; }
    Double_t GetPzIn()  const { return fPz; }

  private:
  
    Double32_t fX_out;
    Double32_t fY_out;
    Double32_t fZ_out;
    Double32_t fPx_out;
    Double32_t fPy_out;
    Double32_t fPz_out;
  
    /** Copy constructor **/
    O2itsPoint(const O2itsPoint& point);
    O2itsPoint operator=(const O2itsPoint& point);

  ClassDef(O2itsPoint,1)
};

#endif

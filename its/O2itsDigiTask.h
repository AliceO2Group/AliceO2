#ifndef O2ITSDIGITASK_H_
#define O2ITSDIGITASK_H_

#include "FairTask.h"                   // for FairTask, InitStatus

#include "Rtypes.h"                     // for Double_t, etc

class TClonesArray;

class O2itsDigiTask : public FairTask
{
  public:

    /** Default constructor **/
    O2itsDigiTask();

    /** Destructor **/
    ~O2itsDigiTask();

    /** Virtual method Init **/
    virtual InitStatus Init();

    /** Virtual method Exec **/
    virtual void Exec(Option_t* opt);

    void SetTimeResolution(Double_t timeInNs) { fTimeResolution = timeInNs; }
    Double_t GetTimeResolution() { return fTimeResolution; }

  private:

    Int_t CalcPad(Double_t posIn, Double_t posOut);
    Double_t CalcTimeStamp(Double_t timeOfFlight);

    Double_t fTimeResolution;

    TClonesArray* fPointArray;
    TClonesArray* fDigiArray;

    O2itsDigiTask(const O2itsDigiTask&);
    O2itsDigiTask& operator=(const O2itsDigiTask&);

    ClassDef(O2itsDigiTask,1);
};

#endif

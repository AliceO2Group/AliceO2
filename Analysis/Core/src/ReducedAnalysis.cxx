/*
 Implementation of ReducedAnalysis 
 */

#include "Analysis/ReducedAnalysis.h"

ClassImp(ReducedAnalysis);

//___________________________________________________________________________
ReducedAnalysis::ReducedAnalysis() :
  TObject(TNamed()),
  fEventCounter(0)
{
  //
  // default constructor
  //
}


//___________________________________________________________________________
ReducedAnalysis::ReducedAnalysis(const char* name, const char* title) :
  TObject(TNamed(name, title)),
  fEventCounter(0)
{
  //
  // named constructor
  //
}


//___________________________________________________________________________
ReducedAnalysis::~ReducedAnalysis() 
{
  //
  // destructor
  //
}

//___________________________________________________________________________
void ReducedAnalysis::Init() {
   //
   // initialization (typically called in AliAnalysisTask::UserCreateOutputObjects())
   //
}

//___________________________________________________________________________
void ReducedAnalysis::Process() {
   //
   // process a given event (typically called in AliAnalysisTask::UserExec())
   //
}

//___________________________________________________________________________
void ReducedAnalysis::Finish() {
   //
   // finish, to be executed after all events were processed
   //
}

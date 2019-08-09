// $Id$
/**
 * @file createTPCFastTransform.C
 * @brief A macro to create TPCFastTransform object
 *
 * <pre>
 * Usage:
 *
 * aliroot
 * .x initTPCcalibration.C("alien://Folder=/alice/data/2015/OCDB",246984,1)
 *  gSystem->Load("libAliTPCFastTransformation")
 * .L createTPCFastTransform.C++
 * GPUCA_NAMESPACE::gpu::TPCFastTransform fastTransform;
 * createTPCFastTransform(fastTransform);
 *
 * </pre>
 *
 * @author sergey gorbunov
 *
 */

#include "AliTPCcalibDB.h"
#include "Riostream.h"
#include "TStopwatch.h"

#define GPUCA_ALIROOT_LIB

#include "TPCFastTransform.h"
#include "TPCFastTransformManager.h"
#include "TPCFastTransformQA.h"

using namespace std;
using namespace GPUCA_NAMESPACE::gpu;

int createTPCFastTransform(TPCFastTransform& fastTransform)
{

  AliTPCcalibDB* tpcCalib = AliTPCcalibDB::Instance();
  if (!tpcCalib) {
    cerr << "AliTPCcalibDB does not exist" << endl;
    return -1;
  }
  AliTPCTransform* origTransform = tpcCalib->GetTransform();
  UInt_t timeStamp = origTransform->GetCurrentTimeStamp();

  TPCFastTransformManager manager;

  TStopwatch timer;
  timer.Start();

  int err = manager.create(fastTransform, origTransform, timeStamp);

  timer.Stop();

  cout << "\n\n Initialisation: " << timer.CpuTime() << " / " << timer.RealTime() << " sec.\n\n"
       << endl;

  if (err != 0) {
    cerr << "Cannot create fast transformation object from AliTPCcalibDB, TPCFastTransformManager returns  " << err << endl;
    return -1;
  }

  // qa

  // GPUCA_NAMESPACE::gpu::TPCFastTransformQA qa;
  // qa.doQA( timeStamp );

  return 0;
}

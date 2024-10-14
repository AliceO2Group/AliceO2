// $Id$
/**
 * @file moveTPCFastTransform.C
 * @brief Example of creation of TPCFastTransform object and moving it to another place
 *
 * <pre>
 * Usage:
 *
 * aliroot
 *  gSystem->Load("libAliTPCFastTransformation")
 * .L initTPCcalibration.C++
 * .L createTPCFastTransform.C++
 * .x moveTPCFastTransform.C
 *
 * </pre>
 *
 * @author sergey gorbunov
 *
 */

#include "TPCFastTransform.h"

using namespace std;
using namespace GPUCA_NAMESPACE::gpu;

int32_t moveTPCFastTransform()
{

  // gSystem->Load("libAliTPCFastTransformation");
  // gROOT->LoadMacro("initTPCcalibration.C++");
  // gROOT->LoadMacro("createTPCFastTransform.C++");

  initTPCcalibration("alien://Folder=/alice/data/2015/OCDB", 246984, 1);

  TPCFastTransform fastTransform;
  createTPCFastTransform(fastTransform);

  // make flat buffer external

  std::unique_ptr<char[]> buff(fastTransform.releaseInternalBuffer());

  // example of moving the transformation object to another place

  {
    char* newBuff = new char[fastTransform.getFlatBufferSize()];
    char* newObj = new char[sizeof(TPCFastTransform)];

    memcpy((void*)newObj, (void*)&fastTransform, sizeof(fastTransform));
    memcpy((void*)newBuff, (void*)buff.get(), fastTransform.getFlatBufferSize());

    TPCFastTransform& newTransform = *(TPCFastTransform*)newObj;
    newTransform.setActualBufferAddress(newBuff);
  }

  // another example of moving the transformation object to another place
  {
    char* newBuff = new char[fastTransform.getFlatBufferSize()];
    char* newObj = new char[sizeof(TPCFastTransform)];

    fastTransform.setFutureBufferAddress(newBuff);

    memcpy((void*)newObj, (void*)&fastTransform, sizeof(fastTransform));
    memcpy((void*)newBuff, (void*)buff.get(), fastTransform.getFlatBufferSize());

    TPCFastTransform& newTransform = *(TPCFastTransform*)newObj;
  }

  return 0;
}

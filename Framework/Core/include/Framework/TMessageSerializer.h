// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_TMESSAGESERIALIZER_H
#define FRAMEWORK_TMESSAGESERIALIZER_H

#include <TMessage.h>
#include <TClonesArray.h>

namespace o2 {
namespace framework {

class FairTMessage : public TMessage
{
  public:
    FairTMessage(void* buf, Int_t len)
        : TMessage(buf, len)
    {
        ResetBit(kIsOwner);
    }
};

// helper function to clean up the object holding the data after it is transported.
static void free_tmessage(void* /*data*/, void* hint)
{
    delete (TMessage*)hint;
}

struct TMessageSerializer
{
  void Serialize(FairMQMessage& msg, TClonesArray* input)
  {
    TMessage* tm = new TMessage(kMESS_OBJECT);
    tm->WriteObject(static_cast<TObject *>(input));
    msg.Rebuild(tm->Buffer(), tm->BufferSize(), free_tmessage, tm);
  }

  void Deserialize(FairMQMessage& msg, TClonesArray*& output)
  {
    FairTMessage tm(msg.GetData(), msg.GetSize());
    output = reinterpret_cast<TClonesArray*>(tm.ReadObject(tm.GetClass()));
  }
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_TMESSAGESERIALIZER_H

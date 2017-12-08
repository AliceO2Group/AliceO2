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

#include <fairmq/FairMQMessage.h>

#include <TMessage.h>
#include <TObjArray.h>
#include <TList.h>
#include <mutex>
#include <TStreamerInfo.h>
#include <memory>

namespace o2 {
namespace framework {

class FairTMessage : public TMessage
{
  public:
    using TMessage::TMessage;
    FairTMessage(void* buf, Int_t len)
        : TMessage(buf, len)
    {
        ResetBit(kIsOwner);
    }

    // helper function to clean up the object holding the data after it is transported.
    static void free(void* /*data*/, void* hint)
    {
      std::default_delete<FairTMessage> deleter;
      deleter(static_cast<FairTMessage*>(hint));
    }
};

struct TMessageSerializer
{
  using StreamerList = std::vector<TVirtualStreamerInfo*>;
  using CompressionLevel = int;
  enum class CacheStreamers { yes, no };

  void Serialize(FairMQMessage& msg, TObject* input, CacheStreamers streamers = CacheStreamers::no,
                 CompressionLevel compressionLevel = -1)
  {
    std::unique_ptr<FairTMessage> tm = std::make_unique<FairTMessage>(kMESS_OBJECT);

    if (streamers==CacheStreamers::yes) {
      tm->EnableSchemaEvolution(true);
    }

      if (compressionLevel >= 0) {
      // if negative, skip to use ROOT default
      tm->SetCompressionLevel(compressionLevel);
    }

    tm->WriteObject(static_cast<TObject*>(input));

    if (streamers==CacheStreamers::yes) {
      updateStreamers(*tm, sStreamers);
    }

    msg.Rebuild(tm->Buffer(), tm->BufferSize(), FairTMessage::free, tm.get());
    tm.release();
  }

  void Deserialize(const FairMQMessage& msg, std::unique_ptr<TObject>& output)
  {
    // we know the message will not be modified by this,
    // so const_cast should be OK here(IMHO).
    FairTMessage tm(const_cast<FairMQMessage&>(msg).GetData(), const_cast<FairMQMessage&>(msg).GetSize());
    output.reset(reinterpret_cast<TObject*>(tm.ReadObject(tm.GetClass())));
  }

  // load the schema information from a message
  void loadSchema(const FairMQMessage& msg);

  // write the schema into an empty message
  void fillSchema(FairMQMessage& msg, const StreamerList& streamers);

  //get the streamers
  static StreamerList getStreamers()
  {
    std::lock_guard<std::mutex> lock{ TMessageSerializer::sStreamersLock };
    return sStreamers;
  }

 private:

  // update the cache of streamer infos for serialized classes
  void updateStreamers(const TMessage& message, StreamerList& streamers);

  // for now this is a static, maybe it would be better to move the storage somewhere else?
  static StreamerList sStreamers;
  static std::mutex sStreamersLock;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_TMESSAGESERIALIZER_H

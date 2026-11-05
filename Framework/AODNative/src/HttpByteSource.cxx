// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "HttpByteSource.h"
#include "AODNative/Errors.h"
#include <curl/curl.h>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace o2::aod::native
{

namespace
{
// Larger ranges are split, so that the pieces go out in parallel.
constexpr size_t kMaxRequest = 16 * 1024 * 1024;

void globalInit()
{
  static std::once_flag once;
  std::call_once(once, [] { curl_global_init(CURL_GLOBAL_DEFAULT); });
}

/// One ranged GET, written straight into the caller's memory.
struct Transfer {
  uint64_t offset;
  size_t length;
  uint8_t* dst;
  size_t received = 0;
  bool overflow = false;
  CURL* handle = nullptr;
  char error[CURL_ERROR_SIZE] = {};
};

size_t writeInto(char* data, size_t size, size_t n, void* userdata)
{
  auto* t = (Transfer*)userdata;
  auto bytes = size * n;
  // More than asked for: the server ignored the range. Abort rather than
  // downloading the whole file.
  if (t->received + bytes > t->length) {
    t->overflow = true;
    return 0;
  }
  std::memcpy(t->dst + t->received, data, bytes);
  t->received += bytes;
  return bytes;
}

size_t discard(char*, size_t size, size_t n, void*) { return size * n; }
} // namespace

/// What all the requests on a source share: connections, TLS sessions and DNS.
struct HttpByteSource::Shared {
  CURLSH* share = nullptr;
  std::mutex locks[CURL_LOCK_DATA_LAST];

  Shared()
  {
    share = curl_share_init();
    curl_share_setopt(share, CURLSHOPT_LOCKFUNC, +[](CURL*, curl_lock_data data, curl_lock_access, void* self) { ((Shared*)self)->locks[data].lock(); });
    curl_share_setopt(share, CURLSHOPT_UNLOCKFUNC, +[](CURL*, curl_lock_data data, void* self) { ((Shared*)self)->locks[data].unlock(); });
    curl_share_setopt(share, CURLSHOPT_USERDATA, this);
    curl_share_setopt(share, CURLSHOPT_SHARE, CURL_LOCK_DATA_CONNECT);
    curl_share_setopt(share, CURLSHOPT_SHARE, CURL_LOCK_DATA_SSL_SESSION);
    curl_share_setopt(share, CURLSHOPT_SHARE, CURL_LOCK_DATA_DNS);
  }
  ~Shared() { curl_share_cleanup(share); }

  CURL* handle(std::string const& url, Options const& options) const
  {
    auto* h = curl_easy_init();
    curl_easy_setopt(h, CURLOPT_URL, url.c_str());
    curl_easy_setopt(h, CURLOPT_SHARE, share);
    curl_easy_setopt(h, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(h, CURLOPT_MAXREDIRS, 10L);
    curl_easy_setopt(h, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(h, CURLOPT_USERAGENT, "o2-aod-native");
    curl_easy_setopt(h, CURLOPT_CONNECTTIMEOUT, 30L);
    // Give up on a transfer slower than 1 kB/s for a minute.
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_LIMIT, 1024L);
    curl_easy_setopt(h, CURLOPT_LOW_SPEED_TIME, 60L);
    if (auto* dir = std::getenv("X509_CERT_DIR")) {
      curl_easy_setopt(h, CURLOPT_CAPATH, dir);
    }
    long version = CURL_HTTP_VERSION_2TLS;
    if (options.httpVersion == Options::HttpVersion::Http1_1) {
      version = CURL_HTTP_VERSION_1_1;
    } else if (options.httpVersion == Options::HttpVersion::Http2) {
      version = CURL_HTTP_VERSION_2_0;
    }
    curl_easy_setopt(h, CURLOPT_HTTP_VERSION, version);
    return h;
  }
};

HttpByteSource::HttpByteSource(std::string url, Options const& options)
  : mUrl(std::move(url)), mName(mUrl.substr(0, mUrl.find('?'))), mOptions(options)
{
  globalInit();
  mShared = std::make_unique<Shared>();
  auto* h = mShared->handle(mUrl, mOptions);
  char error[CURL_ERROR_SIZE] = {};
  curl_easy_setopt(h, CURLOPT_ERRORBUFFER, error);
  curl_easy_setopt(h, CURLOPT_NOBODY, 1L);
  curl_easy_setopt(h, CURLOPT_WRITEFUNCTION, discard);
  auto code = curl_easy_perform(h);
  long status = 0;
  curl_off_t length = -1;
  curl_easy_getinfo(h, CURLINFO_RESPONSE_CODE, &status);
  curl_easy_getinfo(h, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length);
  curl_easy_cleanup(h);
  if (code != CURLE_OK) {
    throw IoError(mName, error[0] ? error : curl_easy_strerror(code));
  }
  if (status != 200) {
    throw IoError(mName, "HEAD returned HTTP " + std::to_string(status));
  }
  if (length < 0) {
    throw IoError(mName, "the server does not tell the size of the file");
  }
  mSize = length;
}

HttpByteSource::~HttpByteSource() = default;

void HttpByteSource::checkRange(uint64_t off, size_t len) const
{
  if (off > mSize || len > mSize - off) {
    throw CorruptData(mName, off, "read of " + std::to_string(len) + " bytes past the end of a " + std::to_string(mSize) + " byte file");
  }
}

void HttpByteSource::readInto(uint64_t off, size_t len, void* dst)
{
  ReadRequest request{off, len, dst};
  readvInto({&request, 1});
}

void HttpByteSource::readvInto(std::span<ReadRequest> requests)
{
  std::vector<Transfer> transfers;
  size_t total = 0;
  for (auto const& r : requests) {
    checkRange(r.offset, r.length);
    total += r.length;
    for (size_t done = 0; done < r.length; done += kMaxRequest) {
      transfers.push_back({r.offset + done, std::min(kMaxRequest, r.length - done), (uint8_t*)r.dst + done});
    }
  }
  if (transfers.empty()) {
    return;
  }

  auto* multi = curl_multi_init();
  curl_multi_setopt(multi, CURLMOPT_PIPELINING, (long)CURLPIPE_MULTIPLEX);
  curl_multi_setopt(multi, CURLMOPT_MAX_HOST_CONNECTIONS, (long)std::max(1, mOptions.httpConnections));
  std::vector<std::string> ranges;
  ranges.reserve(transfers.size());
  for (auto& t : transfers) {
    t.handle = mShared->handle(mUrl, mOptions);
    ranges.push_back(std::to_string(t.offset) + "-" + std::to_string(t.offset + t.length - 1));
    curl_easy_setopt(t.handle, CURLOPT_RANGE, ranges.back().c_str());
    curl_easy_setopt(t.handle, CURLOPT_WRITEFUNCTION, writeInto);
    curl_easy_setopt(t.handle, CURLOPT_WRITEDATA, &t);
    curl_easy_setopt(t.handle, CURLOPT_ERRORBUFFER, t.error);
    curl_easy_setopt(t.handle, CURLOPT_PRIVATE, &t);
    // Wait for a multiplexed connection rather than opening one per request.
    curl_easy_setopt(t.handle, CURLOPT_PIPEWAIT, 1L);
    curl_multi_add_handle(multi, t.handle);
  }

  std::string error;
  int running = 0;
  do {
    auto mc = curl_multi_perform(multi, &running);
    if (mc != CURLM_OK) {
      error = curl_multi_strerror(mc);
      break;
    }
    int queued = 0;
    while (auto* msg = curl_multi_info_read(multi, &queued)) {
      if (msg->msg != CURLMSG_DONE) {
        continue;
      }
      Transfer* t = nullptr;
      curl_easy_getinfo(msg->easy_handle, CURLINFO_PRIVATE, (char**)&t);
      long status = 0;
      curl_easy_getinfo(msg->easy_handle, CURLINFO_RESPONSE_CODE, &status);
      if (!error.empty()) {
        continue;
      }
      if (t->overflow) {
        error = "the server ignored the byte range (HTTP " + std::to_string(status) + ")";
      } else if (msg->data.result != CURLE_OK) {
        error = t->error[0] ? t->error : curl_easy_strerror(msg->data.result);
      } else if (status != 206 && !(status == 200 && t->offset == 0 && t->length == mSize)) {
        error = "range request returned HTTP " + std::to_string(status);
      } else if (t->received != t->length) {
        error = "short read: " + std::to_string(t->received) + " of " + std::to_string(t->length) + " bytes";
      }
    }
    if (running > 0) {
      curl_multi_poll(multi, nullptr, 0, 1000, nullptr);
    }
  } while (running > 0 && error.empty());

  for (auto& t : transfers) {
    curl_multi_remove_handle(multi, t.handle);
    curl_easy_cleanup(t.handle);
  }
  curl_multi_cleanup(multi);
  if (!error.empty()) {
    throw IoError(mName, error);
  }
  mReadCalls.fetch_add(transfers.size(), std::memory_order_relaxed);
  mBytesRead.fetch_add(total, std::memory_order_relaxed);
}

} // namespace o2::aod::native

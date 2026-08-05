/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/*
 * gen_uuid.c --- generate a DCE-compatible uuid
 *
 * Copyright (C) 1996, 1997, 1998, 1999 Theodore Ts'o.
 *
 * %Begin-Header%
 * This file may be redistributed under the terms of the GNU
 * Library General Public License.
 * %End-Header%
 */

/*
 * Force inclusion of SVID stuff since we need it if we're compiling in
 * gcc-wall wall mode
 */
#define _SVID_SOURCE
#define _DEFAULT_SOURCE

#include "FairRunIdGenerator.h"

#include <stdlib.h>      // for rand, srand
#include <string.h>      // for memcpy
#include <sys/errno.h>   // for errno, EAGAIN, EINTR
#include <sys/fcntl.h>   // for open, O_RDONLY, O_NONBLOCK
#include <sys/time.h>    // for gettimeofday
#include <unistd.h>      // for getpid, getuid, read

#define srand(x) srandom(x)
#define rand() random()

// using namespace std;

//#include "genid32.h"    replaced by hrunidgenerator.h

/*
 * uuid.h -- private header file for uuids
 *
 * Copyright (C) 1996, 1997 Theodore Ts'o.
 *
 * %Begin-Header%
 * This file may be redistributed under the terms of the GNU
 * Library General Public License.
 * %End-Header%
 */

/*
 * Offset between 15-Oct-1582 and 1-Jan-70
 */
#define TIME_OFFSET_HIGH 0x01B21DD2
#define TIME_OFFSET_LOW 0x13814000

int FairRunIdGenerator::get_random_fd(void)
{
    struct timeval tv;
    static int fd = -2;
    int i;

    if (fd == -2) {
        gettimeofday(&tv, 0);
        fd = open("/dev/urandom", O_RDONLY);
        if (fd == -1) {
            fd = open("/dev/random", O_RDONLY | O_NONBLOCK);
        }
        srand((getpid() << 16) ^ getuid() ^ tv.tv_sec ^ tv.tv_usec);
    }
    /* Crank the random number generator a few times */
    gettimeofday(&tv, 0);
    for (i = (tv.tv_sec ^ tv.tv_usec) & 0x1F; i > 0; i--) {
        rand();
    }
    return fd;
}

/*
 * Generate a series of random bytes.  Use /dev/urandom if possible,
 * and if not, use srandom/random.
 */
void FairRunIdGenerator::get_random_bytes(void* buf, int nbytes)
{
    int i, fd = get_random_fd();
    int lose_counter = 0;
    char* cp = static_cast<char*>(buf);

    if (fd >= 0) {
        while (nbytes > 0) {
            i = read(fd, cp, nbytes);
            if ((i < 0) && ((errno == EINTR) || (errno == EAGAIN))) {
                continue;
            }
            if (i <= 0) {
                if (lose_counter++ == 8) {
                    break;
                }
                continue;
            }
            nbytes -= i;
            cp += i;
            lose_counter = 0;
        }
    }
    /* XXX put something better here if no /dev/random! */
    for (i = 0; i < nbytes; i++) {
        *cp++ = rand() & 0xFF;
    }
    return;
}

/*
 * Get the ethernet hardware address, if we can find it...
 */
int FairRunIdGenerator::get_node_id(unsigned char* /*node_id*/)
{
#ifdef HAVE_NET_IF_H
    int sd;
    struct ifreq ifr, *ifrp;
    struct ifconf ifc;
    char buf[1024];
    int n, i;
    unsigned char* a;

    /*
     * BSD 4.4 defines the size of an ifreq to be
     * max(sizeof(ifreq), sizeof(ifreq.ifr_name)+ifreq.ifr_addr.sa_len
     * However, under earlier systems, sa_len isn't present, so the size is
     * just sizeof(struct ifreq)
     */
#ifdef HAVE_SA_LEN
#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
#define ifreq_size(i) max(sizeof(struct ifreq), sizeof((i).ifr_name) + (i).ifr_addr.sa_len)
#else
#define ifreq_size(i) sizeof(struct ifreq)
#endif /* HAVE_SA_LEN*/

    sd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    if (sd < 0) {
        return -1;
    }
    memset(buf, 0, sizeof(buf));
    ifc.ifc_len = sizeof(buf);
    ifc.ifc_buf = buf;
    if (ioctl(sd, SIOCGIFCONF, (char*)&ifc) < 0) {
        close(sd);
        return -1;
    }
    n = ifc.ifc_len;
    for (i = 0; i < n; i += ifreq_size(*ifr)) {
        ifrp = (struct ifreq*)((char*)ifc.ifc_buf + i);
        strncpy(ifr.ifr_name, ifrp->ifr_name, IFNAMSIZ);
#ifdef SIOCGIFHWADDR
        if (ioctl(sd, SIOCGIFHWADDR, &ifr) < 0) {
            continue;
        }
        a = (unsigned char*)&ifr.ifr_hwaddr.sa_data;
#else
#ifdef SIOCGENADDR
        if (ioctl(sd, SIOCGENADDR, &ifr) < 0) {
            continue;
        }
        a = (unsigned char*)ifr.ifr_enaddr;
#else
        /*
         * XXX we don't have a way of getting the hardware
         * address
         */
        close(sd);
        return 0;
#endif /* SIOCGENADDR */
#endif /* SIOCGIFHWADDR */
        if (!a[0] && !a[1] && !a[2] && !a[3] && !a[4] && !a[5]) {
            continue;
        }
        if (node_id) {
            memcpy(node_id, a, 6);
            close(sd);
            return 1;
        }
    }
    close(sd);
#endif
    return 0;
}

/* Assume that the gettimeofday() has microsecond granularity */
#define MAX_ADJUSTMENT 10

int FairRunIdGenerator::get_clock(uint32_t* clock_high, uint32_t* clock_low, uint16_t* ret_clock_seq)
{
    static int adjustment = 0;
    static struct timeval last = {0, 0};
    static uint16_t clock_seq;
    struct timeval tv;
    unsigned long long clock_reg;

try_again:
    gettimeofday(&tv, 0);
    if ((last.tv_sec == 0) && (last.tv_usec == 0)) {
        get_random_bytes(&clock_seq, sizeof(clock_seq));
        clock_seq &= 0x1FFF;
        last = tv;
        last.tv_sec--;
    }
    if ((tv.tv_sec < last.tv_sec) || ((tv.tv_sec == last.tv_sec) && (tv.tv_usec < last.tv_usec))) {
        clock_seq = (clock_seq + 1) & 0x1FFF;
        adjustment = 0;
        last = tv;
    } else if ((tv.tv_sec == last.tv_sec) && (tv.tv_usec == last.tv_usec)) {
        if (adjustment >= MAX_ADJUSTMENT) {
            goto try_again;
        }
        adjustment++;
    } else {
        adjustment = 0;
        last = tv;
    }

    clock_reg = tv.tv_usec * 10 + adjustment;
    clock_reg += (static_cast<unsigned long long>(tv.tv_sec)) * 10000000;
    clock_reg += ((static_cast<unsigned long long>(0x01B21DD2)) << 32) + 0x13814000;

    *clock_high = clock_reg >> 32;
    *clock_low = clock_reg;
    *ret_clock_seq = clock_seq;

    fTimeSpec.tv_sec = last.tv_sec;
    //  fTimeSpec.tv_nsec = last.tv_usec*1000.;
    fTimeSpec.tv_nsec = 0.;

    return 0;
}

void FairRunIdGenerator::uuid_generate_time(uuid_t out)
{
    static unsigned char node_id[6];
    static int has_init = 0;
    struct uuid uu;
    uint32_t clock_mid;

    if (!has_init) {
        if (get_node_id(node_id) <= 0) {
            get_random_bytes(node_id, 6);
            /*
             * Set multicast bit, to prevent conflicts
             * with IEEE 802 addresses obtained from
             * network cards
             */
            node_id[0] |= 0x80;
        }
        has_init = 1;
    }
    get_clock(&clock_mid, &uu.time_low, &uu.clock_seq);
    uu.clock_seq |= 0x8000;
    uu.time_mid = static_cast<uint16_t>(clock_mid);
    uu.time_hi_and_version = (clock_mid >> 16) | 0x1000;
    memcpy(uu.node, node_id, 6);
    uuid_pack(&uu, out);
}

void FairRunIdGenerator::uuid_generate_random(uuid_t out)
{
    uuid_t buf;
    struct uuid uu;

    get_random_bytes(buf, sizeof(buf));
    uuid_unpack(buf, &uu);

    uu.clock_seq = (uu.clock_seq & 0x3FFF) | 0x8000;
    uu.time_hi_and_version = (uu.time_hi_and_version & 0x0FFF) | 0x4000;
    uuid_pack(&uu, out);
}

/*
 * This is the generic front-end to uuid_generate_random and
 * uuid_generate_time.  It uses uuid_generate_random only if
 * /dev/urandom is available, since otherwise we won't have
 * high-quality randomness.
 */
void FairRunIdGenerator::uuid_generate(uuid_t out)
{
    /*
  if (get_random_fd() >= 0) {
    uuid_generate_random(out);
  } else {
    uuid_generate_time(out);
  }
  */
    uuid_generate_time(out);
}

/*
 * Internal routine for packing UUID's
 *
 * Copyright (C) 1996, 1997 Theodore Ts'o.
 *
 * %Begin-Header%
 * This file may be redistributed under the terms of the GNU
 * Library General Public License.
 * %End-Header%
 */

void FairRunIdGenerator::uuid_pack(const struct uuid* uu, uuid_t ptr)
{
    uint32_t tmp;
    unsigned char* out = ptr;

    tmp = uu->time_low;
    out[3] = static_cast<unsigned char>(tmp);
    tmp >>= 8;
    out[2] = static_cast<unsigned char>(tmp);
    tmp >>= 8;
    out[1] = static_cast<unsigned char>(tmp);
    tmp >>= 8;
    out[0] = static_cast<unsigned char>(tmp);

    tmp = uu->time_mid;
    out[5] = static_cast<unsigned char>(tmp);
    tmp >>= 8;
    out[4] = static_cast<unsigned char>(tmp);

    tmp = uu->time_hi_and_version;
    out[7] = static_cast<unsigned char>(tmp);
    tmp >>= 8;
    out[6] = static_cast<unsigned char>(tmp);

    tmp = uu->clock_seq;
    out[9] = static_cast<unsigned char>(tmp);
    tmp >>= 8;
    out[8] = static_cast<unsigned char>(tmp);

    memcpy(out + 10, uu->node, 6);
}

/*
 * Internal routine for unpacking UUID
 *
 * Copyright (C) 1996, 1997 Theodore Ts'o.
 *
 * %Begin-Header%
 * This file may be redistributed under the terms of the GNU
 * Library General Public License.
 * %End-Header%
 */

void FairRunIdGenerator::uuid_unpack(const uuid_t in, struct uuid* uu)
{
    const uint8_t* ptr = in;
    uint32_t tmp;

    tmp = *ptr++;
    tmp = (tmp << 8) | *ptr++;
    tmp = (tmp << 8) | *ptr++;
    tmp = (tmp << 8) | *ptr++;
    uu->time_low = tmp;

    tmp = *ptr++;
    tmp = (tmp << 8) | *ptr++;
    uu->time_mid = tmp;

    tmp = *ptr++;
    tmp = (tmp << 8) | *ptr++;
    uu->time_hi_and_version = tmp;

    tmp = *ptr++;
    tmp = (tmp << 8) | *ptr++;
    uu->clock_seq = tmp;

    memcpy(uu->node, ptr, 6);
}

unsigned int FairRunIdGenerator::generateId(void)
{
    uuid_t uu;
    //  unsigned int v;
    //  struct timeval ret_tv;
    uuid_generate(uu);

    /*
  v = ((uu[0] ^ uu[4] ^ uu[8] ^ uu[12]) << 0)
      | ((uu[1] ^ uu[5] ^ uu[9] ^ uu[13]) << 8)
      | ((uu[2] ^ uu[6] ^ uu[10] ^ uu[14]) << 16)
      | ((uu[3] ^ uu[7] ^ uu[11] ^ uu[15]) << 24);
*/

    //  return v & 0x7fffffff;
    return getTID();
}

unsigned int FairRunIdGenerator::getTID() { return (fTimeSpec.tv_sec); }

struct timespec FairRunIdGenerator::getTimeSpecFromTID(unsigned int ms)
{
    struct timespec ret_tv;
    ret_tv.tv_sec = ms;
    ret_tv.tv_nsec = 0.;   // truncate the ns resolution
    return ret_tv;
}

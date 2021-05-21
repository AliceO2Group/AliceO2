/*
SHA-1 in C
By Steve Reid <steve@edmweb.com>
100% Public Domain
Test Vectors (from FIPS PUB 180-1)
"abc"
  A9993E36 4706816A BA3E2571 7850C26C 9CD0D89D
"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"
  84983E44 1C3BD26E BAAE4AA1 F95129E5 E54670F1
A million repetitions of "a"
  34AA973C D4C4DAA4 F61EEB2B DBAD2731 6534016F
*/

/* #define LITTLE_ENDIAN * This should be #define'd already, if true. */
/* #define SHA1HANDSOFF * Copies data before messing with it. */

#ifndef O2_FRAMEWORK_SHA1_H_
#define O2_FRAMEWORK_SHA1_H_

#define SHA1HANDSOFF

#include <cstdio>
#include <cstring>

/* for uint32_t */
#include <cstdint>

#define SHA_rol(value, bits) (((value) << (bits)) | ((value) >> (32 - (bits))))

/* blk0() and blk() perform the initial expand. */
/* I got the idea of expanding during the round function from SSLeay */
#if BYTE_ORDER == LITTLE_ENDIAN
#define SHA_blk0(i) (block->l[i] = (SHA_rol(block->l[i], 24) & 0xFF00FF00) | (SHA_rol(block->l[i], 8) & 0x00FF00FF))
#elif BYTE_ORDER == BIG_ENDIAN
#define SHA_blk0(i) block->l[i]
#else
#error "Endianness not defined!"
#endif
#define SHA_blk(i) (block->l[i & 15] = SHA_rol(block->l[(i + 13) & 15] ^ block->l[(i + 8) & 15] ^ block->l[(i + 2) & 15] ^ block->l[i & 15], 1))

/* (R0+R1), R2, R3, R4 are the different operations used in SHA1 */
#define SHA_R0(v, w, x, y, z, i)                                       \
  z += ((w & (x ^ y)) ^ y) + SHA_blk0(i) + 0x5A827999 + SHA_rol(v, 5); \
  w = SHA_rol(w, 30);
#define SHA_R1(v, w, x, y, z, i)                                      \
  z += ((w & (x ^ y)) ^ y) + SHA_blk(i) + 0x5A827999 + SHA_rol(v, 5); \
  w = SHA_rol(w, 30);
#define SHA_R2(v, w, x, y, z, i)                              \
  z += (w ^ x ^ y) + SHA_blk(i) + 0x6ED9EBA1 + SHA_rol(v, 5); \
  w = SHA_rol(w, 30);
#define SHA_R3(v, w, x, y, z, i)                                            \
  z += (((w | x) & y) | (w & x)) + SHA_blk(i) + 0x8F1BBCDC + SHA_rol(v, 5); \
  w = SHA_rol(w, 30);
#define SHA_R4(v, w, x, y, z, i)                              \
  z += (w ^ x ^ y) + SHA_blk(i) + 0xCA62C1D6 + SHA_rol(v, 5); \
  w = SHA_rol(w, 30);

namespace o2::framework::internal
{

typedef struct
{
  uint32_t state[5];
  uint32_t count[2];
  unsigned char buffer[64];
} SHA1_CTX;

/* Hash a single 512-bit block. This is the core of the algorithm. */
static void SHA1Transform(
  uint32_t state[5],
  const unsigned char buffer[64])
{
  uint32_t a, b, c, d, e;

  typedef union {
    unsigned char c[64];
    uint32_t l[16];
  } CHAR64LONG16;

#ifdef SHA1HANDSOFF
  CHAR64LONG16 block[1]; /* use array to appear as a pointer */

  memcpy(block, buffer, 64);
#else
  /* The following had better never be used because it causes the
     * pointer-to-const buffer to be cast into a pointer to non-const.
     * And the result is written through.  I threw a "const" in, hoping
     * this will cause a diagnostic.
     */
  CHAR64LONG16* block = (const CHAR64LONG16*)buffer;
#endif
  /* Copy context->state[] to working vars */
  a = state[0];
  b = state[1];
  c = state[2];
  d = state[3];
  e = state[4];
  /* 4 rounds of 20 operations each. Loop unrolled. */
  SHA_R0(a, b, c, d, e, 0);
  SHA_R0(e, a, b, c, d, 1);
  SHA_R0(d, e, a, b, c, 2);
  SHA_R0(c, d, e, a, b, 3);
  SHA_R0(b, c, d, e, a, 4);
  SHA_R0(a, b, c, d, e, 5);
  SHA_R0(e, a, b, c, d, 6);
  SHA_R0(d, e, a, b, c, 7);
  SHA_R0(c, d, e, a, b, 8);
  SHA_R0(b, c, d, e, a, 9);
  SHA_R0(a, b, c, d, e, 10);
  SHA_R0(e, a, b, c, d, 11);
  SHA_R0(d, e, a, b, c, 12);
  SHA_R0(c, d, e, a, b, 13);
  SHA_R0(b, c, d, e, a, 14);
  SHA_R0(a, b, c, d, e, 15);
  SHA_R1(e, a, b, c, d, 16);
  SHA_R1(d, e, a, b, c, 17);
  SHA_R1(c, d, e, a, b, 18);
  SHA_R1(b, c, d, e, a, 19);
  SHA_R2(a, b, c, d, e, 20);
  SHA_R2(e, a, b, c, d, 21);
  SHA_R2(d, e, a, b, c, 22);
  SHA_R2(c, d, e, a, b, 23);
  SHA_R2(b, c, d, e, a, 24);
  SHA_R2(a, b, c, d, e, 25);
  SHA_R2(e, a, b, c, d, 26);
  SHA_R2(d, e, a, b, c, 27);
  SHA_R2(c, d, e, a, b, 28);
  SHA_R2(b, c, d, e, a, 29);
  SHA_R2(a, b, c, d, e, 30);
  SHA_R2(e, a, b, c, d, 31);
  SHA_R2(d, e, a, b, c, 32);
  SHA_R2(c, d, e, a, b, 33);
  SHA_R2(b, c, d, e, a, 34);
  SHA_R2(a, b, c, d, e, 35);
  SHA_R2(e, a, b, c, d, 36);
  SHA_R2(d, e, a, b, c, 37);
  SHA_R2(c, d, e, a, b, 38);
  SHA_R2(b, c, d, e, a, 39);
  SHA_R3(a, b, c, d, e, 40);
  SHA_R3(e, a, b, c, d, 41);
  SHA_R3(d, e, a, b, c, 42);
  SHA_R3(c, d, e, a, b, 43);
  SHA_R3(b, c, d, e, a, 44);
  SHA_R3(a, b, c, d, e, 45);
  SHA_R3(e, a, b, c, d, 46);
  SHA_R3(d, e, a, b, c, 47);
  SHA_R3(c, d, e, a, b, 48);
  SHA_R3(b, c, d, e, a, 49);
  SHA_R3(a, b, c, d, e, 50);
  SHA_R3(e, a, b, c, d, 51);
  SHA_R3(d, e, a, b, c, 52);
  SHA_R3(c, d, e, a, b, 53);
  SHA_R3(b, c, d, e, a, 54);
  SHA_R3(a, b, c, d, e, 55);
  SHA_R3(e, a, b, c, d, 56);
  SHA_R3(d, e, a, b, c, 57);
  SHA_R3(c, d, e, a, b, 58);
  SHA_R3(b, c, d, e, a, 59);
  SHA_R4(a, b, c, d, e, 60);
  SHA_R4(e, a, b, c, d, 61);
  SHA_R4(d, e, a, b, c, 62);
  SHA_R4(c, d, e, a, b, 63);
  SHA_R4(b, c, d, e, a, 64);
  SHA_R4(a, b, c, d, e, 65);
  SHA_R4(e, a, b, c, d, 66);
  SHA_R4(d, e, a, b, c, 67);
  SHA_R4(c, d, e, a, b, 68);
  SHA_R4(b, c, d, e, a, 69);
  SHA_R4(a, b, c, d, e, 70);
  SHA_R4(e, a, b, c, d, 71);
  SHA_R4(d, e, a, b, c, 72);
  SHA_R4(c, d, e, a, b, 73);
  SHA_R4(b, c, d, e, a, 74);
  SHA_R4(a, b, c, d, e, 75);
  SHA_R4(e, a, b, c, d, 76);
  SHA_R4(d, e, a, b, c, 77);
  SHA_R4(c, d, e, a, b, 78);
  SHA_R4(b, c, d, e, a, 79);
  /* Add the working vars back into context.state[] */
  state[0] += a;
  state[1] += b;
  state[2] += c;
  state[3] += d;
  state[4] += e;
  /* Wipe variables */
  a = b = c = d = e = 0;
#ifdef SHA1HANDSOFF
  memset(block, '\0', sizeof(block));
#endif
}

/* SHA1Init - Initialize new context */

static void SHA1Init(
  SHA1_CTX* context)
{
  /* SHA1 initialization constants */
  context->state[0] = 0x67452301;
  context->state[1] = 0xEFCDAB89;
  context->state[2] = 0x98BADCFE;
  context->state[3] = 0x10325476;
  context->state[4] = 0xC3D2E1F0;
  context->count[0] = context->count[1] = 0;
}

/* Run your data through this. */

static void SHA1Update(
  SHA1_CTX* context,
  const unsigned char* data,
  uint32_t len)
{
  uint32_t i;

  uint32_t j;

  j = context->count[0];
  if ((context->count[0] += len << 3) < j) {
    context->count[1]++;
  }
  context->count[1] += (len >> 29);
  j = (j >> 3) & 63;
  if ((j + len) > 63) {
    memcpy(&context->buffer[j], data, (i = 64 - j));
    SHA1Transform(context->state, context->buffer);
    for (; i + 63 < len; i += 64) {
      SHA1Transform(context->state, &data[i]);
    }
    j = 0;
  } else {
    i = 0;
  }
  memcpy(&context->buffer[j], &data[i], len - i);
}

/* Add padding and return the message digest. */

static void SHA1Final(
  unsigned char digest[20],
  SHA1_CTX* context)
{
  unsigned i;

  unsigned char finalcount[8];

  unsigned char c;

  for (i = 0; i < 8; i++) {
    finalcount[i] = (unsigned char)((context->count[(i >= 4 ? 0 : 1)] >> ((3 - (i & 3)) * 8)) & 255); /* Endian independent */
  }
  c = 0200;
  SHA1Update(context, &c, 1);
  while ((context->count[0] & 504) != 448) {
    c = 0000;
    SHA1Update(context, &c, 1);
  }
  SHA1Update(context, finalcount, 8); /* Should cause a SHA1Transform() */
  for (i = 0; i < 20; i++) {
    digest[i] = (unsigned char)((context->state[i >> 2] >> ((3 - (i & 3)) * 8)) & 255);
  }
  /* Wipe variables */
  memset(context, '\0', sizeof(*context));
  memset(&finalcount, '\0', sizeof(finalcount));
}

void SHA1(
  char* hash_out,
  const char* str,
  unsigned int len)
{
  SHA1_CTX ctx;
  unsigned int ii;

  SHA1Init(&ctx);
  for (ii = 0; ii < len; ii += 1) {
    SHA1Update(&ctx, (const unsigned char*)str + ii, 1);
  }
  SHA1Final((unsigned char*)hash_out, &ctx);
  hash_out[20] = '\0';
}

} // namespace o2::framework::internal

#undef SHA1HANDSOFF
#undef SHA_rol
#undef SHA_blk0
#undef SHA_blk
#undef SHA_R0
#undef SHA_R1
#undef SHA_R2
#undef SHA_R3
#undef SHA_R4

#endif // O2_FRAMEWORK_SHA1_H_

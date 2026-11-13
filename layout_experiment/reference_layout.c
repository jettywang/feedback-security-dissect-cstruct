/*
 * Independent C reference for the layout experiment.
 *
 * This is deliberately compiled with a real C toolchain (not host ctypes) and
 * prints offsets/sizes via standard <stddef.h> offsetof/sizeof, so it acts as
 * an ABI ground truth against the Python engine and dissect.cstruct models.
 *
 * Native build (matches the running target triple):
 *   cc -Wall -std=c11 layout_experiment/reference_layout.c -o ref_native
 * Cross builds (when the toolchain/sysroot is installed):
 *   cc -m32 ...            # ILP32 on a multilib compiler
 *   aarch64-linux-gnu-gcc  # LP64 cross compiler
 *
 * Simulated ILP32 without a 32-bit sysroot (pointer modelled as uint32_t,
 * which is byte-identical for layout purposes):
 *   cc -Wall -std=c11 -DSIMULATE_ILP32 reference_layout.c -o ref_ilp32
 *
 * PACKED=1 additionally emits the __attribute__((packed)) variant.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef PACKED
#define ATTR __attribute__((packed))
#else
#define ATTR
#endif

struct ATTR Inner {
    uint16_t id;
    uint16_t flags;
};

/* A simulated ILP32 pointer is a plain 4-byte integer: same width and natural
 * alignment as a real 32-bit pointer, so offsets/sizes are ABI-equivalent. */
#ifdef SIMULATE_ILP32
typedef uint32_t target_ptr_t;
#else
typedef void *target_ptr_t;
#endif

#ifdef PACKED
#define ATTR __attribute__((packed))
#else
#define ATTR
#endif

struct ATTR Rec {
    uint32_t magic;
    target_ptr_t next;
    union ATTR U {
        uint32_t type;
        struct ATTR Inner info;
        uint32_t raw;
    } u;
    uint16_t count;
};

struct Block {
    struct Rec item[2];
};

/* Compile-time ground truth, one block per (pointer width, packing). */
#if defined(SIMULATE_ILP32) && defined(PACKED)
_Static_assert(sizeof(struct Rec) == 14, "ILP32 packed Rec");
_Static_assert(sizeof(struct Block) == 28, "ILP32 packed Block");
_Static_assert(offsetof(struct Rec, next) == 4, "ILP32 packed next");
_Static_assert(offsetof(struct Rec, u) == 8, "ILP32 packed union");
_Static_assert(offsetof(struct Rec, count) == 12, "ILP32 packed count");
#elif defined(SIMULATE_ILP32)
_Static_assert(sizeof(struct Rec) == 16, "ILP32 natural Rec");
_Static_assert(sizeof(struct Block) == 32, "ILP32 natural Block");
_Static_assert(offsetof(struct Rec, next) == 4, "ILP32 natural next");
_Static_assert(offsetof(struct Rec, u) == 8, "ILP32 natural union");
_Static_assert(offsetof(struct Rec, count) == 12, "ILP32 natural count");
#elif defined(PACKED)
_Static_assert(sizeof(struct Rec) == 18, "LP64 packed Rec");
_Static_assert(sizeof(struct Block) == 36, "LP64 packed Block");
_Static_assert(offsetof(struct Rec, next) == 4, "LP64 packed next");
_Static_assert(offsetof(struct Rec, u) == 12, "LP64 packed union");
_Static_assert(offsetof(struct Rec, count) == 16, "LP64 packed count");
#else
_Static_assert(sizeof(struct Rec) == 24, "LP64 natural Rec");
_Static_assert(sizeof(struct Block) == 48, "LP64 natural Block");
_Static_assert(offsetof(struct Rec, next) == 8, "LP64 natural next");
_Static_assert(offsetof(struct Rec, u) == 16, "LP64 natural union");
_Static_assert(offsetof(struct Rec, count) == 20, "LP64 natural count");
#endif

int main(void) {
    printf("model pointer width = %zu bytes\n", sizeof(target_ptr_t));
    printf("sizeof(struct Rec) = %zu\n", sizeof(struct Rec));
    printf("sizeof(struct Block=Rec[2]) = %zu\n", sizeof(struct Block));
    printf("offsetof magic = %zu\n", offsetof(struct Rec, magic));
    printf("offsetof next  = %zu\n", offsetof(struct Rec, next));
    printf("offsetof u     = %zu (sizeof union=%zu, alignof union=%zu)\n",
           offsetof(struct Rec, u), sizeof(((struct Rec *)0)->u),
           _Alignof(union U));
    printf("offsetof u.type = %zu\n", offsetof(struct Rec, u.type));
    printf("offsetof u.info.id    = %zu\n", offsetof(struct Rec, u.info.id));
    printf("offsetof u.info.flags = %zu\n", offsetof(struct Rec, u.info.flags));
    printf("offsetof count = %zu\n", offsetof(struct Rec, count));
    printf("alignof Rec = %zu, alignof Inner = %zu\n",
           _Alignof(struct Rec), _Alignof(struct Inner));
    /* Nested array: second element offset must equal one Rec stride. */
    printf("offsetof Block.item[1]       = %zu\n", offsetof(struct Block, item[1]));
    printf("offsetof Block.item[1].count = %zu\n",
           offsetof(struct Block, item[1].count));
    return 0;
}

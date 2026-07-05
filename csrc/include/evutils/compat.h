/* evutils — compiler/portability shims.
 *
 * The parsers are written with GCC/Clang extensions (the GNU builtins, the C99
 * `restrict` keyword, `__attribute__`, and the `__x86_64__`/`__aarch64__` arch
 * macros). MSVC (the default compiler for Windows wheels) spells these
 * differently or lacks them; this header bridges the gap so the same sources
 * build on Linux, macOS and Windows.
 */
#ifndef EVUTILS_COMPAT_H
#define EVUTILS_COMPAT_H

#if defined(_MSC_VER)

  #include <intrin.h>

  /* C99 `restrict` (and GCC's `__restrict__` spelling) -> MSVC's __restrict. */
  #ifndef restrict
    #define restrict __restrict
  #endif
  #define __restrict__ __restrict

  /* Branch-prediction hints: no MSVC equivalent, so compile them out. */
  #define likely(x)   (x)
  #define unlikely(x) (x)

  /* GNU builtins used by the parsers. */
  #define __builtin_unreachable() __assume(0)
  #define __builtin_prefetch(...) ((void)0)
  #ifndef __attribute__
    #define __attribute__(x)          /* e.g. __attribute__((fallthrough)) */
  #endif

  static __forceinline int __builtin_ctz(unsigned int x) {
      unsigned long index;
      _BitScanForward(&index, x);
      return (int)index;
  }

  /* MSVC uses _M_X64 / _M_ARM64 rather than __x86_64__ / __aarch64__; map them
   * so the SIMD dispatch in evt3.c selects the right path. */
  #if defined(_M_X64) && !defined(__x86_64__)
    #define __x86_64__ 1
  #endif
  #if defined(_M_ARM64) && !defined(__aarch64__)
    #define __aarch64__ 1
  #endif

#else  /* GCC / Clang */

  #define likely(x)   __builtin_expect(!!(x), 1)
  #define unlikely(x) __builtin_expect(!!(x), 0)

#endif

#endif /* EVUTILS_COMPAT_H */

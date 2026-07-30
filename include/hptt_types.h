#pragma once

#include <complex>
#include <complex.h>

#define REGISTER_BITS 256 // AVX
#ifdef HPTT_ARCH_ARM
#undef REGISTER_BITS 
#define REGISTER_BITS 128 // ARM
#endif

// `restrict` is a C99 keyword with no C++ equivalent, so every compiler
// exposes it under its own spelling: GCC, Clang and ICC accept
// `__restrict__`, MSVC only accepts `__restrict`. clang-cl defines
// _MSC_VER but still understands the GNU spelling.
//
// This lives in hptt_types.h rather than macros.h because transpose.h --
// a public, installed header -- uses it on the A_/B_ data members.
// macros.h additionally defines the unprefixed `INLINE`, which must not
// leak into consumers' translation units.
#if defined(_MSC_VER) && !defined(__clang__)
#define HPTT_RESTRICT __restrict
#else
#define HPTT_RESTRICT __restrict__
#endif

namespace hptt {

/**
 * \brief Determines the duration of the auto-tuning process.
 *
 * * ESTIMATE: 0 seconds (i.e., no auto-tuning)
 * * MEASURE: 10 seconds
 * * PATIENT: 60 seconds
 * * CRAZY : 3600 seconds
 */
enum SelectionMethod { ESTIMATE, MEASURE, PATIENT, CRAZY };

using FloatComplex = std::complex<float>;
using DoubleComplex = std::complex<double>;

}


/* evutils — library-level entry points.
 *
 * Small surface that does not belong to any one format: a version string and
 * a debug probe used by the test suite to validate the Python <-> C SoA
 * buffer hand-off without depending on a real parser being linked in yet.
 */
#ifndef EVUTILS_H
#define EVUTILS_H

#include "evutils/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Returns a static, NUL-terminated version string. */
const char *evutils_version(void);

/* DEBUG ONLY. Fills `buf` with `buf->capacity` synthetic events:
 *   t[i] = t0 + i,  x[i] = i % 640,  y[i] = i % 480,  p[i] = i & 1
 * Sets buf->size and returns the number written. Lets the Python test
 * confirm that numpy-allocated columns are written in place by C.
 * Returns 0 if any column pointer is NULL. */
size_t evutils_debug_fill_soa(event_buffer_soa_t *buf, uint64_t t0);

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_H */

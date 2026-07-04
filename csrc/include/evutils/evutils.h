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

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_H */

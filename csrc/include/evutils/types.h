/* evutils — shared event/trigger/buffer types.
 *
 * This is the single source of truth for the C ABI that the Python ctypes
 * layer mirrors (see src/evutils/_native.py). If you change a struct here,
 * update the matching ctypes.Structure there.
 *
 * Layout notes (these matter for the Python bindings):
 *   - event32_t is 12 bytes: t@0 (4), x@4 (2), y@6 (2), p@8 (1), + 3 pad.
 *     The AoS numpy dtype must therefore use itemsize=12, not 9.
 *   - The SoA buffer uses uint64 timestamps; the AoS event_t uses uint32.
 *     (See the note in README about reconciling 32- vs 64-bit time.)
 */
#ifndef EVUTILS_TYPES_H
#define EVUTILS_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct event32_s {
    uint32_t t;
    uint16_t x;
    uint16_t y;
    uint8_t  p;
} event32_t;

typedef struct event64_s {
    uint64_t t;
    uint16_t x;
    uint16_t y;
    uint8_t  p;
} event64_t;

typedef event32_t event_t;

typedef struct trigger64_s {
    uint64_t t;
    uint8_t  id;
    uint8_t  p;
} trigger64_t;

typedef struct trigger32_s {
    uint32_t t;
    uint8_t  id;
    uint8_t  p;
} trigger32_t;

typedef trigger32_t trigger_t;

/* Struct-of-arrays event buffer (preferred for the numpy path: one dtype per
 * column, no struct padding to reconcile). */
typedef struct event_buffer_soa_s {
    uint64_t *t;
    uint16_t *x;
    uint16_t *y;
    uint8_t  *p;
    size_t capacity;
    size_t size;
} event_buffer_soa_t;

/* Array-of-structs event buffer. */
typedef struct event_buffer_s {
    event_t *events;
    size_t capacity;
    size_t size;
} event_buffer_t;

typedef struct trigger_buffer_s {
    trigger_t *triggers;
    size_t capacity;
    size_t size;
} trigger_buffer_t;

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_TYPES_H */

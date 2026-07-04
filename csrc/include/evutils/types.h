/* evutils — shared event/trigger/buffer types.
 *
 * This is the single source of truth for the C ABI that the Python ctypes
 * layer mirrors (see src/evutils/_native.py). If you change a struct here,
 * update the matching ctypes.Structure there.
 *
 */
#ifndef EVUTILS_TYPES_H
#define EVUTILS_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Timestamp types. Use 64-bit timestamps for all new code; 32-bit timestamps are only for legacy support.
typedef uint64_t timestamp64_t;
typedef uint32_t timestamp32_t;
typedef timestamp64_t timestamp_t;


typedef struct event32_s {
    uint32_t t;
    uint16_t x;
    uint16_t y;
    uint8_t  p;
} event32_t;

typedef struct event64_s {
    timestamp64_t t;
    uint16_t x;
    uint16_t y;
    uint8_t  p;
} event64_t;

typedef event32_t event_t;

typedef struct trigger64_s {
    timestamp64_t t;
    uint8_t  id;
    uint8_t  p;
} trigger64_t;

typedef struct trigger32_s {
    timestamp32_t t;
    uint8_t  id;
    uint8_t  p;
} trigger32_t;

typedef trigger32_t trigger_t;

/* Struct-of-arrays event buffer (preferred for the numpy path: one dtype per
 * column, no struct padding to reconcile). */
typedef struct event_buffer_soa_s {
    timestamp64_t *t;
    uint16_t *x;
    uint16_t *y;
    uint8_t  *p;
    size_t capacity;
    size_t size;
} event_buffer_soa_t;

typedef struct trigger_buffer_soa_s {
    timestamp64_t *t;
    uint8_t  *id;
    uint8_t  *p;
    size_t capacity;
    size_t size;
} trigger_buffer_soa_t;

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

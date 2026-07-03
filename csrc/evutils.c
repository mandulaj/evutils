#include "evutils/evutils.h"

#ifndef EVUTILS_VERSION
#define EVUTILS_VERSION "0.0.1"
#endif

const char *evutils_version(void) {
    return EVUTILS_VERSION;
}

size_t evutils_debug_fill_soa(event_buffer_soa_t *buf, uint64_t t0) {
    if (!buf || !buf->t || !buf->x || !buf->y || !buf->p) {
        return 0;
    }
    const size_t n = buf->capacity;
    for (size_t i = 0; i < n; ++i) {
        buf->t[i] = t0 + (uint64_t)i;
        buf->x[i] = (uint16_t)(i % 640u);
        buf->y[i] = (uint16_t)(i % 480u);
        buf->p[i] = (uint8_t)(i & 1u);
    }
    buf->size = n;
    return n;
}

#include "evutils/aer.h"

typedef struct aer_state_s {
    uint64_t t_next;   /* timestamp of the next decoded event (sequential) */
    uint64_t t_step;   /* timestamp increment per event (sequential)       */
    int32_t  mode;     /* AER_TS_ZERO / AER_TS_SEQUENTIAL                  */
} aer_state_t;

size_t AER_state_size(void) {
    return sizeof(aer_state_t);
}

void AER_state_configure(aer_state_t *state, int32_t mode,
                         uint64_t t_next, uint64_t t_step) {
    state->mode = mode;
    state->t_next = t_next;
    state->t_step = t_step;
}

EVUTILS_TARGET_CLONES
parser_result_t AER_parse_chunk_soa(
    aer_state_t              *state,
    const aer_input_buffer_t *input_buffer,
    event_buffer_soa_t       *event_buffer,
    trigger_buffer_soa_t     *trigger_buffer) {

    (void)trigger_buffer;  /* AER carries no triggers */

    const uint32_t *restrict current = input_buffer->begin;
    const uint32_t *restrict end = input_buffer->end;

    timestamp_t* restrict out_ts = event_buffer->t;
    uint16_t*    restrict out_x = event_buffer->x;
    uint16_t*    restrict out_y = event_buffer->y;
    uint8_t*     restrict out_p = event_buffer->p;

    size_t n = event_buffer->size;
    const size_t capacity = event_buffer->capacity;

    if (state->mode == AER_TS_SEQUENTIAL) {
        uint64_t t = state->t_next;
        const uint64_t step = state->t_step;
        while (current < end && n < capacity) {
            uint32_t w = *current++;
            out_ts[n] = t;
            t += step;
            out_x[n] = (uint16_t)((w >> 9) & 0x1FF);
            out_y[n] = (uint16_t)(w & 0x1FF);
            out_p[n] = (uint8_t)((w >> 18) & 0x1);
            n++;
        }
        state->t_next = t;
    } else {
        while (current < end && n < capacity) {
            uint32_t w = *current++;
            out_ts[n] = 0;                          /* AER has no timestamps */
            out_x[n] = (uint16_t)((w >> 9) & 0x1FF);
            out_y[n] = (uint16_t)(w & 0x1FF);
            out_p[n] = (uint8_t)((w >> 18) & 0x1);
            n++;
        }
    }

    event_buffer->size = n;

    return (parser_result_t){
        .current = (const void *)current,
        .status = EVUTILS_PARSE_OK
    };
}

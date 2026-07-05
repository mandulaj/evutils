#include "evutils/evt21.h"


enum EVT21_PacketType{
    EVT21_CD_OFF         = 0x0,
    EVT21_CD_ON          = 0x1,
    EVT21_EVT_TIME_HIGH  = 0x8,
    EVT21_EXT_TRIGGER    = 0xA,
    EVT21_OTHERS         = 0xE,
};



typedef struct evt21_state_s {
    uint64_t last_ts_high;   /* current TIME_HIGH << 6 (bits 6..33)            */
    uint64_t ts_high_high;   /* accumulated overflow of the 28-bit field (34+) */
} evt21_state_t;


size_t EVT21_state_size(void) {
    return sizeof(evt21_state_t);
}

/* The TIME_HIGH field is 28 bits carrying time base bits 6..33, so it wraps at
 * 2^34. Bump the overflow accumulator by 2^34 on each wrap. */
#define EVT21_TS_WRAP (1ULL << 34)


/* A single CD vector packet can emit up to 32 events (one per valid bit), so we
 * stop the main loop this far from capacity to guarantee the inner emit never
 * overruns the output buffer. */
#define EVT21_MAX_VECTOR_EVENTS 32


parser_result_t EVT21_parse_chunk_soa(
    evt21_state_t *state,
    const evt21_input_buffer_t *input_buffer,
    event_buffer_soa_t *event_buffer,
    trigger_buffer_soa_t *trigger_buffer)
{

    const uint64_t *restrict current = input_buffer->begin;
    const uint64_t *restrict end = input_buffer->end;

    // Event output buffers
    size_t n_events_read = event_buffer->size;
    const size_t events_capacity = event_buffer->capacity;
    const size_t events_capacity_offset =
        events_capacity > EVT21_MAX_VECTOR_EVENTS ? events_capacity - EVT21_MAX_VECTOR_EVENTS : 0;

    timestamp_t* restrict out_ts = event_buffer->t;
    uint16_t* restrict out_x = event_buffer->x;
    uint16_t* restrict out_y = event_buffer->y;
    uint8_t* restrict out_p = event_buffer->p;

    // Trigger output buffers
    timestamp_t* restrict trigger_ts = trigger_buffer->t;
    uint8_t* restrict trigger_id = trigger_buffer->id;
    uint8_t* restrict trigger_p = trigger_buffer->p;

    size_t n_triggers_read = trigger_buffer->size;
    const size_t triggers_capacity = trigger_buffer->capacity;

    // State variables
    uint64_t last_ts_high = state->last_ts_high;
    uint64_t ts_high_high = state->ts_high_high;

    parse_status_t status = EVUTILS_PARSE_OK;

    while(
        current < end &&
        n_events_read < events_capacity_offset &&
        n_triggers_read < triggers_capacity
    ) {
        const uint64_t word = *current;
        current++;

        // 31-28 type
        uint64_t type = (word & 0xF0000000) >> 28;

        // 27-22 ts (6 low bits of the time base)
        uint64_t ts = ts_high_high | last_ts_high | ((word & 0x0FC00000) >> 22);

        // 21-11 x (11 bits)
        uint64_t x = (word & 0x003FF800) >> 11;

        // 10-0 y (11 bits)
        uint64_t y = (word & 0x000007FF);

        // upper 32 bits: per-column validity mask
        uint32_t valid = (uint32_t)(word >> 32);

        if((type == EVT21_CD_OFF || type == EVT21_CD_ON)){
            uint32_t p = !!type;
            while(valid){
                int lz = __builtin_ctz(valid);
                valid &= valid - 1;

                out_ts[n_events_read] = ts;
                out_x[n_events_read] = (uint16_t)(x + lz);
                out_y[n_events_read] = (uint16_t)y;
                out_p[n_events_read] = (uint8_t)p;
                n_events_read++;
            }
            continue;
        }

        switch(type){
            case EVT21_EVT_TIME_HIGH:
                {
                    uint64_t new_ts_high = (word & 0x0FFFFFFF) << 6;
                    if (new_ts_high < last_ts_high) {
                        ts_high_high += EVT21_TS_WRAP;
                    }
                    last_ts_high = new_ts_high;
                }
                break;
            case EVT21_EXT_TRIGGER:
                trigger_ts[n_triggers_read] = ts;
                // Trigger id bits 44..40 (5 bits)
                trigger_id[n_triggers_read] = (word >> 40) & 0x1F;
                trigger_p[n_triggers_read] = (word >> 32) & 0x1;
                n_triggers_read++;
                break;
            case EVT21_OTHERS:
            default:
                break;
        }
    }

    event_buffer->size = n_events_read;
    trigger_buffer->size = n_triggers_read;
    state->last_ts_high = last_ts_high;
    state->ts_high_high = ts_high_high;

    return (parser_result_t){
        .current = (const void *)current,
        .status = status
    };
}

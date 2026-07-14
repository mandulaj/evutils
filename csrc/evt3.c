
#include "evutils/evt3.h"
#include "evutils/types.h"
#include "evutils/parser.h"

#include <stdio.h>
#include <stdint.h>

#define EVT3_get_packet_type(packet) (((packet) >> 12) & 0xF)
#define EVT3_get_packet_data(packet) ((packet) & 0x0FFF)

enum EVT3_PacketType {
    EVT3_EVT_ADDR_Y = 0x0,
    EVT3_EVT_ADDR_X = 0x2,
    EVT3_VECT_BASE_X = 0x3,
    EVT3_VECT_12 = 0x4,
    EVT3_VECT_8 = 0x5,
    EVT3_EVT_TIME_LOW = 0x6,
    EVT3_CONTINUED_4 = 0x7,
    EVT3_EVT_TIME_HIGH = 0x8,
    EVT3_EXT_TRIGGER = 0xA,
    EVT3_OTHERS = 0xE,
    EVT3_CONTINUED_12 = 0xF
};


typedef struct evt3_state_s {
    uint32_t ts_high_high;
    uint32_t ts_high;
    uint32_t ts_low;
    uint32_t ts;

    uint16_t y;
    uint16_t vecbase_x;
    uint8_t  vecbase_p;
} evt3_state_t;



size_t EVT3_state_size(void) {
    return sizeof(evt3_state_t);
}


#define EMIT_SOA() do { \
    unsigned has = (vec_valid != 0u); \
    uint32_t lz = (uint32_t)__builtin_ctz(vec_valid | 0x80000000u); \
    out_ts[n] = ts; \
    out_x[n] = (uint16_t)(bx + lz); \
    out_y[n] = y; \
    out_p[n] = p; \
    n += has; \
    vec_valid &= vec_valid - 1u; \
} while (0)


__attribute__((always_inline))
static inline const uint16_t * EVT3_parse_vector_12_12_8_soa(
    const uint16_t * __restrict__ current,
    evt3_state_t * __restrict__ state,
    timestamp_t* __restrict__ out_ts, uint16_t* __restrict__ out_x,
    uint16_t* __restrict__ out_y,  uint8_t* __restrict__ out_p,
    size_t * n_events) {

    uint32_t vec_valid = EVT3_get_packet_data(*current);
    current++;

    if(likely(EVT3_get_packet_type(*current) == EVT3_VECT_12)) {
        vec_valid |= (uint32_t)(EVT3_get_packet_data(*current) << 12);
        current++;

        if (likely(EVT3_get_packet_type(*current) == EVT3_VECT_8)) {
            vec_valid |= (uint32_t)(*current & 0x00FF) << 24;
            current++;
        }
    }
    const uint32_t ts = state->ts;
    const uint16_t y  = state->y;
    const uint8_t  p  = state->vecbase_p;
    const uint16_t bx = state->vecbase_x;
    size_t n = *n_events;

    EMIT_SOA(); EMIT_SOA(); EMIT_SOA(); EMIT_SOA();   /* 4 unconditional — no branch between them */

    while (vec_valid) { EMIT_SOA(); }     /* tail: only pop>4 (~23%) re-enters */

    *n_events = n;
    state->vecbase_x = bx + 32;

    return current;
}



// TODO: For future performance optimization, implement dynamic CPU dispatch
// (e.g., using __attribute__((target_clones("avx2", "default"))) on GCC/Clang)
// or manual cpuid + function pointers to enable SIMD decoding on supported
// hardware without breaking universal portability on older CPUs.


#define EVT3_INPUT_PADDING 4
EVUTILS_TARGET_CLONES
parser_result_t EVT3_parse_chunk_soa(
    evt3_state_t *state,
    const evt3_input_buffer_t *input_buffer,
    event_buffer_soa_t *event_buffer,
    trigger_buffer_soa_t *trigger_buffer) {

    // Hoist variables for better optimization
    const uint16_t *restrict current = input_buffer->begin;
    const uint16_t * end_offset = input_buffer->end - EVT3_INPUT_PADDING;

    // Event output buffers
    const size_t events_capacity = event_buffer->capacity;
    const size_t events_capacity_offset = events_capacity - 64; // We need to keep some space for vector messages
    size_t n_events_read = event_buffer->size;

    timestamp_t* restrict out_ts = event_buffer->t;
    uint16_t* restrict out_x = event_buffer->x;
    uint16_t* restrict out_y = event_buffer->y;
    uint8_t* restrict out_p = event_buffer->p;
    
    // Trigger output buffers
    const size_t triggers_capacity = trigger_buffer->capacity;
    size_t n_triggers_read = trigger_buffer->size;

    timestamp_t* restrict trigger_ts = trigger_buffer->t;
    uint8_t* restrict trigger_id = trigger_buffer->id;
    uint8_t* restrict trigger_p = trigger_buffer->p;
    
    // State variables
    evt3_state_t local_state = *state;

    parse_status_t status = EVUTILS_PARSE_OK;


    // Main parsing looop
    while(
        current < end_offset &&
        n_events_read < events_capacity_offset &&
        n_triggers_read < triggers_capacity) {

        uint32_t packet_type = EVT3_get_packet_type(*current);
        uint32_t packet_data = EVT3_get_packet_data(*current);

        // Most common case checks first
        if(packet_type == EVT3_EVT_ADDR_X) {
            out_ts[n_events_read] = local_state.ts;
            out_y[n_events_read] = local_state.y;
            out_x[n_events_read] = (packet_data & 0x7FF);
            out_p[n_events_read] = (packet_data & 0x800) >> 11;
            n_events_read++;
            current++;
            continue;
        }
        if(packet_type == EVT3_VECT_BASE_X) {
            // Implementation omitted for brevity
            local_state.vecbase_x  = packet_data & 0x7FF;
            local_state.vecbase_p = (packet_data & 0x800) >> 11;

            current++;

            // Parse vector messages
            while (EVT3_get_packet_type(*current) == EVT3_VECT_12 && current < end_offset && n_events_read < events_capacity_offset) {
                current = EVT3_parse_vector_12_12_8_soa(current, &local_state, out_ts, out_x, out_y, out_p, &n_events_read);
            }

            continue;
        }

        switch(packet_type) {
            case EVT3_EVT_ADDR_Y:
                local_state.y = packet_data & 0x7FF;
                break;
            case EVT3_EVT_TIME_HIGH:
                {
                    uint32_t new_ts_high = packet_data << 12;

                    if(unlikely(new_ts_high < local_state.ts_high)) {
                        local_state.ts_high_high += 0x1000000;
                    }
                    local_state.ts_high = new_ts_high;

                    current++;
                    packet_type = EVT3_get_packet_type(*current);

                    if(unlikely(packet_type != EVT3_EVT_TIME_LOW)) {
                        continue;
                    }
                    packet_data = EVT3_get_packet_data(*current);
                }
                __attribute__((fallthrough));
            case EVT3_EVT_TIME_LOW:
                local_state.ts_low = packet_data;
                local_state.ts = local_state.ts_high_high | local_state.ts_high | local_state.ts_low;
                break;
            case EVT3_EXT_TRIGGER:
                trigger_ts[n_triggers_read] = local_state.ts;
                trigger_id[n_triggers_read] = packet_data >> 8;
                trigger_p[n_triggers_read] = packet_data & 0x1;
                n_triggers_read++;
                break;
            case EVT3_VECT_12:
                current = EVT3_parse_vector_12_12_8_soa(current, &local_state, out_ts, out_x, out_y, out_p, &n_events_read);
                continue;
                break;
            case EVT3_VECT_8:
            case EVT3_EVT_ADDR_X:
            case EVT3_VECT_BASE_X:
                // printf("Unexpected packet type %d in vector parsing\n", packet_type);
                __builtin_unreachable();
                break;
            case EVT3_OTHERS:
            case EVT3_CONTINUED_12:
            case EVT3_CONTINUED_4:
            default:
                break;
        }
        current++;
    }

    *state = local_state;

    event_buffer->size = n_events_read;
    trigger_buffer->size = n_triggers_read;


    return (parser_result_t){
        .current = (const void *)current,
        .status = status
    };
}


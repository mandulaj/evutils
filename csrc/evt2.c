
#include "evutils/evt2.h"

typedef struct evt2_state_s {
    uint64_t last_ts_high;
} evt2_state_t;


enum EVT2_PacketType {
    EVT2_CD_OFF         = 0x0,
    EVT2_CD_ON          = 0x1,
    EVT2_EVT_TIME_HIGH  = 0x8,
    EVT2_EXT_TRIGGER    = 0xA,
    EVT2_OTHERS         = 0xE,
    EVT2_CONTINUED      = 0xF
};


parser_result_t EVT2_parse_chunk_soa(
    evt2_state_t *state,
    const evt2_input_buffer_t *input_buffer,
    event_buffer_soa_t *event_buffer,
    trigger_buffer_soa_t *trigger_buffer) {
        
    
    const uint32_t *restrict current = input_buffer->begin;
    const uint32_t *restrict end = input_buffer->end;
    
    // Event output buffers
    size_t n_events_read = event_buffer->size;
    const size_t events_capacity = event_buffer->capacity;

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
    
    parse_status_t status = EVUTILS_PARSE_OK;

    while(
        current < end &&
        n_events_read < events_capacity &&
        n_triggers_read < triggers_capacity
    ) {

        uint32_t packet_type = (*current & 0xF0000000) >> 28;
        uint32_t packet_data = *current & 0x0FFFFFFF;
        uint8_t value = 0, channel = 0;


        // printf("%01lx %08lx\n", packet_type, packet_data);
        switch(packet_type){
            case EVT2_CD_OFF:
            case EVT2_CD_ON:
                // Bits 27..22 are the least significant bits of the timestamp
                out_ts[n_events_read] = last_ts_high | ((packet_data >> 22) & 0x3F);
                
                // Bits 10..0 are the x coordinate
                out_x[n_events_read] = packet_data & 0x7FF;
                // Bits 21..11 are the y coordinate
                out_y[n_events_read] = (packet_data >> 11) & 0x7FF;

                out_p[n_events_read] = !!packet_type;
                n_events_read++;
                break;
            case EVT2_EVT_TIME_HIGH:
                // Bits 27..0 are the Most significant bits of the event time base (33..6)
                last_ts_high = packet_data << 6;
                break;
            case EVT2_EXT_TRIGGER:

                trigger_ts[n_triggers_read] = last_ts_high | ((packet_data >> 22) & 0x3F);
                trigger_id[n_triggers_read] = channel;
                trigger_p[n_triggers_read] = value;
                n_triggers_read++;
                break;
            case EVT2_OTHERS:
            case EVT2_CONTINUED:
            default:
                break;
        }
        current++;
    }
    event_buffer->size = n_events_read;
    trigger_buffer->size = n_triggers_read;
    state->last_ts_high = last_ts_high;

    return (parser_result_t){
        .current = (const void *)current,
        .status = status
    };
}



#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static inline int64_t parse_int(const char **cursor, char delimiter) {
    int64_t val = 0;
    int sign = 1;
    const char *c = *cursor;
    
    while (*c == ' ' || *c == '\t') c++;
    
    if (*c == '-') {
        sign = -1;
        c++;
    } else if (*c == '+') {
        c++;
    }
    
    while (*c >= '0' && *c <= '9') {
        val = val * 10 + (*c - '0');
        c++;
    }
    
    while (*c != delimiter && *c != '\n' && *c != '\r' && *c != '\0') c++;
    if (*c == delimiter) c++;
    
    *cursor = c;
    return val * sign;
}

int evutils_read_csv(
    const char *buffer, size_t buffer_len, 
    char delimiter,
    void **out_arrays, 
    int *array_types,
    int *col_mapping,
    int max_csv_cols,
    size_t max_events,
    size_t *bytes_consumed,
    size_t *events_parsed
) {
    const char *cursor = buffer;
    const char *end = buffer + buffer_len;
    size_t count = 0;
    
    while (cursor < end && count < max_events) {
        const char *line_end = cursor;
        while (line_end < end && *line_end != '\n') line_end++;
        
        if (line_end == end) { 
            break; 
        }
        
        if (*cursor == '\n' || *cursor == '\r') {
            cursor++;
            if (cursor < end && *cursor == '\n') cursor++;
            continue;
        }
        
        const char *line_cursor = cursor;
        int col_idx = 0;
        
        while (line_cursor < line_end && col_idx < max_csv_cols) {
            int target_idx = col_mapping[col_idx];
            if (target_idx == -1) {
                while (line_cursor < line_end && *line_cursor != delimiter) line_cursor++;
                if (line_cursor < line_end && *line_cursor == delimiter) line_cursor++;
            } else {
                int64_t val = parse_int(&line_cursor, delimiter);
                int type = array_types[target_idx];
                if (type == 1) {
                    ((uint8_t*)out_arrays[target_idx])[count] = (uint8_t)val;
                } else if (type == 2) {
                    ((uint16_t*)out_arrays[target_idx])[count] = (uint16_t)val;
                } else if (type == 8) {
                    ((int64_t*)out_arrays[target_idx])[count] = val;
                }
            }
            col_idx++;
        }
        
        cursor = line_end + 1;
        count++;
    }
    
    *bytes_consumed = cursor - buffer;
    *events_parsed = count;
    return 0;
}


static inline char* itoa_positive(uint64_t val, char *buf) {
    char *p = buf + 20;
    *p = '\0';
    if (val == 0) {
        *--p = '0';
        return p;
    }
    while (val > 0) {
        *--p = '0' + (val % 10);
        val /= 10;
    }
    return p;
}

static inline char* itoa_signed(int64_t val, char *buf) {
    if (val < 0) {
        char *p = itoa_positive((uint64_t)(-val), buf);
        *--p = '-';
        return p;
    }
    return itoa_positive((uint64_t)val, buf);
}

int evutils_write_csv(
    void **in_arrays,
    int *array_types,
    int num_columns,
    char delimiter,
    size_t num_events,
    char *out_buffer,
    size_t out_buffer_len,
    size_t *bytes_written,
    size_t *events_written
) {
    size_t buf_pos = 0;
    char num_buf[24];
    size_t i = 0;
    
    for (; i < num_events; i++) {
        size_t max_req_len = (size_t)(num_columns * 22);
        if (buf_pos + max_req_len > out_buffer_len) {
            break;
        }
        
        for (int c = 0; c < num_columns; c++) {
            int type = array_types[c];
            char *s;
            if (type == 1) {
                s = itoa_positive(((uint8_t*)in_arrays[c])[i], num_buf);
            } else if (type == 2) {
                s = itoa_positive(((uint16_t*)in_arrays[c])[i], num_buf);
            } else if (type == 8) {
                s = itoa_signed(((int64_t*)in_arrays[c])[i], num_buf);
            } else {
                s = "0";
            }
            
            size_t len = num_buf + 20 - s;
            memcpy(out_buffer + buf_pos, s, len);
            buf_pos += len;
            
            if (c == num_columns - 1) {
                out_buffer[buf_pos++] = '\n';
            } else {
                out_buffer[buf_pos++] = delimiter;
            }
        }
    }
    
    *bytes_written = buf_pos;
    *events_written = i;
    return 0;
}

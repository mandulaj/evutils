#ifndef EVUTILS_CSV_H
#define EVUTILS_CSV_H

#include <stddef.h>
#include "evutils/parser.h"

#ifdef __cplusplus
extern "C" {
#endif

parser_result_t evutils_read_csv(
    const char *buffer, size_t buffer_len,
    char delimiter,
    void **out_arrays,
    int *array_types,
    int *col_mapping,
    int max_csv_cols,
    size_t max_events,
    size_t *events_parsed
);

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
);

#ifdef __cplusplus
}
#endif

#endif /* EVUTILS_CSV_H */

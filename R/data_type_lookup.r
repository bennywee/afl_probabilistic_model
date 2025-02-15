source('R/raw_schema.r')

data_type_lookup <- list(
    ladder = list(fitzRoy::fetch_ladder, ladder_schema),
    results = list(fitzRoy::fetch_results, results_schema)
)

#!/usr/bin/env bash

# Where to search for graphs to benchmark
GRAPH_DIRECTORY="/home/z1ko/univr/tesi/cuhornet/benchmarks/graphs"

# Wher to output the benchmarks results
OUTPUT_DIRECTORY="/home/z1ko/univr/tesi/cuhornet/benchmarks"

# How many iterations for benchmark
BENCHMARK_SIZE=500

# What batch size will be tested
BATCH_SIZES=(100 1000 5000 10000 50000 1000000)

for file in "$GRAPH_DIRECTORY"/*; do
	if ! [ -f "$file" ]; then
		continue
	fi

	# Get filename without the path and the extension
	filename_ext=$(basename "$file")
	filename="${filename_ext%.*}"

	for batch_size in ${BATCH_SIZES[@]}; do

		# Output pattern: benchmark_[type]_[graph]_[batch].csv
		output="$OUTPUT_DIRECTORY/benchmark_insert_${filename}_${batch_size}.csv"

		# Remove old benchmarks
		if [ -f "$output" ]; then
			printf "found old benchmark for $output, removing ... "
			rm "$output"
			printf "done\n"
		fi

		printf "generating $output ... "
		./hornetsnest/build/dbfs_benchmark "$file" "$batch_size" "$BENCHMARK_SIZE" \
			1>/dev/null 2>"$output"
		printf "done\n"

	done
done

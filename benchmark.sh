#!/usr/bin/env bash

# Where to search for graphs to benchmark
GRAPH_DIRECTORY="/home/z1ko/univr/tesi/cuhornet/benchmarks/graphs"

# Wher to output the benchmarks results
OUTPUT_DIRECTORY="/home/z1ko/univr/tesi/cuhornet/benchmarks"

# How many iterations for benchmark
BENCHMARK_SIZE=100

for file in "$GRAPH_DIRECTORY"/*; do
	if ! [ -f "$file" ]; then
		continue
	fi

	# Get filename without the path and the extension
	filename_ext=$(basename "$file")
	filename="${filename_ext%.*}"

	for i in {3..6}; do
		# Bath size increments by powers of ten
		batch_size=$((10 ** $i))

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

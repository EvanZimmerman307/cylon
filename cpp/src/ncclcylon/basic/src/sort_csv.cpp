#include <cudf/table/table_view.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <memory>
#include <string>
#include <utility>
#include <vector>

cudf::io::table_with_metadata read_csv(std::string const& file_path)
{
  auto source_info = cudf::io::source_info(file_path);
  auto builder     = cudf::io::csv_reader_options::builder(source_info);
  auto options     = builder.build();
  return cudf::io::read_csv(options);
}

void write_csv(cudf::table_view const& tbl_view, std::string const& file_path)
{
  auto sink_info = cudf::io::sink_info(file_path);
  auto builder   = cudf::io::csv_writer_options::builder(sink_info, tbl_view);
  auto options   = builder.build();
  cudf::io::write_csv(options);
}

std::unique_ptr<cudf::table> genericsort(cudf::table_view randnumbers_table, int col)
{
  // Specify the column order: defaulting to ASCENDING for all columns
  std::vector<cudf::order> column_order(randnumbers_table.num_columns(), cudf::order::ASCENDING);
  //std::vector<cudf::size_type> sort_columns{col};
  
  // Perform the sort
  auto sorted_table = cudf::sort(randnumbers_table, column_order);
  
  // Return the sorted table
  return sorted_table;
}

std::unique_ptr<cudf::table> sort_by_column(cudf::table_view randnumbers_table, int col)
{
    // Ensure the column index is valid
    if (col < 0 || col >= randnumbers_table.num_columns()) {
        throw std::out_of_range("Column index out of range");
    }

    // Extract the key column to sort by
    cudf::table_view key_table({randnumbers_table.column(col)});

    // Specify the column order: defaulting to ASCENDING for the key column
    std::vector<cudf::order> column_order{cudf::order::ASCENDING};
    // Optionally, specify null ordering (e.g., nulls before or after non-null values)
    std::vector<cudf::null_order> null_precedence{cudf::null_order::BEFORE};

    // Perform the sort by key
    auto sorted_table = cudf::sort_by_key(randnumbers_table, key_table, column_order, null_precedence);

    // Return the sorted table
    return sorted_table;
}

int main(int argc, char** argv)
{
  // Construct a CUDA memory resource using RAPIDS Memory Manager (RMM)
  // This is the default memory resource for libcudf for allocating device memory.
  rmm::mr::cuda_memory_resource cuda_mr{};
  // Construct a memory pool using the CUDA memory resource
  // Using a memory pool for device memory allocations is important for good performance in libcudf.
  // The pool defaults to allocating half of the available GPU memory.
  rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(50)};

  // Set the pool resource to be used by default for all device memory allocations
  // Note: It is the user's responsibility to ensure the `mr` object stays alive for the duration of
  // it being set as the default
  // Also, call this before the first libcudf API call to ensure all data is allocated by the same
  // memory resource.
  rmm::mr::set_current_device_resource(&mr);

  // Read data
  auto randnumbers_with_metadata = read_csv("randnumbers.csv");

  // Process
  auto result = sort_by_column(randnumbers_with_metadata.tbl->view(), 2);

  // Write the sorted table to a new CSV file
  write_csv(result->view(), "sorted_randnumbers.csv");


  return 0;
}
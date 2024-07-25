#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/reduction.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/copying.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/filling.hpp>
#include <cudf/stream_compaction.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

void printDeviceMemInfo(int deviceId) 
{
  cudaSetDevice(deviceId);
  
  size_t free_byte;
  size_t total_byte;
  
  cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
  
  if (cudaSuccess != cuda_status) {
      std::cerr << "Error: cudaMemGetInfo fails, " << cudaGetErrorString(cuda_status) << std::endl;
      return;
  }
  
  double free_db = (double)free_byte;
  double total_db = (double)total_byte;
  double used_db = total_db - free_db;
  
  std::cout << "GPU " << deviceId << " memory usage: used = " 
            << used_db / 1024.0 / 1024.0 << " MB, free = " 
            << free_db / 1024.0 / 1024.0 << " MB, total = " 
            << total_db / 1024.0 / 1024.0 << " MB" << std::endl;
}

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

std::vector<int> calcbuckets(cudf::table_view const& tbl_view, int sortcolumn, int num_gpus) 
{
  cudf::column_view column_view = tbl_view.column(sortcolumn); // Access the column
  std::pair<std::unique_ptr<cudf::scalar>, std::unique_ptr<cudf::scalar>> minmax_result = cudf::minmax(column_view); // calculate minmax

  // Extract the min & max value, what if the column is not INT32?
  auto minval = static_cast<cudf::numeric_scalar<int32_t>*>(minmax_result.first.get())->value();
  auto maxval = static_cast<cudf::numeric_scalar<int32_t>*>(minmax_result.second.get())->value();

  std::vector<int> bucketboundaries;
  for (int i = 0; i <= num_gpus; i++) 
  {
    bucketboundaries.push_back(minval + ((maxval-minval) * i) / num_gpus);
  }

  return bucketboundaries;

}

std::vector<std::unique_ptr<cudf::table>> partition_table(cudf::table_view const& tbl_view, int sortcolumn, 
 const std::vector<int>& bucketboundaries, int num_gpus, std::vector<std::unique_ptr<rmm::mr::device_memory_resource>>& mr_vector)
{
  std::vector<std::unique_ptr<cudf::table>> gpu_partitions(num_gpus);

  for (int i = 0; i < num_gpus; ++i) 
  {
      cudaError_t err = cudaSetDevice(i);
      if (err != cudaSuccess) {
          std::cerr << "CUDA error setting device " << i << ": " << cudaGetErrorString(err) << std::endl;
          return gpu_partitions;
      }

      rmm::mr::set_current_device_resource(mr_vector[i].get());
      // Create a copy of the table on the current device
      auto device_table = cudf::copy_to_device(tbl_view);

      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
          std::cerr << "CUDA error synchronizing device " << i << ": " << cudaGetErrorString(err) << std::endl;
          return gpu_partitions;
      }

      printDeviceMemInfo(i);

      std::cout << "Processing partition " << i << " on GPU " << i << std::endl;

      try 
      {
        auto lower_bound = cudf::numeric_scalar<int32_t>(bucketboundaries[i]);
        auto upper_bound = cudf::numeric_scalar<int32_t>(bucketboundaries[i + 1]);

        std::cout << "Creating lower mask for partition " << i << " on GPU " << i << std::endl;
        auto lower_mask = cudf::make_column_from_scalar(lower_bound, device_table->view().num_rows());
        
        std::cout << "Creating upper mask for partition " << i << " on GPU " << i << std::endl;
        auto upper_mask = cudf::make_column_from_scalar(upper_bound, device_table->view().num_rows());

        std::cout << "Created masks for partition " << i << " on GPU " << i << std::endl;

        std::cout << "Creating lower filter mask for partition " << i << " on GPU " << i << std::endl;
        lower_mask = cudf::binary_operation(device_table->view().column(sortcolumn), lower_mask->view(), cudf::binary_operator::GREATER_EQUAL, cudf::data_type{cudf::type_id::BOOL8});
        
        std::cout << "Creating upper filter mask for partition " << i << " on GPU " << i << std::endl;
        upper_mask = cudf::binary_operation(device_table->view().column(sortcolumn), upper_mask->view(), cudf::binary_operator::LESS, cudf::data_type{cudf::type_id::BOOL8});
        
        std::cout << "Combining filter masks for partition " << i << " on GPU " << i << std::endl;
        lower_mask = cudf::binary_operation(lower_mask->view(), upper_mask->view(), cudf::binary_operator::LOGICAL_AND, cudf::data_type{cudf::type_id::BOOL8});

        upper_mask.reset();

        std::cout << "Created filter masks for partition " << i << " on GPU " << i << std::endl;

        std::cout << "Applying boolean mask for partition " << i << " on GPU " << i << std::endl;
        auto filtered_table = cudf::apply_boolean_mask(device_table->view(), lower_mask->view());
        lower_mask.reset();

        std::cout << "Filtered table created for partition " << i << " on GPU " << i << std::endl;

        gpu_partitions[i] = std::move(filtered_table);

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error after partitioning on GPU " << i << ": " << cudaGetErrorString(err) << std::endl;
            return gpu_partitions;
        }
        printDeviceMemInfo(i);
      } 
      catch (const std::exception& e) 
      {
        std::cerr << "Error processing partition " << i << " on GPU " << i << ": " << e.what() << std::endl;
        return gpu_partitions;
      }
  }
  return gpu_partitions;
}

int main(int argc, char* argv[])
{
  int num_gpus = 0;
  cudaGetDeviceCount(&num_gpus);
  std::cout << "Number of CUDA devices: " << num_gpus << std::endl;

  // Initialize RMM for each GPU
  std::vector<std::unique_ptr<rmm::mr::device_memory_resource>> mr_vector;
  for (int i = 0; i < num_gpus; ++i) {
    cudaSetDevice(i);
    mr_vector.push_back(std::make_unique<rmm::mr::cuda_memory_resource>());
    rmm::mr::set_current_device_resource(mr_vector.back().get());
  }

  cudaSetDevice(0);
  auto cudf_table = read_csv("smallerrandnumbers.csv");


  int sortcolumn = 2;
  auto buckets = calcbuckets(cudf_table.tbl->view(), sortcolumn, num_gpus); // buckets is a vector containing bucket boundaries
  auto gpu_partitions = partition_table(cudf_table.tbl->view(), sortcolumn, buckets, num_gpus, mr_vector);
  return 0;

}
#include <c10d/TCPStore.hpp>
#include <c10d/ProcessGroup.hpp>
#include <c10d/ProcessGroupNCCL.hpp>
// #include <c10d/ProcessGroupGloo.hpp>

class PGWrapper {
 public:
  PGWrapper() {
    grank = 0;
    gsize = 0;
  }

  int initialize() {
    if (gsize != 0) {
      return 0;
    }
    char *env;

    env = std::getenv("OMPI_COMM_WORLD_RANK");
    if (!env) {
      std::cerr<<"comm rank is not set\n";
      return 1;
    }
    grank = std::atoi(env);

    env = std::getenv("OMPI_COMM_WORLD_SIZE");
    if (!env) {
      std::cerr<<"comm size is not set\n";
      return 1;
    }
    gsize = std::atoi(env);

    c10d::TCPStoreOptions store_opts;
    store_opts.isServer = (grank == 0) ? true : false;
    store_ = c10::make_intrusive<c10d::TCPStore>("localhost", store_opts);
    auto pg_opts = c10::make_intrusive<c10d::ProcessGroupNCCL::Options>();

    pg_ = std::unique_ptr<c10d::ProcessGroupNCCL>(
      new ::c10d::ProcessGroupNCCL(store_, grank, gsize, pg_opts));
    return 0;
  }

  c10d::ProcessGroup& getProcessGroup() {
    return *pg_;
  }

 public:
  int grank;
  int gsize;
  std::unique_ptr<::c10d::ProcessGroup> pg_;
  c10::intrusive_ptr<::c10d::TCPStore> store_;
};


int main() {
    PGWrapper pgWrapper;
    if (pgWrapper.initialize() != 0) {
        return 0;
    }

    c10::cuda::CUDAGuard guard(pgWrapper.grank);
    auto options = at::TensorOptions().dtype(at::kInt).device(at::kCUDA, pgWrapper.grank);
    std::vector<at::Tensor> input = {at::ones(10, options)};
    at::Tensor expected = at::full(10, pgWrapper.gsize, options);
    pgWrapper.pg_->allreduce(input);
    TORCH_CHECK(at::allclose(expected, input.at(0)));
    return 0;
}

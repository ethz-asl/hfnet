#include <chrono>
#include <vector>
#include <unordered_map>
#include <queue>

#include <Eigen/Core>
#include <Eigen/StdVector>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <nabo/nabo.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

namespace py = pybind11;

template <template <typename, typename> class Container, typename Type>
using Aligned = Container<Type, Eigen::aligned_allocator<Type>>;

constexpr int kLocalDescriptorSize = 256;
constexpr int kGlobalDescriptorSize = 1024;

struct Image {
  // Undistorted keypoints in the normalized image plane.
  Eigen::Matrix<float, 2, Eigen::Dynamic> normalized_keypoints;
  // 256dim for local descriptors.
  Eigen::Matrix<float, kLocalDescriptorSize, Eigen::Dynamic> local_descriptors;

  // -1 if no correspondence.
  Eigen::VectorXi point_indices;
};

struct Point3d {
  Eigen::Vector3f xyz;

  std::vector<int> observing_images;
};

class HLoc {
public:
  HLoc() {
    image_descriptors_.resize(kGlobalDescriptorSize, 0);
    ratio_test_value_ = 0.9;
  }

  // Adds an Image and returns its index.
  int addImage(Eigen::Ref<Eigen::Matrix<float, 1, kGlobalDescriptorSize, Eigen::RowMajor>> global_descriptor,
               Eigen::Ref<Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor>> normalized_keypoints,
               Eigen::Ref<Eigen::Matrix<float, kLocalDescriptorSize, Eigen::Dynamic, Eigen::RowMajor>> local_descriptors) {
    CHECK_EQ(normalized_keypoints.cols(), local_descriptors.cols());

    const int num_keypoints = normalized_keypoints.cols();
    const int new_image_index = images_.size();

    LOG_IF(WARNING, num_keypoints == 0) << "Adding frame " << new_image_index
                                        << " with no keypoints.";

    Image new_image;
    new_image.normalized_keypoints = normalized_keypoints;
    new_image.local_descriptors = local_descriptors;
    new_image.point_indices =  Eigen::VectorXi::Constant(num_keypoints, -1);
    images_.push_back(new_image);

    CHECK_EQ(image_descriptors_.cols(), new_image_index);
    image_descriptors_.conservativeResize(Eigen::NoChange,
                                          image_descriptors_.cols() + 1);
    image_descriptors_.col(new_image_index) = global_descriptor;

    return new_image_index;
  }

  // Adds a 3D points and returns its index.
  int add3dPoint(Eigen::Ref<Eigen::Matrix<float, 1, 3, Eigen::RowMajor>> xyz,
                 Eigen::Ref<Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>> observing_images_and_kpts) {
    const int new_point_index = points_.size();
    Point3d new_point;
    new_point.xyz = xyz;
    for (int i = 0; i < observing_images_and_kpts.rows(); ++i) {
      const int observing_image_idx = observing_images_and_kpts(i, 0);
      CHECK_LT(observing_image_idx, images_.size());
      CHECK_GE(observing_image_idx, 0);
      new_point.observing_images.push_back(observing_image_idx);

      const int observing_kpt_idx = observing_images_and_kpts(i, 1);
      CHECK_GE(observing_kpt_idx, 0);
      CHECK_LT(observing_kpt_idx, images_[observing_image_idx].point_indices.size());
      images_[observing_image_idx].point_indices(observing_kpt_idx) = new_point_index;
    }
    points_.push_back(new_point);

    return new_point_index;
  }

  py::tuple localize(Eigen::Ref<Eigen::Matrix<float, 1, kGlobalDescriptorSize, Eigen::RowMajor>> global_descriptor,
               Eigen::Ref<Eigen::Matrix<float, 2, Eigen::Dynamic, Eigen::RowMajor>> row_normalized_keypoints,
               Eigen::Ref<Eigen::Matrix<float, kLocalDescriptorSize, Eigen::Dynamic, Eigen::RowMajor>> row_local_descriptors) {
    CHECK_EQ(row_normalized_keypoints.cols(), row_local_descriptors.cols());

    // Global retrieval first.
    constexpr int kNumNeighbors = 10;
    Eigen::VectorXi indices(kNumNeighbors);
    Eigen::VectorXf dists2(kNumNeighbors);

    auto global_start = std::chrono::high_resolution_clock::now();
    CHECK_EQ(global_descriptor.rows(), 1);
    CHECK_EQ(global_descriptor.cols(), kGlobalDescriptorSize);
    nns_->knn(global_descriptor, indices, dists2, kNumNeighbors, 0, Nabo::NNSearchF::SORT_RESULTS | Nabo::NNSearchF::ALLOW_SELF_MATCH);

    auto covis_start = std::chrono::high_resolution_clock::now();

    std::vector<std::vector<int>> components = covisibilityClustering(indices);

    auto ransac_start = std::chrono::high_resolution_clock::now();

    // Copy data as row-major sucks for our arrays.
    Eigen::Matrix<float, 2, Eigen::Dynamic> normalized_keypoints = row_normalized_keypoints;
    Eigen::Matrix<float, kLocalDescriptorSize, Eigen::Dynamic> local_descriptors = row_local_descriptors;

    int num_components_tested = 0;
    bool pnp_success = false;
    int num_inliers = 0;
    int num_iters = 0;

    int total_local_ms = 0;
    int total_pnp_ms = 0;

    int num_db_landmarks = 0;
    int num_matches = 0;

    int last_component_size = -1;

    for (std::vector<int>& component : components) {
      // Limit component size to 5.
      if (component.size() > 5) {
        component.resize(5);
      }

      ++num_components_tested;
      last_component_size = component.size();

      int time_local;
      int time_pnp;
      pnp_success = localizeLocally(component, normalized_keypoints, local_descriptors,
                          &num_db_landmarks, &num_matches, &num_inliers, &num_iters, &time_local, &time_pnp);
      total_local_ms += time_local;
      total_pnp_ms += time_pnp;

      // Break the loop if we succeed.
      if (pnp_success) {
        break;
      }
    }

    auto ransac_end = std::chrono::high_resolution_clock::now();
    auto dur_ransac_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ransac_end - ransac_start);
    auto dur_covis_ms = std::chrono::duration_cast<std::chrono::milliseconds>(ransac_start - covis_start);
    auto dur_global_ms = std::chrono::duration_cast<std::chrono::milliseconds>(covis_start - global_start);
    LOG(INFO) << dur_global_ms.count() << " " << dur_covis_ms.count() << " "
              << dur_ransac_ms.count() << " 2d3d,pnp(" << total_local_ms
              << ", " << total_pnp_ms << ") global/covis/local [ms]";

    return py::make_tuple(pnp_success, components.size(), num_components_tested, last_component_size,
                          num_db_landmarks,
                          num_matches, num_inliers,
                          num_iters, dur_global_ms.count(),
                          dur_covis_ms.count(), total_local_ms, total_pnp_ms);
  }

  void buildIndex() {
    LOG(INFO) << "Found " << images_.size() << " images and "
              << points_.size() << " 3D points. Building index.";
    CHECK_EQ(images_.size(), image_descriptors_.cols());

    CHECK_EQ(image_descriptors_.rows(), kGlobalDescriptorSize);
    CHECK_GT(image_descriptors_.cols(), 0);
    nns_ = Nabo::NNSearchF::createKDTreeLinearHeap(image_descriptors_);
  }

private:
  std::unordered_set<int> getConnectedImages(const int image_idx) const {
    CHECK_LT(image_idx, images_.size());
    CHECK_GE(image_idx, 0);

    const Eigen::VectorXi& point_indices = images_[image_idx].point_indices;
    std::unordered_set<int> connected_images;
    for (int i = 0; i < point_indices.size(); ++i) {
      const int connected_point_idx = point_indices(i);
      CHECK_GT(connected_point_idx, -1);
      CHECK_LT(connected_point_idx, points_.size());
      CHECK_GE(connected_point_idx, 0);
      connected_images.insert(points_[connected_point_idx].observing_images.begin(),
                              points_[connected_point_idx].observing_images.end());
    }
    return connected_images;
  }

  bool doPnpRansac(const opengv::points_t& points, const opengv::bearingVectors_t& bearing_vectors, int* num_inliers, int* num_iters) const {
    CHECK_NOTNULL(num_inliers);
    CHECK_NOTNULL(num_iters);

    CHECK_EQ(points.size(), bearing_vectors.size());

    const double focal_length = 800;
    const double pixel_sigma_px = 15;
    const double ransac_threshold = 1.0 - cos(atan(pixel_sigma_px / focal_length));
    const int max_ransac_iters = 1000;

    opengv::absolute_pose::CentralAbsoluteAdapter adapter(bearing_vectors,
      points);

    opengv::sac::Ransac<
      opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;

      std::shared_ptr<opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem>absposeproblem_ptr(
        new opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem(
          adapter, opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP));

    ransac.sac_model_ = absposeproblem_ptr;
    ransac.threshold_ = ransac_threshold;
    ransac.max_iterations_ = max_ransac_iters;
    const bool ransac_success = ransac.computeModel() && ransac.inliers_.size() > 11;

    *num_inliers = ransac.inliers_.size();
    *num_iters = ransac.iterations_;

    LOG(INFO) << "Ransac " << ransac_success << ": " << *num_inliers << " inliers, " << *num_iters << " it, " << points.size() << " points.";

    return ransac_success;
  }

  bool localizeLocally(const std::vector<int>& frame_component,
                       const Eigen::Matrix<float, 2, Eigen::Dynamic>& normalized_keypoints,
                       const Eigen::Matrix<float, kLocalDescriptorSize, Eigen::Dynamic>& local_descriptors,
                       int* num_db_landmarks, int* num_matches, int* num_inliers, int* num_iters, int* time_local_ms, int* time_ransac_ms) const {
    CHECK_NOTNULL(num_db_landmarks);
    CHECK_NOTNULL(num_matches);
    CHECK_NOTNULL(time_local_ms);
    CHECK_NOTNULL(time_ransac_ms);

    CHECK(!frame_component.empty());

    auto local_prep_start = std::chrono::high_resolution_clock::now();

    // First check the total size of the db to preallocate memory.
    int total_num_db_points = 0;
    for (const int frame_idx : frame_component) {
      total_num_db_points += images_[frame_idx].point_indices.size();
    }

    if (total_num_db_points == 0) {
      // No db points, quit early.
      *num_db_landmarks = 0;
      *num_matches = 0;
      *num_inliers = 0;
      *num_iters = 0;
      auto local_bailout = std::chrono::high_resolution_clock::now();
      *time_local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(local_bailout - local_prep_start).count();
      *time_ransac_ms = 0;

      return false;
    }

    Eigen::MatrixXf db_local_descriptors(kLocalDescriptorSize, total_num_db_points);
    Eigen::VectorXi db_point_indices(total_num_db_points);

    // Stick descriptors and point indices together.
    int index = 0;
    for (const int frame_idx : frame_component) {
      const Image& image = images_[frame_idx];
      const int num_points = image.point_indices.size();

      // Copy point indices.
      db_point_indices.segment(index, num_points) = image.point_indices;
      // Copy descriptors.
      db_local_descriptors.block(0, index, kLocalDescriptorSize, num_points) = image.local_descriptors;

      index += num_points;
    }

    *num_db_landmarks = total_num_db_points;

    constexpr int kLocalNumNeighbors = 2;
    Eigen::MatrixXi indices(kLocalNumNeighbors, local_descriptors.cols());
    Eigen::MatrixXf dists2(kLocalNumNeighbors, local_descriptors.cols());

    auto local_build = std::chrono::high_resolution_clock::now();

    CHECK_EQ(db_local_descriptors.rows(), kLocalDescriptorSize);
    CHECK_GT(db_local_descriptors.cols(), 0);
    Nabo::NNSearchF* local_nns = Nabo::NNSearchF::createKDTreeTreeHeap(db_local_descriptors);

    auto local_search = std::chrono::high_resolution_clock::now();

    CHECK_EQ(local_descriptors.rows(), kLocalDescriptorSize);
    CHECK_GT(local_descriptors.cols(), 0);
    local_nns->knn(local_descriptors, indices, dists2, kLocalNumNeighbors, 0, Nabo::NNSearchF::SORT_RESULTS | Nabo::NNSearchF::ALLOW_SELF_MATCH);

    auto filtering_start = std::chrono::high_resolution_clock::now();

    delete local_nns;

    const Eigen::MatrixXf dists = dists2.array().sqrt();

    const double kRatioTestValue = 0.9;

    // Assuming all matches will fit, will shrink to size below the loop.
    opengv::points_t points(indices.cols());
    opengv::bearingVectors_t bearing_vectors(indices.cols());
    int idx = 0;
    for (int i = 0; i < indices.cols(); ++i) {
      const int point_idx_0 = db_point_indices(indices(0, i));
      const int point_idx_1 = db_point_indices(indices(1, i));
      if (point_idx_0 == point_idx_1 || dists(0, i) < kRatioTestValue * dists(1, i)) {
        bearing_vectors[idx] = Eigen::Vector3f(
          normalized_keypoints(0, i), normalized_keypoints(1, i), 1).cast<double>();
        bearing_vectors[idx].normalize();

        CHECK_LT(db_point_indices(indices(0, i)), points_.size());
        points[idx] = points_[db_point_indices(indices(0, i))].xyz.cast<double>();
        ++idx;
      }
    }
    points.resize(idx);
    bearing_vectors.resize(idx);

    *num_matches = idx;

    auto pnp_start = std::chrono::high_resolution_clock::now();

    if (points.size() < 12) {
      // Bail out early.
      *num_inliers = 0;
      *num_iters = 0;
      auto local_bailout = std::chrono::high_resolution_clock::now();
      *time_local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pnp_start - local_prep_start).count();
      *time_ransac_ms = 0;

      return false;
    }

    const bool pnp_success = doPnpRansac(points, bearing_vectors, num_inliers, num_iters);

    auto pnp_end = std::chrono::high_resolution_clock::now();

    LOG(INFO) << "Num db/query " << db_local_descriptors.cols() << " / " << local_descriptors.cols();

    *time_local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pnp_start - local_prep_start).count();
    *time_ransac_ms = std::chrono::duration_cast<std::chrono::milliseconds>(pnp_end - pnp_start).count();

    return pnp_success;
  }

  std::vector<std::vector<int>> covisibilityClustering(const Eigen::VectorXi& indices) const {
    std::unordered_set<int> visited;
    std::vector<std::vector<int>> components;

    std::unordered_set<int> frame_ids;
    for (int i = 0; i < indices.size(); ++i) {
      frame_ids.insert(indices(i));
    }

    for (int i = 0; i < indices.size(); ++i) {
      if (visited.count(indices(i)) > 0u) {
        continue;
      }

      // New component.
      components.resize(components.size() + 1);

      std::queue<int> queue;
      queue.push(indices(i));
      while (!queue.empty()) {
        const int exploration_frame = queue.front();
        queue.pop();

        // If already there, look at the next item. If not there, insert.
        if (!visited.insert(exploration_frame).second) {
          continue;
        }

        components.back().push_back(exploration_frame);
        std::unordered_set<int> connected_frames = getConnectedImages(exploration_frame);

        for (const int connected_frame : connected_frames) {
          if (frame_ids.count(connected_frame) > 0u && visited.count(connected_frame) == 0u) {
            queue.push(connected_frame);
          }
        }
      }
    }

    return components;
  }

  std::vector<Image> images_;
  Aligned<std::vector, Point3d> points_;
  Nabo::NNSearchF* nns_;
  // 1024dim for global descriptor. Eigen is column major by default.
  Eigen::MatrixXf image_descriptors_;
  double ratio_test_value_;
};

PYBIND11_MODULE(_hloc_cpp, m) {
    m.doc() = "pybind11 Hierarchical Localization cpp backend";

    py::class_<HLoc>(m, "HLoc")
    .def(py::init())
    .def("addImage", &HLoc::addImage, py::return_value_policy::copy)
    .def("add3dPoint", &HLoc::add3dPoint, py::return_value_policy::copy)
    .def("buildIndex", &HLoc::buildIndex)
    .def("localize", &HLoc::localize, py::return_value_policy::copy);
}

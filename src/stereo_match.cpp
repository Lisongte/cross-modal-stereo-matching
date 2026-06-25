#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

enum class MatchMode {
    RawBaseline,
    NoSgm,
    Sgm4,
    Sgm8,
};

struct Args {
    std::string config_path;
    std::string mode = "sgm4";
    std::string left = "output/rectified/left_rgb_rectified.png";
    std::string right = "output/rectified/right_nir_rectified.png";
    std::string rectification = "output/rectified/rectification.yml";
    std::string gt_depth = "output/rectified/left_depth_rectified.png";
    std::string out_dir = "output/matching_sgm";
    int max_disp = 64;
    int census_radius = 3;
    int block_radius = 20;
    int sgm_directions = 4;
    int p1 = 4;
    int p2 = 32;
};

struct Metrics {
    int valid_count = 0;
    double depth_mse = 0.0;
    double depth_rmse = 0.0;
    double disp_epe = 0.0;
    double d1_all = 0.0;
    double abs_rel = 0.0;
};

struct CostVolumeResult {
    cv::Mat disparity;
    std::vector<std::uint16_t> data_cost;
    int disp_count = 0;
};

static void print_usage(const char* argv0) {
    std::cerr
        << "Usage:\n"
        << "  " << argv0
        << " --config config/matching.yml\n"
        << "  " << argv0
        << " --left output/rectified/left_rgb_rectified.png"
        << " --right output/rectified/right_nir_rectified.png"
        << " --rectification output/rectified/rectification.yml"
        << " --gt-depth output/rectified/left_depth_rectified.png"
        << " --out output/matching"
        << " [--mode raw_baseline|no_sgm|sgm4|sgm8]"
        << " [--max-disp 128 --census-radius 3 --block-radius 4]"
        << " [--block-diameter 9] [--sgm-directions 0|4|8 --p1 4 --p2 32]\n";
}

static std::string lower_ascii(std::string s) {
    for (char& c : s) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

static MatchMode parse_mode_name(const std::string& mode) {
    const std::string m = lower_ascii(mode);
    if (m == "raw_baseline" || m == "pure_baseline" || m == "baseline" || m == "raw") {
        return MatchMode::RawBaseline;
    }
    if (m == "no_sgm" || m == "wta" || m == "census_wta") {
        return MatchMode::NoSgm;
    }
    if (m == "sgm4" || m == "sgm_4") {
        return MatchMode::Sgm4;
    }
    if (m == "sgm8" || m == "sgm_8") {
        return MatchMode::Sgm8;
    }
    throw std::runtime_error("unknown mode: " + mode);
}

static std::string mode_to_string(MatchMode mode) {
    switch (mode) {
        case MatchMode::RawBaseline:
            return "raw_baseline";
        case MatchMode::NoSgm:
            return "no_sgm";
        case MatchMode::Sgm4:
            return "sgm4";
        case MatchMode::Sgm8:
            return "sgm8";
    }
    return "unknown";
}

static std::string algorithm_label(MatchMode mode) {
    switch (mode) {
        case MatchMode::RawBaseline:
            return "Raw grayscale SAD + box aggregation + WTA";
        case MatchMode::NoSgm:
            return "Census + box aggregation + WTA";
        case MatchMode::Sgm4:
            return "Census + box aggregation + SGM 4-direction";
        case MatchMode::Sgm8:
            return "Census + box aggregation + SGM 8-direction";
    }
    return "unknown";
}

static int sgm_directions_for_mode(MatchMode mode) {
    if (mode == MatchMode::Sgm4) {
        return 4;
    }
    if (mode == MatchMode::Sgm8) {
        return 8;
    }
    return 0;
}

static void set_block_diameter(Args& args, int diameter) {
    if (diameter < 1 || diameter % 2 == 0) {
        throw std::runtime_error("--block-diameter must be a positive odd number");
    }
    args.block_radius = (diameter - 1) / 2;
}

static void read_optional_string(cv::FileStorage& fs, const std::string& name, std::string& value) {
    cv::FileNode node = fs[name];
    if (!node.empty()) {
        node >> value;
    }
}

static void read_optional_int(cv::FileStorage& fs, const std::string& name, int& value) {
    cv::FileNode node = fs[name];
    if (!node.empty()) {
        node >> value;
    }
}

static void load_config(Args& args, const std::string& path) {
    cv::FileStorage cfg(path, cv::FileStorage::READ);
    if (!cfg.isOpened()) {
        throw std::runtime_error("failed to open config: " + path);
    }

    read_optional_string(cfg, "mode", args.mode);
    read_optional_string(cfg, "left", args.left);
    read_optional_string(cfg, "right", args.right);
    read_optional_string(cfg, "rectification", args.rectification);
    read_optional_string(cfg, "gt_depth", args.gt_depth);
    read_optional_string(cfg, "out", args.out_dir);
    read_optional_int(cfg, "max_disp", args.max_disp);
    read_optional_int(cfg, "census_radius", args.census_radius);
    read_optional_int(cfg, "block_radius", args.block_radius);
    read_optional_int(cfg, "p1", args.p1);
    read_optional_int(cfg, "p2", args.p2);

    int block_diameter = 0;
    read_optional_int(cfg, "block_diameter", block_diameter);
    if (block_diameter > 0) {
        set_block_diameter(args, block_diameter);
    }
}

static std::string need_value(int argc, char** argv, int& i, const std::string& name) {
    if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + name);
    }
    return argv[++i];
}

static void apply_arg(Args& args, const std::string& key, const std::string& value) {
    if (key == "--mode") {
        args.mode = value;
    } else if (key == "--left") {
        args.left = value;
    } else if (key == "--right") {
        args.right = value;
    } else if (key == "--rectification") {
        args.rectification = value;
    } else if (key == "--gt-depth") {
        args.gt_depth = value;
    } else if (key == "--out") {
        args.out_dir = value;
    } else if (key == "--max-disp") {
        args.max_disp = std::stoi(value);
    } else if (key == "--census-radius") {
        args.census_radius = std::stoi(value);
    } else if (key == "--block-radius") {
        args.block_radius = std::stoi(value);
    } else if (key == "--block-diameter") {
        set_block_diameter(args, std::stoi(value));
    } else if (key == "--sgm-directions") {
        const int directions = std::stoi(value);
        if (directions == 0) {
            args.mode = "no_sgm";
        } else if (directions == 4) {
            args.mode = "sgm4";
        } else if (directions == 8) {
            args.mode = "sgm8";
        } else {
            throw std::runtime_error("--sgm-directions must be 0, 4, or 8");
        }
    } else if (key == "--p1") {
        args.p1 = std::stoi(value);
    } else if (key == "--p2") {
        args.p2 = std::stoi(value);
    } else {
        throw std::runtime_error("unknown argument: " + key);
    }
}

static Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (key == "--config") {
            args.config_path = need_value(argc, argv, i, key);
        } else if (!key.empty() && key[0] == '-') {
            (void)need_value(argc, argv, i, key);
        }
    }

    if (!args.config_path.empty()) {
        load_config(args, args.config_path);
    }

    for (int i = 1; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--help" || key == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        }
        std::string value = need_value(argc, argv, i, key);
        if (key == "--config") {
            continue;
        }
        apply_arg(args, key, value);
    }
    return args;
}

static cv::Mat read_required_matrix(cv::FileStorage& fs, const std::string& name) {
    cv::Mat m;
    fs[name] >> m;
    if (m.empty()) {
        throw std::runtime_error("missing matrix in rectification file: " + name);
    }
    m.convertTo(m, CV_64F);
    return m;
}

static cv::Mat to_gray8(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }

    if (gray.type() != CV_8U) {
        double min_v = 0.0;
        double max_v = 0.0;
        cv::minMaxLoc(gray, &min_v, &max_v);
        if (max_v > min_v) {
            gray.convertTo(gray, CV_8U, 255.0 / (max_v - min_v), -255.0 * min_v / (max_v - min_v));
        } else {
            gray = cv::Mat::zeros(gray.size(), CV_8U);
        }
    }
    return gray;
}

static cv::Mat preprocess_gray(const cv::Mat& img) {
    cv::Mat gray = to_gray8(img);

    cv::Mat equalized;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
    clahe->apply(gray, equalized);
    cv::GaussianBlur(equalized, equalized, cv::Size(3, 3), 0.0);
    return equalized;
}

static int census_bit_count(int radius) {
    const int side = 2 * radius + 1;
    return side * side - 1;
}

static std::vector<std::uint64_t> census_transform(const cv::Mat& gray, int radius) {
    if (gray.type() != CV_8U) {
        throw std::runtime_error("census_transform expects CV_8U input");
    }
    if (radius < 1 || census_bit_count(radius) > 64) {
        throw std::runtime_error("census radius must produce between 1 and 64 bits");
    }

    const int rows = gray.rows;
    const int cols = gray.cols;
    std::vector<std::uint64_t> census(static_cast<std::size_t>(rows) * cols, 0);

    for (int y = radius; y < rows - radius; ++y) {
        const uchar* center_row = gray.ptr<uchar>(y);
        for (int x = radius; x < cols - radius; ++x) {
            const uchar center = center_row[x];
            std::uint64_t desc = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                const uchar* row = gray.ptr<uchar>(y + dy);
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx == 0 && dy == 0) {
                        continue;
                    }
                    desc <<= 1;
                    if (row[x + dx] < center) {
                        desc |= 1ULL;
                    }
                }
            }
            census[static_cast<std::size_t>(y) * cols + x] = desc;
        }
    }
    return census;
}

static inline int hamming64(std::uint64_t a, std::uint64_t b) {
    return __builtin_popcountll(a ^ b);
}

static inline std::size_t cost_index(int y, int x, int d, int cols, int disp_count) {
    return (static_cast<std::size_t>(y) * cols + x) * disp_count + d;
}

static CostVolumeResult build_intensity_cost_volume_and_initial_disparity(const cv::Mat& left_gray,
                                                                          const cv::Mat& right_gray,
                                                                          cv::Size size,
                                                                          int max_disp,
                                                                          int block_radius) {
    if (left_gray.type() != CV_8U || right_gray.type() != CV_8U) {
        throw std::runtime_error("raw baseline expects CV_8U grayscale input");
    }

    const int rows = size.height;
    const int cols = size.width;
    const float invalid_cost = 255.0f;
    const int block_size = 2 * block_radius + 1;
    const int disp_count = max_disp + 1;

    CostVolumeResult result;
    result.disp_count = disp_count;
    result.data_cost.assign(static_cast<std::size_t>(rows) * cols * disp_count,
                            static_cast<std::uint16_t>(invalid_cost));

    cv::Mat best_cost(rows, cols, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat disparity(rows, cols, CV_32F, cv::Scalar(0.0f));
    cv::Mat cost(rows, cols, CV_32F);
    cv::Mat aggregated(rows, cols, CV_32F);

    for (int d = 0; d <= max_disp; ++d) {
        for (int y = 0; y < rows; ++y) {
            const uchar* left_row = left_gray.ptr<uchar>(y);
            const uchar* right_row = right_gray.ptr<uchar>(y);
            float* cost_row = cost.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const int xr = x - d;
                if (xr < 0 || xr >= cols) {
                    cost_row[x] = invalid_cost;
                    continue;
                }
                cost_row[x] = static_cast<float>(std::abs(static_cast<int>(left_row[x]) -
                                                          static_cast<int>(right_row[xr])));
            }
        }

        if (block_radius > 0) {
            cv::boxFilter(cost, aggregated, CV_32F, cv::Size(block_size, block_size),
                          cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
        } else {
            aggregated = cost;
        }

        for (int y = 0; y < rows; ++y) {
            const float* agg_row = aggregated.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                result.data_cost[cost_index(y, x, d, cols, disp_count)] =
                    static_cast<std::uint16_t>(std::clamp<int>(static_cast<int>(std::lround(agg_row[x])), 0, 65535));
            }
        }

        for (int y = 0; y < rows; ++y) {
            const float* agg_row = aggregated.ptr<float>(y);
            float* best_row = best_cost.ptr<float>(y);
            float* disp_row = disparity.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                if (agg_row[x] < best_row[x]) {
                    best_row[x] = agg_row[x];
                    disp_row[x] = static_cast<float>(d);
                }
            }
        }

        if (d % 16 == 0 || d == max_disp) {
            std::cout << "processed disparity " << d << "/" << max_disp << "\n";
        }
    }

    disparity(cv::Rect(0, 0, std::min(max_disp + 1, cols), rows)).setTo(0);
    disparity.rowRange(0, std::min(block_radius, rows)).setTo(0);
    disparity.rowRange(std::max(0, rows - block_radius), rows).setTo(0);
    result.disparity = disparity;
    return result;
}

static CostVolumeResult build_cost_volume_and_initial_disparity(
                                 const std::vector<std::uint64_t>& census_left,
                                 const std::vector<std::uint64_t>& census_right,
                                 cv::Size size,
                                 int max_disp,
                                 int census_radius,
                                 int block_radius) {
    const int rows = size.height;
    const int cols = size.width;
    const int bits = census_bit_count(census_radius);
    const float invalid_cost = static_cast<float>(bits);
    const int block_size = 2 * block_radius + 1;
    const int disp_count = max_disp + 1;

    CostVolumeResult result;
    result.disp_count = disp_count;
    result.data_cost.assign(static_cast<std::size_t>(rows) * cols * disp_count,
                            static_cast<std::uint16_t>(bits));

    cv::Mat best_cost(rows, cols, CV_32F, cv::Scalar(std::numeric_limits<float>::max()));
    cv::Mat disparity(rows, cols, CV_32F, cv::Scalar(0.0f));
    cv::Mat cost(rows, cols, CV_32F);
    cv::Mat aggregated(rows, cols, CV_32F);

    for (int d = 0; d <= max_disp; ++d) {
        for (int y = 0; y < rows; ++y) {
            float* cost_row = cost.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                const int xr = x - d;
                if (x < census_radius || x >= cols - census_radius ||
                    y < census_radius || y >= rows - census_radius ||
                    xr < census_radius || xr >= cols - census_radius) {
                    cost_row[x] = invalid_cost;
                    continue;
                }
                const std::size_t li = static_cast<std::size_t>(y) * cols + x;
                const std::size_t ri = static_cast<std::size_t>(y) * cols + xr;
                const int c = hamming64(census_left[li], census_right[ri]);
                cost_row[x] = static_cast<float>(c);
            }
        }

        if (block_radius > 0) {
            cv::boxFilter(cost, aggregated, CV_32F, cv::Size(block_size, block_size),
                          cv::Point(-1, -1), true, cv::BORDER_REPLICATE);
        } else {
            aggregated = cost;
        }

        for (int y = 0; y < rows; ++y) {
            const float* agg_row = aggregated.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                result.data_cost[cost_index(y, x, d, cols, disp_count)] =
                    static_cast<std::uint16_t>(std::clamp<int>(static_cast<int>(std::lround(agg_row[x])), 0, 65535));
            }
        }

        for (int y = 0; y < rows; ++y) {
            const float* agg_row = aggregated.ptr<float>(y);
            float* best_row = best_cost.ptr<float>(y);
            float* disp_row = disparity.ptr<float>(y);
            for (int x = 0; x < cols; ++x) {
                if (agg_row[x] < best_row[x]) {
                    best_row[x] = agg_row[x];
                    disp_row[x] = static_cast<float>(d);
                }
            }
        }

        if (d % 16 == 0 || d == max_disp) {
            std::cout << "processed disparity " << d << "/" << max_disp << "\n";
        }
    }

    disparity(cv::Rect(0, 0, std::min(max_disp + census_radius + 1, cols), rows)).setTo(0);
    disparity.rowRange(0, std::min(census_radius + block_radius, rows)).setTo(0);
    disparity.rowRange(std::max(0, rows - census_radius - block_radius), rows).setTo(0);
    result.disparity = disparity;
    return result;
}

static void aggregate_sgm_direction(const std::vector<std::uint16_t>& data_cost,
                                    std::vector<std::int32_t>& sum_cost,
                                    cv::Size size,
                                    int disp_count,
                                    int dx,
                                    int dy,
                                    int p1,
                                    int p2) {
    const int rows = size.height;
    const int cols = size.width;
    const std::size_t total = static_cast<std::size_t>(rows) * cols * disp_count;
    std::vector<std::int32_t> path_cost(total, 0);

    const int x_begin = dx > 0 ? 0 : cols - 1;
    const int x_end = dx > 0 ? cols : -1;
    const int x_step = dx > 0 ? 1 : -1;
    const int y_begin = dy > 0 ? 0 : rows - 1;
    const int y_end = dy > 0 ? rows : -1;
    const int y_step = dy > 0 ? 1 : -1;

    for (int y = y_begin; y != y_end; y += y_step) {
        for (int x = x_begin; x != x_end; x += x_step) {
            const int px = x - dx;
            const int py = y - dy;
            const std::size_t base = cost_index(y, x, 0, cols, disp_count);

            if (px < 0 || px >= cols || py < 0 || py >= rows) {
                for (int d = 0; d < disp_count; ++d) {
                    const std::int32_t v = data_cost[base + d];
                    path_cost[base + d] = v;
                    sum_cost[base + d] += v;
                }
                continue;
            }

            const std::size_t prev = cost_index(py, px, 0, cols, disp_count);
            std::int32_t prev_min = path_cost[prev];
            for (int d = 1; d < disp_count; ++d) {
                prev_min = std::min(prev_min, path_cost[prev + d]);
            }

            for (int d = 0; d < disp_count; ++d) {
                std::int32_t best = path_cost[prev + d];
                if (d > 0) {
                    best = std::min(best, path_cost[prev + d - 1] + p1);
                }
                if (d + 1 < disp_count) {
                    best = std::min(best, path_cost[prev + d + 1] + p1);
                }
                best = std::min(best, prev_min + p2);

                const std::int32_t v = static_cast<std::int32_t>(data_cost[base + d]) + best - prev_min;
                path_cost[base + d] = v;
                sum_cost[base + d] += v;
            }
        }
    }
}

static cv::Mat compute_disparity_sgm(const std::vector<std::uint16_t>& data_cost,
                                     cv::Size size,
                                     int disp_count,
                                     int max_disp,
                                     int census_radius,
                                     int block_radius,
                                     int directions,
                                     int p1,
                                     int p2) {
    const int rows = size.height;
    const int cols = size.width;
    const std::size_t total = static_cast<std::size_t>(rows) * cols * disp_count;
    std::vector<std::int32_t> sum_cost(total, 0);

    std::vector<cv::Point> dirs = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1}
    };
    if (directions >= 8) {
        dirs.push_back({1, 1});
        dirs.push_back({-1, -1});
        dirs.push_back({1, -1});
        dirs.push_back({-1, 1});
    }

    for (const cv::Point& dir : dirs) {
        std::cout << "SGM direction (" << dir.x << "," << dir.y << ")\n";
        aggregate_sgm_direction(data_cost, sum_cost, size, disp_count, dir.x, dir.y, p1, p2);
    }

    cv::Mat disparity(rows, cols, CV_32F, cv::Scalar(0.0f));
    for (int y = 0; y < rows; ++y) {
        float* disp_row = disparity.ptr<float>(y);
        for (int x = 0; x < cols; ++x) {
            const std::size_t base = cost_index(y, x, 0, cols, disp_count);
            int best_d = 0;
            std::int32_t best = sum_cost[base];
            for (int d = 1; d < disp_count; ++d) {
                const std::int32_t v = sum_cost[base + d];
                if (v < best) {
                    best = v;
                    best_d = d;
                }
            }
            disp_row[x] = static_cast<float>(best_d);
        }
    }

    disparity(cv::Rect(0, 0, std::min(max_disp + census_radius + 1, cols), rows)).setTo(0);
    disparity.rowRange(0, std::min(census_radius + block_radius, rows)).setTo(0);
    disparity.rowRange(std::max(0, rows - census_radius - block_radius), rows).setTo(0);
    return disparity;
}

static void save_colormap(const cv::Mat& src_32f,
                          const cv::Mat& valid_mask,
                          const fs::path& output,
                          double min_value,
                          double max_value) {
    if (max_value <= min_value) {
        cv::imwrite(output.string(), cv::Mat::zeros(src_32f.size(), CV_8UC3));
        return;
    }
    cv::Mat normalized;
    src_32f.convertTo(normalized, CV_8U,
                      255.0 / (max_value - min_value),
                      -255.0 * min_value / (max_value - min_value));
    normalized.setTo(0, ~valid_mask);
    cv::Mat colored;
    cv::applyColorMap(normalized, colored, cv::COLORMAP_TURBO);
    colored.setTo(cv::Scalar(0, 0, 0), ~valid_mask);
    cv::imwrite(output.string(), colored);
}

static void save_grayscale(const cv::Mat& src_32f,
                           const cv::Mat& valid_mask,
                           const fs::path& output,
                           double min_value,
                           double max_value,
                           bool invert) {
    if (max_value <= min_value) {
        cv::imwrite(output.string(), cv::Mat::zeros(src_32f.size(), CV_8U));
        return;
    }

    cv::Mat clipped;
    cv::min(src_32f, max_value, clipped);
    cv::max(clipped, min_value, clipped);

    cv::Mat gray;
    clipped.convertTo(gray, CV_8U,
                      255.0 / (max_value - min_value),
                      -255.0 * min_value / (max_value - min_value));
    if (invert) {
        gray = 255 - gray;
    }
    gray.setTo(0, ~valid_mask);
    cv::imwrite(output.string(), gray);
}

static cv::Mat disparity_to_depth(const cv::Mat& disparity, double fx, double baseline) {
    cv::Mat depth(disparity.size(), CV_32F, cv::Scalar(0.0f));
    for (int y = 0; y < disparity.rows; ++y) {
        const float* disp_row = disparity.ptr<float>(y);
        float* depth_row = depth.ptr<float>(y);
        for (int x = 0; x < disparity.cols; ++x) {
            const float d = disp_row[x];
            if (d > 0.5f) {
                depth_row[x] = static_cast<float>(fx * baseline / d);
            }
        }
    }
    return depth;
}

static void save_metric_depth_16u(const cv::Mat& depth_m, const fs::path& output) {
    cv::Mat depth_16u(depth_m.size(), CV_16U, cv::Scalar(0));
    for (int y = 0; y < depth_m.rows; ++y) {
        const float* src = depth_m.ptr<float>(y);
        ushort* dst = depth_16u.ptr<ushort>(y);
        for (int x = 0; x < depth_m.cols; ++x) {
            const float v = src[x];
            if (std::isfinite(v) && v > 0.0f) {
                const double scaled = std::min<double>(65535.0, std::round(v * 256.0));
                dst[x] = static_cast<ushort>(scaled);
            }
        }
    }
    cv::imwrite(output.string(), depth_16u);
}

static cv::Mat read_gt_depth_m(const std::string& path, cv::Size expected_size) {
    if (path.empty()) {
        return {};
    }
    cv::Mat raw = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (raw.empty()) {
        throw std::runtime_error("failed to read GT depth: " + path);
    }
    if (raw.size() != expected_size) {
        throw std::runtime_error("GT depth size does not match disparity size");
    }
    cv::Mat depth_m;
    raw.convertTo(depth_m, CV_32F, 1.0 / 256.0);
    return depth_m;
}

static Metrics evaluate(const cv::Mat& disparity,
                        const cv::Mat& depth_m,
                        const cv::Mat& gt_depth_m,
                        double fx,
                        double baseline,
                        cv::Mat* error_vis_src,
                        cv::Mat* error_valid_mask) {
    Metrics metrics;
    if (gt_depth_m.empty()) {
        return metrics;
    }

    cv::Mat error(depth_m.size(), CV_32F, cv::Scalar(0.0f));
    cv::Mat valid(depth_m.size(), CV_8U, cv::Scalar(0));

    double sum_depth_sq = 0.0;
    double sum_disp_abs = 0.0;
    double sum_abs_rel = 0.0;
    int d1_count = 0;

    for (int y = 0; y < depth_m.rows; ++y) {
        const float* disp_row = disparity.ptr<float>(y);
        const float* pred_depth_row = depth_m.ptr<float>(y);
        const float* gt_depth_row = gt_depth_m.ptr<float>(y);
        float* err_row = error.ptr<float>(y);
        uchar* valid_row = valid.ptr<uchar>(y);
        for (int x = 0; x < depth_m.cols; ++x) {
            const float gt_z = gt_depth_row[x];
            const float pred_z = pred_depth_row[x];
            const float pred_d = disp_row[x];
            if (gt_z <= 0.0f || pred_z <= 0.0f || pred_d <= 0.5f) {
                continue;
            }
            const double gt_d = fx * baseline / gt_z;
            const double disp_err = std::abs(static_cast<double>(pred_d) - gt_d);
            const double depth_err = std::abs(static_cast<double>(pred_z) - gt_z);

            ++metrics.valid_count;
            sum_depth_sq += depth_err * depth_err;
            sum_disp_abs += disp_err;
            sum_abs_rel += depth_err / gt_z;
            if (disp_err > 3.0 && disp_err / std::max(gt_d, 1e-6) > 0.05) {
                ++d1_count;
            }
            err_row[x] = static_cast<float>(depth_err);
            valid_row[x] = 255;
        }
    }

    if (metrics.valid_count > 0) {
        metrics.depth_mse = sum_depth_sq / metrics.valid_count;
        metrics.depth_rmse = std::sqrt(metrics.depth_mse);
        metrics.disp_epe = sum_disp_abs / metrics.valid_count;
        metrics.abs_rel = sum_abs_rel / metrics.valid_count;
        metrics.d1_all = static_cast<double>(d1_count) / metrics.valid_count;
    }

    if (error_vis_src != nullptr) {
        *error_vis_src = error;
    }
    if (error_valid_mask != nullptr) {
        *error_valid_mask = valid;
    }
    return metrics;
}

static void write_metrics(const Metrics& m,
                          const Args& args,
                          MatchMode mode,
                          double fx,
                          double baseline,
                          const fs::path& output) {
    std::ofstream f(output);
    if (!f) {
        throw std::runtime_error("failed to write metrics: " + output.string());
    }
    f << std::fixed << std::setprecision(6);
    f << "mode: " << mode_to_string(mode) << "\n";
    f << "algorithm: " << algorithm_label(mode) << "\n";
    f << "max_disp: " << args.max_disp << "\n";
    f << "census_radius: " << args.census_radius << "\n";
    f << "block_radius: " << args.block_radius << "\n";
    f << "block_diameter: " << (2 * args.block_radius + 1) << "\n";
    f << "sgm_directions: " << args.sgm_directions << "\n";
    f << "p1: " << args.p1 << "\n";
    f << "p2: " << args.p2 << "\n";
    f << "fx_px: " << fx << "\n";
    f << "baseline_m: " << baseline << "\n";
    f << "valid_count: " << m.valid_count << "\n";
    f << "depth_mse_m2: " << m.depth_mse << "\n";
    f << "depth_rmse_m: " << m.depth_rmse << "\n";
    f << "disp_epe_px: " << m.disp_epe << "\n";
    f << "d1_all: " << m.d1_all << "\n";
    f << "abs_rel: " << m.abs_rel << "\n";
}

int main(int argc, char** argv) {
    try {
        Args args = parse_args(argc, argv);
        MatchMode mode = parse_mode_name(args.mode);
        args.mode = mode_to_string(mode);
        args.sgm_directions = sgm_directions_for_mode(mode);

        if (args.max_disp < 1) {
            throw std::runtime_error("--max-disp must be positive");
        }
        if (args.block_radius < 0) {
            throw std::runtime_error("--block-radius must be non-negative");
        }
        if (args.p1 < 0 || args.p2 < 0 || args.p2 < args.p1) {
            throw std::runtime_error("expected 0 <= p1 <= p2");
        }

        int bits = 0;
        if (mode != MatchMode::RawBaseline) {
            bits = census_bit_count(args.census_radius);
            if (bits > 64) {
                throw std::runtime_error("--census-radius is too large for a 64-bit descriptor");
            }
        }

        cv::Mat left = cv::imread(args.left, cv::IMREAD_COLOR);
        cv::Mat right = cv::imread(args.right, cv::IMREAD_GRAYSCALE);
        if (left.empty()) {
            throw std::runtime_error("failed to read left image: " + args.left);
        }
        if (right.empty()) {
            throw std::runtime_error("failed to read right image: " + args.right);
        }
        if (left.size() != right.size()) {
            throw std::runtime_error("left/right rectified images must have the same size");
        }

        cv::FileStorage rect(args.rectification, cv::FileStorage::READ);
        if (!rect.isOpened()) {
            throw std::runtime_error("failed to read rectification file: " + args.rectification);
        }
        cv::Mat P1 = read_required_matrix(rect, "P1");
        cv::Mat P2 = read_required_matrix(rect, "P2");
        const double fx = P1.at<double>(0, 0);
        const double baseline = std::abs(P2.at<double>(0, 3) / P2.at<double>(0, 0));

        std::cout << "matching with max_disp=" << args.max_disp
                  << ", block=" << (2 * args.block_radius + 1) << "x"
                  << (2 * args.block_radius + 1) << "\n";
        std::cout << "mode: " << args.mode << " (" << algorithm_label(mode) << ")\n";

        CostVolumeResult cost_result;
        cv::Mat disparity;
        if (mode == MatchMode::RawBaseline) {
            std::cout << "using raw grayscale intensity cost\n";
            cv::Mat left_gray = to_gray8(left);
            cv::Mat right_gray = to_gray8(right);
            cost_result = build_intensity_cost_volume_and_initial_disparity(
                left_gray, right_gray, left.size(), args.max_disp, args.block_radius);
            disparity = cost_result.disparity;
        } else {
            std::cout << "preprocessing images\n";
            cv::Mat left_gray = preprocess_gray(left);
            cv::Mat right_gray = preprocess_gray(right);

            std::cout << "computing census descriptors (" << bits << " bits)\n";
            std::vector<std::uint64_t> census_left = census_transform(left_gray, args.census_radius);
            std::vector<std::uint64_t> census_right = census_transform(right_gray, args.census_radius);

            cost_result = build_cost_volume_and_initial_disparity(
                census_left, census_right, left.size(), args.max_disp, args.census_radius, args.block_radius);
            disparity = cost_result.disparity;

            if (args.sgm_directions > 0) {
                std::cout << "refining with SGM, directions=" << args.sgm_directions
                          << ", p1=" << args.p1
                          << ", p2=" << args.p2 << "\n";
                disparity = compute_disparity_sgm(cost_result.data_cost, left.size(), cost_result.disp_count,
                                                  args.max_disp, args.census_radius, args.block_radius,
                                                  args.sgm_directions, args.p1, args.p2);
            } else {
                std::cout << "skipping SGM; using WTA disparity\n";
            }
        }
        cv::Mat valid_disp = disparity > 0.5f;

        cv::Mat depth_m = disparity_to_depth(disparity, fx, baseline);
        cv::Mat valid_depth = depth_m > 0.0f;
        cv::Mat gt_depth_m = read_gt_depth_m(args.gt_depth, disparity.size());

        cv::Mat error_m;
        cv::Mat error_valid;
        Metrics metrics = evaluate(disparity, depth_m, gt_depth_m, fx, baseline, &error_m, &error_valid);

        fs::create_directories(args.out_dir);
        const fs::path out_dir(args.out_dir);

        double max_depth_vis = 80.0;
        if (!gt_depth_m.empty()) {
            double gt_max = 0.0;
            cv::minMaxLoc(gt_depth_m, nullptr, &gt_max, nullptr, nullptr, gt_depth_m > 0.0f);
            if (gt_max > 0.0) {
                max_depth_vis = std::min(120.0, gt_max);
            }
        }

        if (!gt_depth_m.empty()) {
            cv::Mat valid_gt_depth = gt_depth_m > 0.0f;
            cv::Mat depth_gt_masked = depth_m.clone();
            depth_gt_masked.setTo(0.0f, ~valid_gt_depth);
            save_grayscale(depth_gt_masked, valid_gt_depth, out_dir / "pred_depth.png",
                           0.0, max_depth_vis, false);
            save_grayscale(gt_depth_m, valid_gt_depth, out_dir / "gt_depth.png",
                           0.0, max_depth_vis, false);
        } else {
            save_grayscale(depth_m, valid_depth, out_dir / "pred_depth.png",
                           0.0, max_depth_vis, false);
        }

        write_metrics(metrics, args, mode, fx, baseline, out_dir / "metrics.txt");

        std::cout << "wrote: " << args.out_dir << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "fx: " << fx << " px\n";
        std::cout << "baseline: " << baseline << " m\n";
        std::cout << "valid GT pixels: " << metrics.valid_count << "\n";
        if (metrics.valid_count > 0) {
            std::cout << "depth RMSE: " << metrics.depth_rmse << " m\n";
            std::cout << "disparity EPE: " << metrics.disp_epe << " px\n";
            std::cout << "D1-all: " << metrics.d1_all << "\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }
    return 0;
}
